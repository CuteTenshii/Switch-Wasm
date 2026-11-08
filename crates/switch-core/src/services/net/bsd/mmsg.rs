//! `RecvMMsg` and `SendMMsg`: several messages in one buffer, which is also how libnx's
//! `recvmsg` and `sendmsg` reach the service. The buffer is libnx's `_serializeMmsg`: a `0x08`
//! byte, then per message `u32 namelen`, the name, `i32 iovlen`, per vector `u64 len` and its
//! bytes, `u32 controllen`, the control data, `i32 flags` and `i32 msg_len`. The reply rewrites
//! it with the lengths actually used, which libnx reads back in the same order.

use super::*;

/// The byte a serialized buffer starts with.
const MMSG_HEADER: u8 = 0x08;

/// Most messages one call carries, as libnx allows.
const MAX_MESSAGES: u32 = 0x20;

/// `MSG_TRUNC` in a message's flags: its datagram was longer than its vectors.
const BSD_MSG_TRUNC: i32 = 0x10;

/// One `mmsghdr` as the buffer carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Message {
    name: Vec<u8>,
    vectors: Vec<Vec<u8>>,
    control: Vec<u8>,
    flags: i32,
    length: i32,
}

/// A cursor over the serialized buffer; every read is bounds-checked.
struct Cursor<'a> {
    raw: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn bytes(&mut self, len: usize) -> Option<&[u8]> {
        let end = self.at.checked_add(len)?;
        let bytes = self.raw.get(self.at..end)?;
        self.at = end;
        Some(bytes)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.bytes(4)?.try_into().ok()?))
    }

    fn i32(&mut self) -> Option<i32> {
        Some(i32::from_le_bytes(self.bytes(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.bytes(8)?.try_into().ok()?))
    }
}

/// The first `count` messages of a serialized buffer, or `None` if it does not hold them.
fn parse(raw: &[u8], count: u32) -> Option<Vec<Message>> {
    if raw.first() != Some(&MMSG_HEADER) {
        return None;
    }
    let mut cursor = Cursor { raw, at: 1 };
    let mut messages = Vec::new();
    for _ in 0..count {
        let name_len = cursor.u32()? as usize;
        let name = cursor.bytes(name_len)?.to_vec();
        let vector_count = usize::try_from(cursor.i32()?).ok()?;
        let mut vectors = Vec::new();
        for _ in 0..vector_count {
            let len = usize::try_from(cursor.u64()?).ok()?;
            vectors.push(cursor.bytes(len)?.to_vec());
        }
        let control_len = cursor.u32()? as usize;
        let control = cursor.bytes(control_len)?.to_vec();
        let flags = cursor.i32()?;
        let length = cursor.i32()?;
        messages.push(Message {
            name,
            vectors,
            control,
            flags,
            length,
        });
    }
    Some(messages)
}

fn serialize(messages: &[Message]) -> Vec<u8> {
    let mut out = vec![MMSG_HEADER];
    for message in messages {
        out.extend_from_slice(&(message.name.len() as u32).to_le_bytes());
        out.extend_from_slice(&message.name);
        out.extend_from_slice(&(message.vectors.len() as i32).to_le_bytes());
        for vector in &message.vectors {
            out.extend_from_slice(&(vector.len() as u64).to_le_bytes());
            out.extend_from_slice(vector);
        }
        out.extend_from_slice(&(message.control.len() as u32).to_le_bytes());
        out.extend_from_slice(&message.control);
        out.extend_from_slice(&message.flags.to_le_bytes());
        out.extend_from_slice(&message.length.to_le_bytes());
    }
    out
}

/// Spread `data` over the vectors in order: each keeps what landed in it, no more than it held.
fn scatter(vectors: &mut [Vec<u8>], data: &[u8]) {
    let mut rest = data;
    for vector in vectors {
        let take = vector.len().min(rest.len());
        vector.clear();
        vector.extend_from_slice(&rest[..take]);
        rest = &rest[take..];
    }
}

impl Cpu {
    /// `RecvMMsg(fd, vlen, flags, timespec)`: up to `vlen` messages into the buffer's vectors.
    /// The first waits as a `recv` would; the rest take only what is already queued, as with
    /// `MSG_WAITFORONE`. The `timespec` is not honoured: libnx's `recvmsg` passes zero.
    pub(super) fn bsd_receive_messages(
        &mut self,
        tls: u32,
        fd: i32,
        count: u32,
        flags: u32,
    ) -> Result<()> {
        let Some((addr, size)) = self.ipc_output_buffer(tls, 0) else {
            return self.bsd_reply(tls, -1, BSD_EINVAL);
        };
        let raw = self.read_bytes(addr, size);
        let messages = (1..=MAX_MESSAGES)
            .contains(&count)
            .then(|| parse(&raw, count))
            .flatten();
        let Some(mut messages) = messages else {
            return self.bsd_reply(tls, -1, BSD_EINVAL);
        };
        let yielding = self.pending_yield;
        let mut received = 0;
        for (index, message) in messages.iter_mut().enumerate() {
            let room = message.vectors.iter().map(Vec::len).sum();
            let flags = if index == 0 {
                flags
            } else {
                flags | BSD_MSG_DONTWAIT
            };
            // Only the first message can wait, and nothing has been taken when it does.
            let Some(receipt) = self.bsd_receive_bytes(fd, room, flags) else {
                return Ok(());
            };
            if receipt.ret < 0 {
                if index == 0 {
                    self.bsd_end_wait(fd);
                    return self.bsd_reply(tls, -1, receipt.errno);
                }
                // What the first messages got is the answer, with no reason to yield.
                self.pending_yield = yielding;
                break;
            }
            scatter(&mut message.vectors, &receipt.data);
            let mut name = receipt.source.unwrap_or_default();
            name.truncate(message.name.len());
            message.name = name;
            message.control.clear();
            message.flags = if receipt.truncated { BSD_MSG_TRUNC } else { 0 };
            message.length = receipt.ret;
            received += 1;
            // End-of-file: nothing more will come.
            if receipt.ret == 0 {
                break;
            }
        }
        self.bsd_end_wait(fd);
        let reply = serialize(&messages[..received]);
        for (offset, &byte) in reply.iter().enumerate() {
            self.mem.write_u8(addr.wrapping_add(offset as u32), byte)?;
        }
        self.bsd_reply(tls, received as i32, 0)
    }

    /// `SendMMsg(fd, vlen, flags)`: each message's vectors, joined, to its name when it has one,
    /// as a `sendto` would. Stops at the first that fails; that failure is the answer only if
    /// it was the first message.
    pub(super) fn bsd_send_messages(&mut self, tls: u32, fd: i32, count: u32) -> Result<()> {
        let Some((addr, size)) = self.ipc_output_buffer(tls, 0) else {
            return self.bsd_reply(tls, -1, BSD_EINVAL);
        };
        let raw = self.read_bytes(addr, size);
        let messages = (1..=MAX_MESSAGES)
            .contains(&count)
            .then(|| parse(&raw, count))
            .flatten();
        let Some(mut messages) = messages else {
            return self.bsd_reply(tls, -1, BSD_EINVAL);
        };
        let mut sent = 0;
        for (index, message) in messages.iter_mut().enumerate() {
            let data = message.vectors.concat();
            let address = (!message.name.is_empty()).then_some(message.name.as_slice());
            let (ret, errno) = self.bsd_send_bytes(fd, data, address);
            if ret < 0 {
                if index == 0 {
                    return self.bsd_reply(tls, -1, errno);
                }
                break;
            }
            message.length = ret;
            sent += 1;
        }
        let reply = serialize(&messages[..sent]);
        for (offset, &byte) in reply.iter().enumerate() {
            self.mem.write_u8(addr.wrapping_add(offset as u32), byte)?;
        }
        self.bsd_reply(tls, sent as i32, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;
    use crate::kernel::ipc::testing::*;

    const BUFFER: u32 = SCRATCH + 0x200;
    const BUFFER_SIZE: u32 = 0x1000;

    /// A message as libnx's `_serializeMmsg` lays it out before a call.
    fn message(name: &[u8], vectors: &[&[u8]]) -> Message {
        Message {
            name: name.to_vec(),
            vectors: vectors.iter().map(|v| v.to_vec()).collect(),
            control: Vec::new(),
            flags: 0,
            length: 0,
        }
    }

    /// Room for a receive: vectors of zeros of these lengths, and a name of `name_len` zeros.
    fn room(name_len: usize, lengths: &[usize]) -> Message {
        Message {
            name: vec![0; name_len],
            vectors: lengths.iter().map(|&len| vec![0; len]).collect(),
            control: Vec::new(),
            flags: 0,
            length: 0,
        }
    }

    /// Issue command `cmd` with `words` and `messages` serialized into the buffer; the reply
    /// and the messages it wrote back, `count` of them.
    fn call(
        cpu: &mut Cpu,
        cmd: u32,
        words: &[u32],
        messages: &[Message],
        count: usize,
    ) -> ((i32, i32), Vec<Message>) {
        let mut raw = serialize(messages);
        raw.resize(BUFFER_SIZE as usize, 0);
        place(cpu, BUFFER, &raw);
        let payload: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        write_map_buffer_request(cpu, cmd, &payload, BUFFER, BUFFER_SIZE, false);
        cpu.bsd_request(TLS, 9, Some(cmd)).unwrap();
        let back = cpu.read_bytes(BUFFER, BUFFER_SIZE);
        (
            bsd_result(cpu),
            parse(&back, count as u32).unwrap_or_default(),
        )
    }

    /// `RecvMMsg(fd, vlen, flags, 0, timespec 0)`.
    fn receive(
        cpu: &mut Cpu,
        fd: i32,
        messages: &[Message],
        count: usize,
    ) -> ((i32, i32), Vec<Message>) {
        let words = [fd as u32, messages.len() as u32, 0, 0, 0, 0, 0, 0];
        call(cpu, 29, &words, messages, count)
    }

    fn pair() -> (Cpu, i32, i32) {
        let (mut cpu, listener) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x2000).unwrap();
        let (client, server) = connected_pair(&mut cpu, listener);
        (cpu, client, server)
    }

    #[test]
    fn the_buffer_is_libnx_s_layout() {
        let mut expected = vec![0x08];
        expected.extend_from_slice(&2u32.to_le_bytes());
        expected.extend_from_slice(b"ab");
        expected.extend_from_slice(&2i32.to_le_bytes());
        expected.extend_from_slice(&1u64.to_le_bytes());
        expected.push(b'x');
        expected.extend_from_slice(&0u64.to_le_bytes());
        expected.extend_from_slice(&0u32.to_le_bytes());
        expected.extend_from_slice(&7i32.to_le_bytes());
        expected.extend_from_slice(&9i32.to_le_bytes());
        let mut sent = message(b"ab", &[b"x", b""]);
        sent.flags = 7;
        sent.length = 9;
        assert_eq!(serialize(std::slice::from_ref(&sent)), expected);
        assert_eq!(parse(&expected, 1), Some(vec![sent]));
        assert_eq!(parse(&expected[..expected.len() - 1], 1), None, "cut short");
        assert_eq!(parse(&expected[1..], 1), None, "no header byte");
    }

    #[test]
    fn recvmsg_spreads_a_stream_over_its_vectors() {
        let (mut cpu, client, server) = pair();
        send_on(&mut cpu, client, b"hello world");
        let (result, back) = receive(&mut cpu, server, &[room(16, &[5, 20])], 1);
        assert_eq!(result, (1, 0));
        assert_eq!(back[0].vectors, vec![b"hello".to_vec(), b" world".to_vec()]);
        assert_eq!(back[0].length, 11);
        assert!(back[0].name.is_empty(), "a stream has no source to name");
        assert_eq!(back[0].flags, 0);
    }

    #[test]
    fn recvmsg_on_an_empty_blocking_socket_waits() {
        let (mut cpu, _client, server) = pair();
        receive(&mut cpu, server, &[room(16, &[8])], 0);
        assert!(cpu.pending_retry.take().is_some());
    }

    #[test]
    fn recvmmsg_takes_the_datagrams_there_are_and_marks_a_cut_one() {
        let (mut cpu, fd) = bsd_socket(BSD_SOCK_DGRAM);
        cpu.mem.map_zero(SCRATCH, 0x2000).unwrap();
        cpu.set_network(true);
        let a = cpu.net.synthesize("a.example").unwrap();
        let mut destination = loopback_sockaddr(53);
        destination[4..8].copy_from_slice(&a);
        let words = [fd as u32, 1, 0];
        call(&mut cpu, 30, &words, &[message(&destination, &[b"q"])], 1);
        cpu.take_net_commands();
        cpu.deliver_net_events([
            NetEvent::Data {
                stream: 1,
                data: b"first".to_vec(),
            },
            NetEvent::Data {
                stream: 1,
                data: b"second, longer".to_vec(),
            },
        ]);

        let rooms = [room(16, &[8]), room(16, &[8]), room(16, &[8])];
        let (result, back) = receive(&mut cpu, fd, &rooms, 2);
        assert_eq!(result, (2, 0), "two of three: the third does not wait");
        assert!(cpu.pending_retry.is_none());
        assert_eq!(back[0].vectors, vec![b"first".to_vec()]);
        assert_eq!(super::super::sockaddr_in(&back[0].name), Some((a, 53)));
        assert_eq!(back[0].flags, 0);
        assert_eq!(back[1].vectors, vec![b"second, ".to_vec()]);
        assert_eq!(back[1].length, 8);
        assert_eq!(back[1].flags, BSD_MSG_TRUNC);
    }

    #[test]
    fn sendmmsg_sends_each_message_to_its_name() {
        let (mut cpu, fd) = bsd_socket(BSD_SOCK_DGRAM);
        cpu.mem.map_zero(SCRATCH, 0x2000).unwrap();
        cpu.set_network(true);
        let a = cpu.net.synthesize("a.example").unwrap();
        let b = cpu.net.synthesize("b.example").unwrap();
        let to = |ip: [u8; 4]| {
            let mut raw = loopback_sockaddr(53);
            raw[4..8].copy_from_slice(&ip);
            raw
        };
        let messages = [message(&to(a), &[b"he", b"llo"]), message(&to(b), &[b"x"])];
        let (result, back) = call(&mut cpu, 30, &[fd as u32, 2, 0], &messages, 2);
        assert_eq!(result, (2, 0));
        assert_eq!((back[0].length, back[1].length), (5, 1));
        let sends: Vec<_> = cpu
            .take_net_commands()
            .into_iter()
            .filter_map(|command| match command {
                NetCommand::Send { stream, data } => Some((stream, data)),
                _ => None,
            })
            .collect();
        assert_eq!(sends, vec![(1, b"hello".to_vec()), (2, b"x".to_vec())]);
    }

    #[test]
    fn sendmsg_on_a_stream_reaches_the_peer() {
        let (mut cpu, client, server) = pair();
        let (result, _) = call(
            &mut cpu,
            30,
            &[client as u32, 1, 0],
            &[message(&[], &[b"ab", b"c"])],
            1,
        );
        assert_eq!(result, (1, 0));
        assert_eq!(recv_on(&mut cpu, server, 8), ((3, 0), b"abc".to_vec()));
    }

    #[test]
    fn a_failure_on_the_first_message_is_the_answer() {
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x2000).unwrap();
        let (result, _) = call(
            &mut cpu,
            30,
            &[fd as u32, 1, 0],
            &[message(&[], &[b"a"])],
            0,
        );
        assert_eq!(result, (-1, BSD_ENOTCONN));
        let (result, _) = call(&mut cpu, 30, &[fd as u32, 0, 0], &[], 0);
        assert_eq!(result, (-1, BSD_EINVAL), "no messages");
    }
}
