# Relay protocol

How the browser reaches the Internet when the guest opens a TCP connection, a UDP socket or a listener. The page opens one WebSocket to a relay (`crates/switch-relay`, a native binary the user runs) and multiplexes every guest stream over it. The relay does DNS and the real sockets; the guest's TLS stays in the emulator, so the relay only ever sees ciphertext.

Nothing here is needed to play. With no relay URL set the network stays empty.

## Using it

Run `switch-relay` (no arguments), open the Network tab, and turn it on. The tab shows the relay's state and the guest's connections, UDP sockets and listeners: peer or port, state and bytes each way. The address defaults to `ws://127.0.0.1:18787` and is remembered.

## Transport

- One WebSocket, binary messages only. One message is one frame; there is no framing inside a message.
- A stream is one guest TCP connection, one guest UDP socket (which talks to any number of peers), or one guest listener. The client picks the ids of the streams it opens (`u32`, below `0x80000000`, never reused within a WebSocket); the relay numbers the connections a listener accepts from `0x80000000` up.
- All integers are little-endian.
- A relay that gets a malformed frame, an unknown op, a stream id it does not expect, or a frame larger than the limits below closes the WebSocket with code 1002. A text message closes it with 1003.

## Frame

```
u8 op | u32 stream | payload
```

`HELLO` and `ALIVE` have no stream; their stream field is 0.

A peer is `u8 kind` and then: 0, an IPv4 address (4 bytes); 1, a name (`u8 len`, UTF-8 bytes); 2, an IPv6 address (16 bytes).

### Client to relay

| Op | Name | Payload |
| --- | --- | --- |
| `0x01` | `CONNECT` | `u16 port`, `u32 window`, peer. A TCP connection |
| `0x02` | `DATA` | the bytes, at most 16 KiB |
| `0x03` | `SHUTDOWN_WRITE` | none. The guest is done sending; the peer may still answer |
| `0x04` | `CLOSE` | none. Drop the stream; the relay may forget the id |
| `0x05` | `CREDIT` | `u32 bytes`. The guest has read this many more bytes |
| `0x06` | `UDP_OPEN` | `u16 port` (the guest's, 0 for any), `u32 window`. A UDP socket |
| `0x07` | `SEND_TO` | `u16 port`, peer, then one datagram of at most 65,507 bytes |
| `0x08` | `LISTEN` | `u16 port`, `u32 window` for each connection it accepts |

### Relay to client

| Op | Name | Payload |
| --- | --- | --- |
| `0x80` | `HELLO` | `u8 version` (3), `u16 max_streams` |
| `0x81` | `CONNECTED` | none. The connection, socket or listener is open |
| `0x82` | `FAILED` | `u8 reason`: 0 refused, 1 unreachable, 2 timed out, 3 name not found, 4 denied by policy, 5 port in use |
| `0x83` | `DATA` | the bytes, at most 16 KiB |
| `0x84` | `CLOSED` | none. The peer finished sending; the guest may still send |
| `0x85` | `ALIVE` | none. The relay is still there; see [Keepalive](#keepalive) |
| `0x86` | `RESET` | none. The connection was reset or broke: nothing more either way |
| `0x87` | `SENT` | `u32 bytes`. The relay wrote this many more of the stream's bytes to the peer |
| `0x88` | `RECEIVED_FROM` | `u16 port`, peer, then one datagram |
| `0x89` | `INCOMING` | `u32 listener`, `u16 port`, peer. A connection to `listener`, as the new stream in the frame's stream field |

`HELLO` is the relay's first message. A client that does not understand the version closes the WebSocket. The page understands version 3 only; an older relay is refused and the Network tab says no relay answered.

## Streams

1. The client sends `CONNECT`, `UDP_OPEN` or `LISTEN`. The relay answers `CONNECTED` or `FAILED`, once. A stream that fails is over; the client sends no `CLOSE`.
2. On a TCP stream both sides send `DATA`. `CLOSED` means the peer finished sending, and the guest may go on sending. `RESET` means the connection is gone both ways; it is the last frame for the stream, and the relay forgets the id.
3. On a UDP stream the client sends `SEND_TO` and the relay `RECEIVED_FROM`. One stream sends every datagram from one host port, whatever the destination.
4. On a listener the relay sends `INCOMING` for each connection it accepts, then treats the new stream as a TCP stream the client opened.
5. `CLOSE` from the client ends a stream; frames for it that are still in flight are dropped on both ends.
6. A relay that already forgot a stream ignores `DATA`, `SEND_TO`, `SHUTDOWN_WRITE`, `CLOSE` and `CREDIT` for it. It never answers an unknown id.

Names are resolved by the relay, to IPv4 and IPv6. A TCP connection tries every allowed address, the two families taking turns from whichever the resolver put first, each attempt 250 ms ahead of the next (happy eyeballs, RFC 8305); the lookup and all attempts share 10 seconds. A name that does not resolve is `FAILED` with reason 3, so the guest sees it at `connect`, not at `getaddrinfo` (the console hands out a synthetic address per name, see `docs/services.md`). A UDP datagram to a name goes to its first allowed IPv4 address, or its first IPv6 one if it has none: a datagram cannot tell that a family's path is broken. The relay keeps what a name resolved to for 60 seconds per socket.

The guest has only IPv4. An IPv6 peer reaches it as a synthetic address standing for the address's text, which the relay resolves back to it. A datagram from a host the guest reached by name comes back as from that name, so the guest sees the reply come from the address it sent to.

## Flow control

A browser WebSocket cannot slow a sender down, so neither side may send more than the other has room for.

- `CONNECT`, `UDP_OPEN` and `LISTEN` carry `window`, the bytes the relay may send on that stream before the client reads any.
- The relay keeps a count per stream. Each `DATA` or `RECEIVED_FROM` it sends lowers it; each `CREDIT` raises it. At zero it stops reading from the peer (TCP) or drops datagrams (UDP) until credit arrives.
- The client sends `CREDIT` as the guest's `recv` drains what was delivered, batched, not per byte.
- The other way, the relay sends `SENT` after writing each `DATA` to the peer. The guest keeps at most 256 KiB written and not yet acknowledged per connection (`docs/services.md`), so a slow peer slows the guest instead of filling the relay's memory. TLS streams are outside this: the relay acknowledges ciphertext, and the page does not pass those acknowledgements on.
- Datagrams the guest sends have no window: a UDP sender is not slowed down by its network either.

## Keepalive

A connection that has not finished the WebSocket upgrade within 10 seconds is closed. The question about an unknown origin comes after the upgrade, so this does not limit how long it waits for an answer.

The relay sends a WebSocket Ping every 15 seconds. A page that sends nothing for 45 seconds, not even the Pong a browser returns on its own, is gone: the relay drops the WebSocket without a close frame and closes every stream it held.

The browser answers pings without showing them to the page, so the relay also sends `ALIVE` with each one. A page that hears nothing from the relay for 45 seconds treats the link as lost, as if the socket had closed: what was opening fails, what was open is reset, and the Network tab shows the relay unreachable. Time the page's worker did not run, such as a background tab or a sleeping machine, does not count, since the frames sent meanwhile have not been delivered yet.

A quiet stream on a page that still answers is left open, however long it stays quiet. The guest may hold a connection on purpose, and a page that answers is still there to close it.

## Relay policy

The relay is a proxy to anywhere its host can reach, and any web page can open a WebSocket to a loopback port, so the defaults are closed and need no configuration.

- It listens on `127.0.0.1:18787`, and on `[::1]:18787` where the host has IPv6. The page connects to the first by default.
- It trusts pages by `Origin`, which a browser always sends and a page cannot forge. `http://localhost:8000` and `https://tenshii.moe` are trusted. Any other origin is asked about on the relay's terminal, once per run; with no terminal it is refused. A client with no `Origin` is not a browser and is refused.
- An untrusted origin gets a WebSocket close with code 1008 before `HELLO`.
- It refuses private, loopback, link-local, carrier-grade NAT, multicast and reserved destinations, IPv4 and IPv6 (`FAILED`, reason 4; a datagram to one is dropped). This is checked on the resolved address, not the name. An IPv6 address that carries an IPv4 one (IPv4-mapped, NAT64's `64:ff9b::/96`, 6to4) is judged by the IPv4 address.
- Inbound traffic is off by default. A UDP socket only hears from hosts it has sent to, on any of their ports, as a NAT that filters by address does, and gets whatever host port is free. `LISTEN` is refused (reason 4).
- `--allow-inbound` lets guests host: a UDP socket gets the guest's own port when it is free (any port otherwise) and hears from anyone, and `LISTEN` opens that port on every interface, IPv4 and IPv6, failing with reason 5 when it is taken. Reaching it from the Internet is up to the user's router.
- `--port`, `--allow-origin`, `--allow-private` and `--allow-inbound` exist for the cases that need them.
- Limits: 256 streams per WebSocket (an open over it is reason 4; an accepted connection over it is dropped), a `window` capped at 1 MiB, 16 KiB of `DATA` per frame, 65,507 bytes per datagram, and per WebSocket message a full datagram with its header.

## Not covered

- Broadcast, multicast and LAN discovery. There is no LAN behind a relay to reach.
- ICMP errors: a UDP socket sends to every peer from one unconnected host socket, which the host does not tell about unreachable ports.
- Encryption between page and relay beyond `wss://`. TLS between guest and server is end to end.
