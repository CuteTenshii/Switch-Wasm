# Relay protocol

How the browser reaches the Internet when the guest opens a TCP or UDP connection. The page opens one WebSocket to a relay (`crates/switch-relay`, a native binary the user runs) and multiplexes every guest stream over it. The relay does DNS and the real sockets; the guest's TLS stays in the emulator, so the relay only ever sees ciphertext.

Nothing here is needed to play. With no relay URL set the network stays empty.

## Using it

Run `switch-relay` (no arguments), open the Network tab, and turn it on. The tab shows the relay's state and the guest's connections: host, port, state and bytes each way. The address defaults to `ws://127.0.0.1:18787` and is remembered.

## Transport

- One WebSocket, binary messages only. One message is one frame; there is no framing inside a message.
- A stream is one guest connection: a TCP connection, or a UDP socket paired with one remote address. The client picks the stream id (`u32`, never reused within a WebSocket).
- All integers are little-endian.
- A relay that gets a malformed frame, an unknown op, or a stream id it does not expect closes the WebSocket with code 1002.

## Frame

```
u8 op | u32 stream | payload
```

`HELLO` has no stream; its stream field is 0.

### Client to relay

| Op | Name | Payload |
| --- | --- | --- |
| `0x01` | `CONNECT` | `u8 proto` (0 TCP, 1 UDP), `u16 port`, `u32 window`, then the peer: `u8 kind` 0 = IPv4 (4 bytes), 1 = name (`u8 len`, UTF-8 bytes) |
| `0x02` | `DATA` | the bytes; for UDP, exactly one datagram |
| `0x03` | `SHUTDOWN_WRITE` | none. The guest is done sending; the peer may still answer |
| `0x04` | `CLOSE` | none. Drop the stream; the relay may forget the id |
| `0x05` | `CREDIT` | `u32 bytes`. The guest has read this many more bytes |

### Relay to client

| Op | Name | Payload |
| --- | --- | --- |
| `0x80` | `HELLO` | `u8 version` (1), `u16 max_streams` |
| `0x81` | `CONNECTED` | none |
| `0x82` | `FAILED` | `u8 reason`: 0 refused, 1 unreachable, 2 timed out, 3 name not found, 4 denied by policy |
| `0x83` | `DATA` | the bytes; for UDP, exactly one datagram |
| `0x84` | `CLOSED` | none. The peer finished sending or the connection was reset; nothing more arrives on this stream |

`HELLO` is the relay's first message. A client that does not understand the version closes the WebSocket.

## Streams

1. The client sends `CONNECT`. The relay answers `CONNECTED` or `FAILED`, once. A stream that fails is over; the client sends no `CLOSE`.
2. After `CONNECTED`, both sides send `DATA`. UDP streams are connected: the relay sends the guest's datagrams to the one remote address and forwards only that address's replies.
3. `CLOSED` from the relay ends the receiving side. `CLOSE` from the client ends the stream; frames for it that are still in flight are dropped on both ends.
4. A relay that already forgot a stream ignores `DATA`, `SHUTDOWN_WRITE`, `CLOSE` and `CREDIT` for it. It never answers an unknown id.

Names are resolved by the relay, to IPv4 only: the guest has no IPv6. A name that does not resolve is `FAILED` with reason 3, so the guest sees it at `connect`, not at `getaddrinfo` (the console hands out a synthetic address per name, see `docs/services.md`).

## Flow control

A browser WebSocket cannot slow a sender down, so the relay must not send more than the guest has room for.

- `CONNECT` carries `window`, the bytes the relay may send on that stream before the client reads any.
- The relay keeps a count per stream. Each `DATA` it sends lowers it; each `CREDIT` raises it. At zero it stops reading from the peer (TCP) or drops datagrams (UDP) until credit arrives.
- The client sends `CREDIT` as the guest's `recv` drains what was delivered, batched, not per byte.

Client to relay needs no window: the guest writes at its own pace, and the WebSocket's own buffering is enough.

## Relay policy

The relay is a proxy to anywhere its host can reach, and any web page can open a WebSocket to a loopback port, so the defaults are closed and need no configuration.

- It listens on `127.0.0.1:18787`. The page connects there by default.
- It trusts pages by `Origin`, which a browser always sends and a page cannot forge. `http://localhost:8000` and `https://tenshii.moe` are trusted. Any other origin is asked about on the relay's terminal, once per run; with no terminal it is refused. A client with no `Origin` is not a browser and is refused.
- An untrusted origin gets a WebSocket close with code 1008 before `HELLO`.
- It refuses private, loopback, link-local, carrier-grade NAT, multicast and reserved destinations (`FAILED`, reason 4). This is checked on the resolved address, not the name.
- `--port`, `--allow-origin` and `--allow-private` exist for the rare case that needs them.
- Limits: 256 streams per WebSocket (a `CONNECT` over it is reason 4), a `window` capped at 1 MiB, and at most 16 KiB of `DATA` per frame.

## Not covered

- IPv6, and UDP sockets that talk to many peers over one stream (the client opens one stream per destination).
- Listening sockets: the guest cannot accept connections from outside.
- Encryption between page and relay beyond `wss://`. TLS between guest and server is end to end.
