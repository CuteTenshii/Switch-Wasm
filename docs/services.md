# Services

What each system service must answer, and why. Extracted from AGENTS.md, which
keeps the *marshalling* rules (`## IPC`); this is the per-service inventory.
`svc.rs` dispatches by session name; each module owns its own constants, state
and tests.

## The services

- **`nvdrv`** — the real `INvDrvServices`, dispatched into `gpu::nvdrv`.
- **`hid`** is the *negotiation*, not the input: the data lives in a 256 KiB
  shared memory region the guest reads with no IPC per frame
  (`CreateAppletResource` → `GetSharedMemoryHandle`). The `Set*`/`Get*` pairs
  must read back. Vibration returns through `SendVibrationValue` →
  `Cpu::vibration` → the Gamepad API's `dual-rumble`.
- **`pl:u`/`pl:s`** — the shared fonts; see below.
- **`lm`** carries a title's `NN_LOG` output: a 0x18-byte LogPacket header then
  TLV chunks (key 2 message, key 6 module) in a map-alias buffer, split across
  packets with `flags` bit 0 head / bit 1 tail. Retail builds often compile
  logging out, so an empty log is not evidence of a bug.
- **`fatal:u`** carries the `Result` that stopped a process.
- **`erpt`** journals *context*, one record per category (`ErrorInfo`,
  `GpuCrashInfo`, `ThermalInfo`), resubmitted rather than appended to;
  `CreateReport` writes it out whole. Nothing persists and nothing uploads.
  `IManager`'s report-created event is the one event here that genuinely fires.
- **`ssl`** — contexts and options are real; **`CreateConnection` deliberately
  reports unimplemented** rather than handing back a connection that can never
  connect.
- **`bsd`** models a link that is up and a network where nothing answers: local
  operations succeed, anything needing a peer fails at once with a definite
  errno (`ECONNREFUSED`, `ENOTCONN`/`ENETUNREACH`, `EAGAIN`). **Errnos are
  FreeBSD's** (`EAGAIN` is 35) and `fcntl`'s flags are stored verbatim. A
  `poll` *with* a timeout asks for a reschedule (`Cpu::pending_yield`) before
  returning zero, or a poll loop owns the CPU forever.
  **A datagram socket needs no peer**, and none of the three things it does
  may claim otherwise — the errno that says the link is gone contradicts the
  `nifm` that just said it was up. A `SendTo` naming an `AF_INET` destination
  **is sent**: a link that is up hands the datagram over and reports the byte
  count without waiting for anyone, and "nothing answers" is a thing that
  happens to the *reply*. A read reports `EAGAIN` and reschedules — nothing
  has arrived *yet* — rather than `ENETUNREACH`. And `select`/`poll` call it
  **writable**, or a caller that waits for readiness before sending never
  sends. Refusing any of them describes an interface that is *down*, which is
  a different console: RakNet's `BindShared` sends a test datagram to the
  address it just bound and reads a failed send as `BR_FAILED_SEND_TEST`, so
  `ENETUNREACH` there failed every `RakPeerInterface::Startup` — and
  Minecraft answers that by destroying its peer, nulling its pointer to it
  and calling through it anyway. A datagram with *no* destination, and a
  stream socket with no connection, fail as before.
- **`sfdnsres`** — `EAI_NONAME` / `HOST_NOT_FOUND`, the *definitive* failure
  rather than try-again, in the **first** word of `SfdnsresRequestResults`.
- **`pctl`** reports the console unrestricted. Watch the direction:
  `Confirm*`/`Check*Permission` reply with a bare `Result` where success *is*
  permitted, `IsRestriction*` is `false`, `IsFreeCommunicationAvailable`/
  `IsStereoVisionPermitted` are `true`.
- **`acc`** — exactly one user, always signed in, uid `ACCOUNT_UID` (nonzero;
  0 is the "no user" sentinel). `acc:u0` and `acc:u1`/`acc:su` share 0..=51 but
  **diverge from 100 up**, so those arms dispatch on the service name. The
  nickname is real state; `LoadImage` returns a real JPEG.
- **`apm`** must *agree* with `am`; `GetPerformanceConfiguration` returns what
  `Set*` was last handed (defaults nonzero — 0 is `Invalid`).
- **`ts`** reports the SoC and PCB sensors at an idle reading. `MilliC` is
  `GetTemperature` × 1000 and both sit inside `GetTemperatureRange`.
  **`ISession` is a different interface from its server** — its
  `GetTemperature` is command 4, the same id as the server's `OpenSession`. The
  device code's **high byte** picks the sensor (`0x41…` SoC, `0x43…` PCB).
- **`set:sys`** is a **store**, not a table of answers: the `Get`/`Set` pairs
  read and write one block, and it is kept in system save data
  `8000000000000050` — so it persists, through the same host flush that
  persists a title's save. A setting that has two homes has none: `nfc:sys`
  and `btm:sys` read the radio flags from here, `set`'s language and region
  are these fields, and `am`'s desired language and keyboard layout are too.
  `GetSettingsItemValue` is the firmware's separate key/value table; an item
  that is not in it is **refused** (`ResultSettingsItemNotFound`), because a
  caller reads the size back and then that many bytes.
  `GetFirmwareVersion`/`2` are **not cosmetic**: libnx seeds `hosversionGet()`
  from them and everything version-gated branches on that.
- **`csrng`** fills from `Cpu::next_random_u64` (splitmix64) — not a CSPRNG,
  but the generic reply left the buffer untouched, which is non-random *and*
  undetectably so.
- **`spl:`** — an Icosa retail console, not in debug mode. Atmosphère's
  extensions at 65000+ answer zero, i.e. "no CFW", which is true.
- **`pdm:qry`** — a console nothing has been played on.
- **`pm:*`** are four interfaces on four names. `pm`'s process id must equal
  `svcGetProcessId`'s; `pm:info`'s program id defaults to the Album applet's.
- **`pcv`/`clkrst`** are the same manager either side of 8.0.0, numbered **by
  an offset**: a `clkrst` device code is `0x40000000 + module + 1`. A rate a
  guest sets reads back.
- **`fsp-srv`**'s `DisableAutoSaveDataCreation` (1003) is accepted and
  deliberately **not** honoured — saves are created on open.

## What the Home Menu opens that homebrew never does

`lbl`, `audctl`, `nfc:sys`, `btm:sys`, `ldn:m`, `lp2p:m`, `ovln:*`, `olsc:s`,
`friend:*`, `news:*`, `bcat:*`, `notif:*`. Most are a creator plus the objects
it creates (`olsc:s` is five deep), and a fabricated object id is not callable,
so each sub-interface gets a name of its own (`Cpu::ipc_interface`) in
`svc.rs`'s dispatch.

**The answer is an empty console, not a broken one.** No friends, news, BCAT,
cloud saves, local network, NFC or paired gamepad — every one a state a real
console reaches, so callers already have a path for it; a *failure* puts them
on the path built for hardware that broke. None of these events ever signal.
The settings among them (`backlight`, `audio_control`, `notif_alarms`, …) are
stored, not answered — one caller writes, another reads back.
