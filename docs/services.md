# Services

What each system service must answer, and why. `svc.rs` dispatches by session name; each module owns its constants, handlers and tests, while service state lives on `Process` (per process) or `Machine` (per console).

## The services

- **`sm:`** - `GetService` (1) hands out a handle for any name, an empty one included; a name with no implementation then gets the fabricated replies (see `kernel/ipc.rs` below). Commands it does not implement are refused, Atmosphère's `HasService` (65100) included.
- **`nvdrv`** - the real `INvDrvServices`, dispatched into `gpu::nvdrv`. Each fd belongs to the process that opened it: another process's ioctl or close is refused, and a process's files close when it ends, with their channels and address spaces.
- **`hid`** is the *negotiation*, not the input: the data lives in a 256 KiB shared memory region the guest reads with no IPC per frame (`CreateAppletResource` → `GetSharedMemoryHandle`). The region is one console-wide `SharedPages` block that every process maps, so input is written once and the Home Menu and a running applet read the same pad. The `Set*`/`Get*` pairs must read back. Vibration goes `SendVibrationValue` → `Cpu::vibration` → the Gamepad API's `dual-rumble`.
- **`pl:u`/`pl:s`** - the shared fonts; see `pl.rs` below.
- **`lm`** carries a title's `NN_LOG` output: a 0x18-byte LogPacket header then TLV chunks (key 2 message, key 6 module), split across packets with `flags` bit 0 head / bit 1 tail. Retail builds often compile logging out, so an empty log is not evidence of a bug.
- **`fatal:u`** carries the `Result` that stopped a process. Each process keeps its report (`Cpu::guest_fatal` reads the running one's) after the diagnostic is drained, so the browser can treat the booted title's later `ExitProcess` as a crash rather than a clean halt. A library applet process that reports and then exits still ends as an exit, so its caller gets a result.
- **`erpt`** journals *context*, one record per category (`ErrorInfo`, `GpuCrashInfo`, ...), replaced on resubmission rather than appended; `CreateReport` writes the journal out whole. Nothing persists and nothing uploads. `IManager`'s event fires when a report is created.
- **`ssl`** - contexts and options are real; **`CreateConnection` deliberately reports unimplemented** rather than handing back a connection that can never connect.
- **`bsd`** models a link that is up and a network where nothing answers: loopback within the process works, anything needing an outside peer fails at once with a definite FreeBSD errno. Datagram sockets are the exception; see `net.rs` below.
- **`ns`** reports a console with nothing installed: `ListApplicationRecord` (app manager cmd 0, read-only record cmd 3) answers zero records with the count written, and `HasApplicationRecord` is `false`.
- **`sfdnsres`** - every lookup fails definitively; see `net.rs` below.
- **`nsd`** - `Resolve`/`ResolveEx` replace `%` with the production environment `lp1` and succeed; other commands are refused.
- **`usb:ds`/`usb:hs`** - no cable attached and nothing plugged in; see `usb.rs` below.
- **`pctl`** reports the console unrestricted. Watch the direction: `Confirm*`/`Check*Permission` reply with a bare `Result` where success *is* permitted, `IsRestriction*` is `false`, `IsFreeCommunicationAvailable`/`IsStereoVisionPermitted` are `true`.
- **`acc`** - up to eight users, all signed in, set by the host with `Cpu::set_users` before a title starts, with one of them playing. "Who is playing" questions (`GetLastOpenedUser`, `TrySelectUserWithoutInteraction`, `ListOpenUsers`, the preselected-user launch parameter) all name that user; `ListAllUsers` names everyone. `IProfile` answers for the user it was opened for. `IProfileEditor` stores are kept and reported to the host.
- **`fsp-srv` saves** are keyed by save id and the `SaveDataAttribute` uid: each user has their own save of a title, and system and device saves (uid 0) are shared. `ISaveDataInfoReader` lists the saves the session holds (every stored save is restored at boot): a save with a uid is an account save, a zero uid with the top id bit set a system save, any other a device save. Cache storage is never listed.
- **`am`** is the applet manager every title opens first, and `nnSdk` answers a refused command from it with an `svcBreak`, so a refusal ends the boot there. Its shapes matter more than its answers: `GetGpuErrorDetectedSystemEvent` (130) must hand back a real event, or `nn::oe::Initialize` aborts; an event nothing here can fire (210, the one behind the exit-request flow) is handed out and never signalled. `PopLaunchParameter` is a **pop**: a kind nobody left fails, and one that was left is handed over once. Gameplay recording (66, 67) and the copyright notice drawn over screenshots (100-102) are accepted and do nothing, since nothing here captures; Nintendo Switch Sports aborts on the copyright buffer setup otherwise.
- **`vi`** is one display with one layer. `SetLayerScalingMode` scales nothing but checks the mode the way the service does: `ScaleToWindow` (2) and `PreserveAspectRatio` (4) succeed, the other known modes are not supported, and anything above them fails. Mario Kart 8 Deluxe sets one at boot and aborts on a refusal.
- **`apm`** must *agree* with `am`; `GetPerformanceConfiguration` returns what `Set*` was last handed (defaults nonzero; 0 is `Invalid`).
- **`ts`** reports the SoC and PCB sensors at an idle reading. `MilliC` is `GetTemperature` × 1000 and both sit inside `GetTemperatureRange`. **`ISession` is a different interface from its server**: its `GetTemperature` is command 4, the same id as the server's `OpenSession` (shared dispatch made NX-Fetch draw "8 C"). The device code's **high byte** picks the sensor (`0x41…` SoC, `0x43…` PCB).
- **`set:sys`** is a **store**, not a table of answers: the `Get`/`Set` pairs read and write one block, kept in system save data `8000000000000050`, so it persists through the same host flush that persists a title's save. A setting that has two homes has none: `nfc:sys` and `btm:sys` read the radio flags from here, `set`'s language and region are these fields, and `am`'s desired language and keyboard layout are too. `GetSettingsItemValue` is the firmware's separate key/value table; an item that is not in it is **refused** (`ResultSettingsItemNotFound`), because a caller reads the size back and then that many bytes. `GetFirmwareVersion`/`2` are **not cosmetic**: libnx seeds `hosversionGet()` from them and everything version-gated branches on that.
- **`csrng`** fills from `Cpu::next_random_u64` (splitmix64 seeded from the emulated clock; wasm32-unknown-unknown has no OS entropy). Not for keys, but a reply that leaves the buffer untouched is non-random *and* undetectably so.
- **`spl:`** - an Icosa retail console, production, not in debug mode. Atmosphère's extensions at 65000+ (API version, 65007 emummc type, asked by NX-Fetch) answer zero, i.e. "no CFW", since claiming one would promise behaviour that is not there.
- **`pdm:qry`** and **`prepo`** - a factory-fresh console that has never played anything and never transmits.
- **`pm:*`** are four interfaces on four names. `pm`'s process id must equal `svcGetProcessId`'s: both are `PROCESS_ID` (1) in every process, library applet processes included, whatever their kernel pid; `pm:info`'s program id defaults to the Album applet's.
- **`pcv`/`clkrst`** are the same manager either side of 8.0.0, numbered **by an offset**: a `clkrst` device code is `0x40000000 + module + 1` (reading its low bits as the module put the GPU rate under "CPU" in NX-Fetch). A rate a guest sets reads back.
- **`mii`** is a database with no user Miis in it and the six built-in faces `nn::mii` carries: `BuildDefault` and `BuildRandom` hand those out, and every list read (`Get` through `Get3`) reports zero records rather than refusing.
- **`mm:u`** holds the multimedia clock requests the video decoder makes before it runs. `Initialize` must hand out a request: NVIDIA's multimedia library calls `SetAndWait` through a client pointer only a successful `Initialize` fills in (Just Dance 2019 jumped to address 0 without it). The floor a request asks for is what `Get` reads back.

## What the Home Menu opens that homebrew never does

`lbl`, `audctl`, `nfc:sys`, `btm:sys`, `ldn:m`, `lp2p:m`, `ovln:*`, `olsc:s`, `friend:*`, `news:*`, `bcat:*`, `notif:*`. Most are a creator plus the objects it creates (`olsc:s` is five deep), and a fabricated object id is not callable, so each sub-interface gets a name of its own (`Cpu::ipc_interface`) in `svc.rs`'s dispatch.

**The answer is an empty console, not a broken one.** No friends, news, BCAT, cloud saves, local network, NFC or paired gamepad: every one a state a real console reaches, so callers already have a path for it; a *failure* puts them on the path built for hardware that broke. None of these events ever signal. The settings among them (`backlight`, `audio_control`, `notif_alarms`, …) are stored, not answered: one caller writes, another reads back.

## Notes by file

### `crates/switch-core/src/services/time.rs`

- No network sync or per-region offset: user, network, and local system clocks are the same clock; `SetCurrentTime`/`SetSystemClockContext` are accepted but ignored. Steady clock is `cycles / 1_000_000` seconds (only monotonicity matters). No TZif database: all conversions are UTC, `LoadTimeZoneRule` fills nothing, `ToPosixTime` always reports one match, and the location name comes from `set:sys` so the two services cannot disagree.

### `crates/switch-core/src/services/acc.rs`

- `nn::account::Initialize` runs before save data mounts, and `GetLastOpenedUser`/`TrySelectUserWithoutInteraction` pick whose save opens, so a zero uid ("nobody") is never returned. A title asks once and keeps the answer, so users are set before it starts. `TrySelectUserWithoutInteraction` returns the playing user even with several users, since there is no selector applet; the playerSelect applet answers only when there is one user (see `am`). `IsUserAccountSwitchLocked` is true and `IsUserRegistrationRequestPermitted` false for the same reason.
- `acc:u0`/`acc:u1`/`acc:su` share commands 0..=51; from 100 up ids differ (100 is `InitializeApplicationInfo` on `acc:u0` but `GetUserRegistrationNotifier` on `acc:u1`), so the domain object keeps its service name and those arms dispatch on it.
- `acc:su` 250 `GetBaasAccountAdministrator` returns an `IAdministrator`: `IsRegistered` (200) is true, matching the network service account `GetAccountId` reports, `IsLinkedWithNintendoAccount` (250) is false, and the rest is the manager's commands.
- `InitializeApplicationInfo` is 100, 140 (6.0.0+), and 160 (current SDK, sent by Tomodachi Life as a type-6 request with pid and one u64 placeholder). Refusing it aborts `nnSdk`. 141 is `ListQualifiedUsers`, not another form of it.
- List commands carry no count: callers scan for the first all-zero uid, so the whole buffer must be written (the Home Menu aborted when stale stack data looked like a third uid). Likewise `Get` zeroes `AccountUserData` and `GetNintendoAccountUserResourceCache` zeroes its output buffers, since unwritten caller buffers read back as stack garbage.
- Notifiers have a real event that never signals (nothing registers or changes users); a signalled event sends `nnSdk`'s system worker looking for a missing callback. `IAsyncContext` is returned already complete; a context that never completes hangs its waiter.
- `CheckAvailability` reports success (same trade as `nifm`'s always-connected link) so titles start; the ID token is empty, so real authentication fails at that point. License cache: zeroes, 16-byte reply because the s64 is 8-aligned.
- `GetImageId` must be stable and nonzero (zero means "no icon"; garbage made caches never hit); it is two seeded FNV-1a hashes of uid and picture.
- `LoadImage` must return a real JPEG because callers decode it directly. A user without a picture gets a baseline solid-colour JPEG: DC-only blocks (only the first per component has a nonzero difference), quantization 8 so `8x` round-trips to `x`, minimal but complete Huffman tables (Kraft sum 1). Colour picked by uid with u32 arithmetic so wasm32 and 64-bit hosts agree.
- Nicknames are truncated to 0x1F bytes on a char boundary to avoid mojibake.

### `crates/switch-core/src/display/buffer_queue.rs`

- `QueueBufferInput` `crop` and `transform` must be honored: ignoring them drew Minecraft upside down and put A Short Hike's 720p frame in the corner of a 1080p screen. `scalingMode` is ignored because the canvas scales anyway.
- `REQUEST_BUFFER` must return the registered flattened `GraphicBuffer` (kept verbatim per slot), not `nonNull = 0` with success: the app's `Surface` cache is empty on the first request per slot, and A Short Hike dereferenced the null buffer (fence at `buffer + 0x60`), which landed in soft-mapped low pages and deadlocked the WSI thread far from the cause. An unregistered slot returns Android's bad-index status.
- Default geometry comes from `OperationMode::Handheld.display_size()` and moves with docking via `set_default_size`, but only as a default; dequeued/queued buffers keep the size the guest asked for.

### `crates/switch-core/src/services/hid.rs`

- `nnSdk` calls methods on the `IAppletResource` returned by `CreateAppletResource`, so it must be a real object (libnx only maps shared memory by size).
- `hid` and `hid:dbg` are both `IHidServer`; `hid:sys` is `IHidSystemServer` with its own dispatch (libnx opens it in `hidsysInitialize`).
- A refused command is fatal under `nnSdk`, so many void setters are accepted (e.g. `SetGestureOutputRanges`, which Tomodachi Life sends with 1280x720; `ActivateGesture` pairs with it; `GetLastActiveNpad` blocked the Mii editor).
- `SetSupportedNpadStyleSet` determines how pads are published (`NPAD_PRESENTATIONS`): publishing FullKey and Handheld regardless made titles that only accept Joy-Con pairs abort with 2202-0710. Style set 0 (libnx defaults) means a Pro Controller in slot 0 and handheld in slot 8. Resolution is best-first.
- `GetRegisteredDevices` (`hid:sys` 548, 543 before 13.0.0) reports no paired devices; refusing it made the controller applet abort with 10-0221, which Echoes of Wisdom then relaunched forever.
- System applets (18.0.1 qlaunch) never send `SetSupportedNpadStyleSet`; they use `ApplyNpadSystemCommonPolicy` (303/308), which grants every publishable style, and 310 must report the same set.
- SystemExt is a second copy every pad carries (qlaunch reads only SystemExt), published regardless of style.
- `GetNpadJoyHoldType` reads `NpadCondition` from shared memory and aborts (2202-0710, seen in 21.2.0 Home Menu) unless `is_valid` is set, so hold type changes must be written there too.
- The style-set update event (106) is a single kept object, starting signalled.
- Vibration: the two band amplitudes map to the Gamepad API's dual-rumble strong/weak magnitudes; only the first value of `SendVibrationValues` is used.
- Interface types: the slot 0 Pro Controller is wired (USB), the handheld pad is on the rails; nothing is Bluetooth.

### `crates/switch-core/src/services/nv.rs`

- Every nvdrv command replies with a `u32` NvError word; replying with an empty raw section looks like success under libnx but libtransistor checks the raw size (`SetAruid`, `Initialize`, `GetStatus`).
- Syncpoint events are handed out pre-signalled and manual-reset because each submission completes inside its ioctl; left unsignalled, the Home Menu polled forever. The `nvhost-ctrl-gpu` fault event stays dark: signalling it makes the guest tear down its renderer.
- Ioctl argument buffers are never truncated to the declared size: a video engine `SUBMIT` sends command buffers, relocations and fences after its 16-byte header.
- `nvIoctl3`'s second receive buffer must be filled, or callers get zeroed out-of-line payloads (e.g. GPU characteristics).
- Unset config variables (`NV_CONFIG_VAR_NOT_FOUND`) are normal probe answers, not errors.

### `crates/switch-core/src/services/pl.rs`

- Every shared-font type is answered: the Home Menu looks glyphs up across the whole set. With no firmware fonts, the host font stands in for each type.
- `GetSharedFontInOrderOfPriority` command 6 (system variant) is answered like 5; without it callers saw "loaded, zero fonts" and retried forever. Language priority order is not modelled.
- The smallest of the three output buffers bounds the entry count, and the reply count must match what was written.

### `crates/switch-core/src/kernel/thread_report.rs`

- `THREAD_TYPE_SIZE` was measured from titles that allocate `ThreadType`s back to back; searching further would pick up the neighbour's name.
- Thread names are found as the first word in the `ThreadType` pointing back into the struct at readable text, since the offset varies by SDK version. Unnamed threads (`Thread_0x...`) print their entry function, since all share the SDK trampoline.
- Stack walks only accept frames whose return address follows a call (functions like zlib's `inflate_fast` use x29/x30 as data).
- A thread handle must stay unsignalled until the thread ends; an early join let a title destroy a live thread object.

### `crates/switch-core/src/services/am/mod.rs`

- `ILibraryAppletAccessor::Start` runs an installed applet (`install_program`) as a process of its own (`start_installed_applet`). The storages the caller pushed become the applet's `PopInData`; the applet's `PushOutData` and `PushInteractiveOutData` land on the caller's accessor and fire its pop events; the caller's `PushInteractiveInData` goes to the running applet.
- `RequestExit` (20) queues `ExitRequest` to a running applet, which answers by exiting. `Terminate` (25) ends its process at once.
- `GetResult` succeeds once the applet process has exited or pushed out data; one that faults or is terminated without pushing reports `am` 22 (`LibAppletExitReason_Canceled`). The state-changed event (non-auto-clearing, one per slot) is signalled once the applet has finished.
- An applet that is not installed, fails to start, or is the error applet finishes immediately on `Start` and reports cancelled, the outcome callers are written to survive; a success with an empty output storage would be read as user input. The exception is playerSelect with a single user: it succeeds and pops a `PselUiReturnArg` naming that user, and its pop-out event is signalled until the storage is popped. With several users it stays cancelled.
- swkbd in mode 0 is answered by the page, ahead of any installed applet: `PushInData` storages are kept, `Start` reads the second one as `SwkbdConfigCommon` and leaves the applet running until the host calls `answer_keyboard`, which pops a 0x7D8-byte result (`u32 SwkbdResult`, then the text in UTF-16, or UTF-8 when the config asks). Without a config, or inline (any other mode), it is cancelled. The initial text lives in the transfer-memory work buffer and is not shown.
- An error applet's start logs what it was asked to show and queues it as an `ErrorDialog` for the host, from its last pushed storage: the `ErrorCommonHeader` type, the Result at 0x10 when the result flag (byte 6) is set and otherwise the two words at 8, and any text at 0x18. The layout follows libnx's `error.h` and is not checked against a real error applet.
- When a library applet is the booted title, the host is its caller: `PopInData` is synthesized (`LibAppletCommonArguments` plus each applet's private launch storages; swkbd and the controller applet need two, swkbd a third 0x1000 work buffer). `PushOutData` and `PushInteractiveOutData` are kept for the host and logged; interactive replies come from the host via `push_applet_interactive_in_data`. Pop events track queue state (manual-reset), since a signalled event over an empty queue sends `nnSdk` to a fatal pop.
- `LaVersion` per applet is the number its 18.0.1 build expects (swkbd 0x8000D, controller 0x8, myPage 0x10000, web 0x80000, miiEdit 3); claiming version 1 describes a different, smaller launch struct.
- Applet title ids `0100000000001001..1013` follow `AppletId` order, except that `overlayDisp` (`100C`) and `starter` (`1012`, a SystemApplication) are not library applets.
- `appletOE` sub-interfaces are resolved by domain object id (libnx) or by per-session handle name (`nnSdk`, which does not convert to a domain).
- Which proxy a process opens declares its applet kind: application (0), library applet (200/201), system applet (100/110, the Home Menu, which aborts if this fails), system application (350, e.g. `starter`), overlay (300). Commands that open no proxy (450, 460) must not change that flag, which decides whether the startup message is `FocusStateChanged` (applications) or `ChangeIntoForeground` (applets).
- `ICommonStateGetter::GetEventHandle` starts signalled when a message is queued (one at startup) and is auto-clearing; applets poll it with a zero timeout before `ReceiveMessage`, and nothing draws until told it has focus. `ReceiveMessage` returns that message once, then "no message".
- `SetHandlesRequestToDisplay` must queue `RequestToDisplay` (41); the Home Menu waits for it (then calls `ApproveToDisplay`) before dequeuing a buffer.
- Operation mode: Handheld is 0, Console is 1. Mode and default display resolution both come from `OperationMode` so they cannot disagree (NX-Fetch printed "1280x720 @ 60Hz [Docked]" when they did). `GetDefaultDisplayResolutionChangeEvent` is one shared event so a dock can signal it.
- `PreselectedUser` must be a strict 0x88-byte block (magic, version, uid at 8); a zero uid makes `nn::account::OpenPreselectedUser` assert. `UserChannel` is not answered.
- `EnsureSaveData` creates the running title's save for the given user (a zero uid creates nothing) and reports 0 bytes still needed.
- `ExtendSaveData` is granted and remembered so `GetSaveDataSize` reads back the new size. `GetSaveDataSizeMax` reports the NACP's ceiling as-is (0 means "never grows").
- Capture buffer acquires must name a real slot (the first past the two shared frame buffers, soft-mapped zeros); answering "nothing written, slot -1" makes `nnSdk` retry forever.
- `ILockAccessor` events are created signalled and manual-reset: the Home Menu aborts if the <kbd>HOME</kbd> lock event reads clear.
- `IGlobalStateController` sleep/shutdown/reboot are deliberately not implemented: answering "done" to a shutdown not performed is worse than refusing.
- Interface-returning commands must return a real object: `nnSdk` reads a move handle, and a bare success yields a null `SharedPointer` that faults later.
- Setters with a matching getter (idle time detection, auto-sleep, <kbd>HOME</kbd> double-click) must store and read back the value.
- `CreateTransferMemoryStorage` returns a zero-filled storage of the requested size: transfer memory records no address here, and refusing it stopped swkbd's caller before `Start`. `CreateHandleStorage` is refused.

### `crates/switch-core/src/services/log.rs`

- `fatal:u` commands carry the guest's `Result`, its only account of why it stopped; it is reported, and the call succeeds (no error screen policy).
- `lm` is where `nnSdk`'s `NN_LOG` output goes (not `svcOutputDebugString`). `Log` reads its packet from the map-alias send buffer. The prefix goes on a message's head packet and the newline on its tail.

### `crates/switch-core/src/services/net.rs`

- Model: an empty network, a state a real console reaches and every caller handles. `sfdnsres` resolves nothing; `nifm` reports a wired LAN link that is up with internet; `ssl` builds contexts that never handshake; `bsd` connections to anything but this process are refused immediately (`ECONNREFUSED`, not a timeout, since a frame loop has no other thread to run while blocked).
- Loopback works: asio builds a socket pair per `io_context` (bind `127.0.0.1:0`, connect to the port `getsockname` reports, accept) to wake its own `select`; Asphalt 9 asserts on it. Connect completes immediately because both ends are this process.
- `GetSockName` must report a normalized address (`sin_len` set, real port for a port-0 bind) and the third reply word (length) on `GetSockName`, `GetPeerName`, `Accept`, `RecvFrom`, `GetSockOpt`: nnSdk passes the length on, and a zero length made Asphalt 9's connect fail with `EINVAL` inside the SDK.
- Errnos are FreeBSD's (`EAGAIN` 35). `sfdnsres` uses FreeBSD positive `EAI_*`. Failures are definitive (`EAI_NONAME`, `HOST_NOT_FOUND`), not try-again. Numeric addresses also fail (the packed `addrinfo` layout is unverified). The error goes in the first word of `SfdnsresRequestResults`, errno 0.
- Polls/selects that find nothing park the thread for their timeout, capped at 1 ms of guest time since nothing wakes them early and asio's loopback wakeups must still land. A poll that returns at once keeps its whole time slice spinning (NXpotify's Zeroconf `poll(&pfd,1,200)` loop starved main), and a yield was not enough: Just Dance 2017's `select` with a 1 s timeout spun thousands of times a frame. Zero timeout is a probe and does not park; a null `select` timeout or a negative `poll` one waits forever. A blocking recv with nothing queued still yields.
- `select` out-sets are always written; callers read readiness from the output buffer. Descriptor n is bit n of the byte array for both 32- and 64-bit `fd_mask`.
- Unknown descriptors: not ready for `select`, no events (not `POLLNVAL`) for `poll`, since guests poll pipes and stdio alongside.
- **A datagram socket needs no peer**, and none of the three things it does may claim otherwise, or it contradicts the `nifm` that just said the link was up: a `SendTo` naming an `AF_INET` destination is sent and reports the byte count; a read with nothing queued is `EAGAIN` (nothing has arrived *yet*), not `ENETUNREACH`; `select`/`poll` call it writable, or a caller that waits for readiness never sends. RakNet's `BindShared` sends a test datagram to its own address and reads a failed send as `BR_FAILED_SEND_TEST`, so `ENETUNREACH` there failed every `RakPeerInterface::Startup`, and Minecraft then called through the peer it had just destroyed. A datagram with *no* destination, and a stream socket with no connection, still fail.
- `fcntl` flags and `FIONBIO` share one verbatim flags word, since `O_NONBLOCK` differs between FreeBSD, newlib and Linux. Socket family is not validated (`AF_INET6` differs too).
- `DuplicateSocket` copies local state only, not the connection.
- Descriptors are allocated monotonically; listings are sorted for deterministic runs.
- `ssl`: `GetCertificates` must return the firmware store (system data 0x0100000000000800, `/ssl_TrustedCerts.bdf`); an empty store aborted the browser applet. Imported PKI is accepted and given an id but not kept.
- `nifm`: all three names (`nifm:u`, `nifm:s`, `nifm:a`) route here. `IGeneralService` command ids: 12 `GetCurrentIpAddress`, 15 `GetCurrentIpConfigInfo`, 18 `GetInternetConnectionStatus` (12 and 18 were once crossed). IP config (address, /24, gateway .1) must agree with `bsd`. `GetCurrentNetworkProfile` must write its buffer. `IRequest` is accepted immediately, and `GetSystemEventReadableHandles` returns two copy handles, both signalled.

### `crates/switch-core/src/services/erpt.rs`

- `erpt:r` must read back exactly what was filed: callers that cannot find a report they filed conclude the journal is broken.

### `crates/switch-core/src/services/mii.rs`

- The Mii database is empty (no NAND), but the six default Miis live in `nn::mii`'s image, so `BuildDefault` builds them; an empty count plus no defaults would leave every Mii picker unable to open.
- Default Mii colours are stored in the 3DS/Wii U palette and widened to the Switch palette (`MII_HAIR_COLORS`, `MII_EYE_COLORS`); faceline colour numbers are the same in both.
- Create ids are counted, not random (deterministic across runs for trace comparison), with version-4/variant bits set. The tag's last byte holds the count (256 ids, above the 100-Mii database limit). Default and random Miis use different tags so a built Mii never shares a built-in identity (databases key on create id).
- `BuildRandom` only filters on Gender (the only attribute the defaults differ in) and walks matches in sequence, since editors call it once per face.
- `nnSdk` converts the mii session to a domain before `GetDatabaseService`; the control reply must return an object id.
- `IsBrokenDatabaseWithClearFlag` must be answered: unanswered, the editor reads its stack and may offer to wipe the database.
- `miiimg` must answer `GetCount`: a fabricated reply made the editor query nonexistent images about 500k times ("running but drawing nothing").

### `crates/switch-core/src/services/online.rs`

- Each service's root command (`CreateFriendService`, `CreateBcatService`, `CreateMonitorService`, `OpenSender`/`OpenReceiver`, the olsc getter chain) must return a real object: a fabricated object id made entire interfaces unreachable (the Home Menu waited on handle 0 four objects deep in olsc).
- `ldn:m` `Initialize`/`Finalize`: official software aborts if they fail.
- `lp2p` `GetGroupInfo` returns an empty group instead of the real service's refusal, keeping callers on the "nobody to play with" path.
- olsc: 17.0.0 moved the interface behind `GetOlscServiceForSystemService` (cmd 10000); before that the session was the interface, so both dispatch to the same handler. Transfer start/end holders must be distinct objects with distinct events.
- friend `GetUserPresenceView` (20600, 20601) and `LoadUserSetting` (20800, 20802) fill a zeroed struct: offline, every setting at zero.
- friend `Pop`, bcat `GetImpl`, the news data service and the bcat file/directory services are refused rather than answered with zeroed data that callers would treat as real.
- news: the five `news:*` names are permission levels; permissions are not modelled.

### `crates/switch-core/src/services/ns.rs`

- Most of `ns` hands out sub-interfaces, and a fabricated object id is not callable, so an unimplemented getter ends the caller's whole chain.
- From 3.0.0 the manager sits behind the getter services (`ns:am2` and the rest, `IServiceGetterInterface`): eleven interfaces at 7988..=7999 (7990 unassigned), ids per libnx `ns.c`. Pre-3.0.0 `ns:am` is the manager itself; both routes land in the same handler.
- Record-count replies must write the zero count: a success with no out-data makes callers read the count off their stack ("several billion titles").
- `GetApplicationRecordUpdateSystemEvent` is handed out signalled (hardware's record set is already current); the Home Menu waits on it before listing titles, and it is one event per console (`Machine`). Media events are unsignalled but the same object every time because the Home Menu keeps one waiter per event.
- `CheckSdCardMountStatus` refusing means "the card went away"; the emulated card is always mounted. Total/free space use the same 32 GiB, half-used card as `fsp-srv`, and free must not exceed total or callers underflow.
- `IDynamicRightsInterface::HasAccountRestrictedRightsInRunningApplications` is checked by a 20.0.0 Home Menu before launch; refusing it aborted. `ns:vm` 1200: the web applet aborts with 2010-0221 without an answer.
- `aoc:u`: a fabricated reply to `CountAddOnContent` was read as a count, sending titles after nonexistent DLC. Listed indices and mountable content come from the same registration (a listed but unmountable index is worse than an unlisted one). Base id is the NACP's, else the program id with its low 13 bits masked plus 0x1000; a DLC id is base id + index below 0x800. `PrepareAddOnContent` is an acknowledgement only. `CheckAddOnContentMountStatus` has no out value; failure means removed DLC.
- `caps:a`: the Album applet polls `IsAlbumMounted` and `GetAutoSavingStorage` every frame. Mounted-and-empty is a state the applet shows; unmounted is the card-removed error. Cmd 18 is unnamed, issued once first with a 0x40-byte buffer. `GetAlbumAccessResultForDebug` (50011) reports success.
- 20.0.0+ unnamed `IApplicationManagerInterface` event getters (4022, 4088) are signalled before handout.
- `GetApplicationControlData` (400) answers for titles the page has opened: it stores each title's `control.nacp` and icon and hands them to every new session, since a homebrew save manager (EdiZon) runs without the game loaded. Any other title gets ncm's `ContentMetaNotFound` (2005-0007); which code a real console returns is unverified.

### `crates/switch-core/src/kernel/ipc.rs`

- `POINTER_BUFFER_SIZE` is 0x8000 (what a real `fsp-srv` reports), answered for every session before dispatch: with 0, `nnSdk` refused explicit `HipcPointer` arguments (`sf` 11-141 `PointerBufferTooSmall`, Tomodachi Life abort). Both descriptor forms land in the same address space, so the number only picks which descriptor callers fill.
- AutoSelect buffers: `cmifRequestInAutoBuffer` fills a static and a map-alias descriptor and nulls the unused one, so services must use `ipc_input_buffer`/`ipc_output_buffer`. Reading only the map-alias side handed `nvdrv` nothing once ioctl args went through pointers.
- Header counts are decoded in one place (`HipcHeader`, `ipc_header`): separate walks each re-derived them, and one skipped a special header's pid but not its copy/move handles (four bytes short per handle).
- TIPC (12.0.0+ `sm:`): command id in the type field as `16 + cmd`, arguments directly in the data area, no SFCI, no 16-byte alignment, no domains. Without it, `cabinet`'s first request had no command id and the applet aborted.
- The SFCI header position depends on descriptors: nvdrv `KICKOFF_PB` lands at 0x40; a fixed scan missed it, GPU submits got a generic success, and hbmenu's fence never signalled. Domain requests push the payload to 0x20; assuming 0x10 made `fsFileRead` read 0 bytes at offset 0 and broke `romfsMountSelf`. Pre-CMIF libnx sessions (NX-Shell's `fsDirRead`) send `{type=2, object_id, cmd_id}` without SFCI.
- Map-alias descriptors are read as low 32 bits only since guest memory is `u32`-indexed.
- Receive-static descriptors sit after the raw data (`IProfile::Get`'s `AccountUserData`).
- Copy vs move handles live in different descriptor fields in that order; a copy handle in the move slot reads as 0 (`nnSdk` waited on handle 0 after `GetGpuErrorDetectedSystemEvent`; `vi`'s vsync event once did the same).
- Reply type field must be 0: libtransistor rejects anything but 0 or 4 (error 0x7E0DD; a libtransistor title failed to open fsp-srv).
- Replies are written over the request, so the declared section (header, raw data and four padding words) is zeroed first. An empty success on a command with an out parameter still hands back stale request bytes past that: `ListDisplayModes` read its count from the previous request and cost the Home Menu a billion instructions, and `set:sys` blocks wider than 16 bytes (`TvSettings` 0x20, `NotificationSettings` 0x18) leaked a NaN gamma. Write the full out width, or refuse.
- Plain sessions (libtransistor never converts to domains) need a real moved session handle for sub-interfaces; domain objects broke `fsp_srv_open_sd_card_filesystem`.
- Control messages share command ids with service commands, so they are checked first, with `ipc_is_control_request`: `nnSdk` sends them as type 7 (with context) and requests as 6, and testing `== 5` ran `appletOE`'s and the Home Menu's `QueryPointerBufferSize` as command 3 (`set` 3 told `nnSdk` its pointer buffer was 18 bytes, aborting Just Dance 2017; on `vi` it ran a binder transaction). Domain-ness is the domain header's type byte, not the hipc type.
- Close requests (type 2) and domain closes carry no command id and are handled in `svc.rs` before dispatch; scanning for SFCI would find the previous request's id (the Home Menu's `IStorage` close ran as a Read). `ssl` context counts and Opus decoder state are released on close.
- `kept_event`: a second ask must return the same event or the caller waits on an unsignalled copy. Most are never signalled, describing things that never happen here.
- A command an implemented service does not know is refused with cmif 10-221 (`unimplemented_command`) instead of a blanket success (that made `nn::oe::SetupGpuErrorHandler` wait on handle 0). A service with no implementation at all gets `reply_with_fabricated_object`: a real sub-session or domain object, the raw id, and an unsignalled copy event, reused per `(session, command)`, because the intended out type is unknown. A missing move handle made boot2 call through a null `SharedPointer` after `gpio`'s `OpenSession2`, and a missing event stalled the Home Menu's message thread.
- `warn_stub` marks answers with nothing behind them (invented values, unrecorded latches, never-signalled events), once per `(interface, command)` via the diagnostic channel; true emulated facts (the host's users, no DLC, no network) are not marked. Refusals, missing services and stubs are also counted for `Cpu::take_service_gaps`.
- `btm:sys`'s `GetCore` must be real because every other command goes through it; the radio flag is `set:sys`'s Bluetooth flag. `nfc:sys` enabled flag is `set:sys`'s NFC flag; device commands are refused since no device handle was ever handed out. `ngc` must write the output text (callers otherwise read uninitialised buffers) and return a nonzero content version. `npns` `Receive`/`ReceiveRaw` are refused (the empty-queue error is undocumented).

### `crates/switch-core/src/kernel/svc.rs`

- `CloneCurrentObject` must return a new session move handle; `nnSdk` clones `fsp-srv` before `MountRom`.
- `svcResetSignal` must fail when nothing was signalled; succeeding unconditionally made drain loops spin.

### `crates/switch-core/src/services/usb.rs`

- `usb:ds` and `usb:hs` describe a console with no cable and nothing plugged in, using the 11.0.0+ command ids (`usb:ds` 0 opens the `IDsService`). Device state is Detached, speed None, the host side lists no interfaces, and the state, setup and completion events are never signalled. Setup commands (descriptors, `Enable`, endpoints) succeed; transfers, `GetSetupPacket` and `AcquireUsbIf` are refused because they need a host or a device on the other end. Goldleaf panicked (`svcBreak` value 0x1764) on fabricated `GetState`/`GetSpeed` replies.

### `crates/switch-core/src/services/fs.rs`

- Any `fsp-srv` command that hands back an object must never answer a bare success: the caller reads out-object id 0, wraps it, and calls through a null vtable. Guest memory is soft-mapped from zero, so the fault surfaces far later at `pc=0` (Asphalt 9 with cmd 9; Just Dance 2017 with cmd 203; JKSV with cmd 68). Answer "not found" (2002-0001) for content the console lacks.
- 203 (patch RomFS) must answer `TargetNotFound` (2002-1001/1002): `QueryMountRomCacheSize` treats only those as "no patch".
- `ISaveDataInfoReader` reporting zero entries is the only termination signal (Checkpoint looped on a fabricated success).
- `IStorage::Read(s64 offset, u64 size)` has no leading option word (unlike `IFile::Read`; confusing them returned "0 bytes at offset 0x50"), and is all or nothing: out-of-range reads return 2002-3005 rather than clamping.
- `CreateFile` on an existing file must fail with "already exists" (`fsdev` opens for write that way); `SetSize` must really truncate (`O_TRUNC`). `IFile` writes must persist (Checkpoint re-reads its config).
- 1003 `DisableAutoSaveDataCreation` is accepted but not honoured: saves are created on open and there is no installer.
- Save-quota defaults (64 MiB save, 16 MiB journal) are deliberately generous; nothing enforces quotas, and under-reporting makes titles refuse to save. A real NACP's 0 is passed through.
- Detection-notifier events never fire (no slot changes) and are one per slot, shared by all callers to avoid handle leaks.

### `crates/switch-core/src/services/ldr.rs`

- `ldr:ro` maps a copy of the caller's NRO (page storage is not shareable, same constraint as `svcMapMemory` via `Memory::copy_range`); writes to the source buffer do not reach the module, which no guest relies on.
- NRR registrations are recorded but only the magic is checked (no console key to verify the signature chain), matching a console with the check patched out.
- `RegisterProcessHandle` is `nn::ro::Initialize`'s first call.
- `UnloadModule` accepts either the mapped address or the source buffer address because both are a `u64 nro_address` and unmapping the wrong module is worse.
- Module region allocation is first fit over live mappings so plugin load/unload cycles do not exhaust it. `.text` is read-execute while mapped and the BSS must be at least the header's size.
- Results use module 22.

### `crates/switch-core/src/services/power.rs`

- `CLOCK_RATES_HZ` are original-console handheld rates (GPU 384 MHz, not docked 768 MHz) to match `am`'s operation mode and `apm` Normal; only the GPU default follows the dock, since the CPU rate defines emulated time (`GetSystemTick`, timed waits).
- `gpio`: every pad reads High. Buttons are active-low and boot2 enters maintenance mode when both volume pads read Low, so answering 0 boots into maintenance mode.
- `psm`: charger type maps the host's charging bool to EnoughPower/Unconnected; `IPsmSession` events are never signalled (battery is polled via `Cpu::set_battery`).
- `psc:m` module events never fire (no sleep/shutdown).

### `crates/switch-core/src/services/settings/mod.rs`

- `FIRMWARE_VERSION` picks which side of feature gates titles take (Tomodachi Life uses `am`/`hid` commands from 18.0.0 and 20.0.0), not what is finished.
- Settings block format: magic `swsetsys`, then records tagged by the `set:sys` command id; unknown tags are skipped, missing ones keep defaults, wrong-width values are ignored, a truncated record ends the read. The whole block is rewritten on each change.
- Settings load lazily because saves are restored after the session is built.
- Defaults matter where zero is wrong: `SleepSettings` plans are indices (0 = sleep after one minute, so default is `Never` = 5), `PlatformRegion` has no zero (the error applet took an svcBreak on it), `ProductModel` starts at 1, `KeyboardLayout` 0 is Japanese, `TouchScreenMode` 0 is Stylus, `InitialLaunchSettings` must mark setup complete and an accepted EULA must exist or the Home Menu hands over to `starter`.
- Pre-4.0.0 `GetAvailableLanguageCodes` (1) reports 15 codes, not 12: `MakeLanguageCode` indexes by `SetLanguage`, and capping at 12 aborted Minecraft on `fr-CA`. Answering command 1 with no data aborted Just Dance 2017.
- `HOME_MENU_SCHEME` is a stub, not measured. `MII_AUTHOR_ID` is fixed so Miis stay this console's across sessions.
- Settings items exclude `hid_debug` (hid here is emulated, not the sysmodule).
- `pctl`: "is restricted" and "is allowed" query families read in opposite senses; a blanket false reported free communication as unavailable. Refusing the event getters (synchronisation among them) took the Home Menu down. `GetPlayTimerRemainingTime` returns `i32::MAX` since zero means time is up.
- `lbl`: the settings applet sets a brightness and reads the applied value, so they must agree.
- `notif`: command 1000 means different things on `notif:a` (Initialize) and `notif:s` (GetNotificationCount).

### `crates/switch-core/src/services/vi.rs`

- `TransactParcel` (0) must work, not only `TransactParcelAuto` (3, added in 3.0.0): pre-3.0.0 SDK titles like Just Dance 2017 send only 0, and an empty success queued every frame into nothing. Same for domain sessions (libnx's default).
- Event handles (`GetNativeHandle`, `GetDisplayVsyncEvent`) must go in the copy slot.
- Home Menu uses the system shared buffer when `IsSystemBufferSharingEnabled` succeeds; otherwise it builds its own swapchain and never draws into it.
- `PresentSharedFrameBuffer`: `android::Fence` is 36 bytes, so the crop is at 0x24 and transform at 0x34; reading one field later decoded swap interval 1 as `FLIP_H`.
- The shared buffer is reserved at docked size since docking can happen after creation, but its pool is laid out at 720p and must not change on dock: qlaunch keeps drawing into the old slots, and resizing gave a black docked frame.

### `crates/switch-core/src/services/hwopus.rs`

- `GetWorkBufferSize` must be nonzero: 0 stopped decoding entirely. Packet format and decoder rules are in `docs/audio.md`, as are the `audout` and `audren` lessons.
