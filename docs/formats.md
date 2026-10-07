# Containers and formats

NCA, NSP/XCI, RomFS, tickets, NSO/NRO, NPDM and the range-reading stack.

## `crates/switch-core/src/control.rs`

- NACP parsing only requires bytes up to the end of the error-code category (0x30B0), not the full 0x4000. The save-extension ceilings (0x3148..) and cache-storage fields (0x3170..) are optional and read as 0 when the NACP is shorter, so truncated NACPs that carry a name and icon still parse.
- A Control NCA's RomFS is read whole (icons plus NACP), capped at 64 MiB; a Program NCA's RomFS is always streamed.
- Icon choice: the icon for the language the name came from, falling back to any icon in the image. Legacy icon file names are still emitted by repack tools for the last two language slots.
- `SaveDataQuota::from` is the single conversion both loaders use; assembling a quota by hand risks a missing field that the reporting command answers as 0 with success.

## `crates/switch-core/src/source.rs`

- Retail containers exceed both wasm32's 4 GiB linear memory and the 2 GiB (`isize::MAX`) single-allocation limit, so all container reads go through `ByteSource` ranges. Lengths from guest/container data are checked against `isize::MAX` and buffers reserved before filling, because allocation failure on wasm is an `unreachable` trap with no message.
- `ByteSource::read_at` contract: a short fill means end-of-source only; I/O failures are errors so callers can tell "no more data" from "unreadable".

## `crates/switch-core/src/ticket.rs`

- Ticket layout (offsets relative to the body, which starts after a signature block of 0x80 ECDSA / 0x140 RSA-2048 / 0x240 RSA-4096): 0x00 issuer [0x40], 0x40 title key block [0x100] (Common: first 16 bytes are the AES-128-ECB-wrapped key; Personalized is RSA-wrapped with a console ETicket key, unsupported), 0x140 format_version, 0x141 titlekey_type (0 Common, 1 Personalized), 0x142 ticket_version (u16), 0x144 license_type, 0x145 common_key_id, 0x160 rights_id [0x10]. Body is 0x2c0 bytes.
- The ticket's `common_key_id` is not used to pick the `titlekek`: retail tickets may say 0 while the content needs a later generation (Asphalt 9: 0 vs `titlekek_07`). The NCA's key generation selects it.
- A ticket bundled in the container overrides a `title.keys` entry for the same rights id (the dump may be for a different revision).

## `crates/switch-core/examples/romfs_selftest.rs`

- RomFS has no IVFC verification, so wrong bytes from the HostSource -> NCA window -> AES-CTR -> compression/cache stack are silently served. The self-test relies on the invariant that a range's bytes must not depend on how it is read (whole, chunked, reversed, with evicting reads between). Samples target file edges and 16 KiB compression-block boundaries (`CACHED_BLOCKS` is 4). `SEED`, `WINDOW`, and `INJECT=1` (canary that must be detected) control it.

## `crates/switch-core/src/nsp.rs`

- PFS0 and HFS0 share one parser: only the magic and entry stride differ. File offsets are resolved to absolute positions in the source so XCI partitions and `.nsp` look the same to the rest of the stack.
- Header size math is done in `u64` and sizes are `u64`: on wasm32 `usize` truncation let entries past 4 GiB pass bounds checks and read wrong bytes.
- Only the header is read from the source; `files` is not preallocated from the untrusted count (allocation failure aborts on wasm).
- Some repack tools write absolute entry offsets. The absolute reading is chosen only when the relative reading overruns the image and the absolute one fits (and no entry points into the header). It used to key on "an entry points into the header", which missed repacks with a padded/aligned payload area (e.g. a 7 GiB Just Dance 2022 `.nsp`). The fallback is never applied to nested partitions (written by the cartridge master).

## `crates/switch-core/src/bucket.rs`

- Compression, sparse and BKTR tables share one bucket-tree layout, differing in node size and entry format.
- Only entry sets are read, into one sorted list with binary search; the index nodes are skipped. A large retail table is ~2.3 MiB / 98,846 entries, cheap next to a multi-GB container.
- `MAX_TABLE` exists only so a corrupt header cannot request an impossible allocation in the browser.

## `crates/switch-core/src/compressed.rs`

- `nn::fssystem` stacks the compression layer directly on the hash layer, so table offsets are relative to the hash layer's image (RomFS past IVFC, ExeFS past its hash table), not the section.
- Decompressed blocks are cached because the guest reads RomFS in small `IStorage` pieces; without the cache each read re-decompresses a 64 KiB block.
- Physical LZ4 data is 0x10-aligned per entry.

## `crates/switch-core/src/nca.rs`

- Base header (0x400) and the four FS headers (0x200 each, sectors 2..5) are AES-128-XTS with `header_key`. Section bodies are AES-128-CTR, keyed by key-area slot 2 (unlocked by `key_area_key_<kind>_<gen>`) or by the title key.
- Title keys from `title.keys` or tickets are still wrapped under `titlekek_XX` and must be unwrapped with the NCA's key generation (the ticket's own generation is unreliable). Using them raw decrypts to noise that looks like "wrong keys".
- Section table entries are `u32 start, u32 end` in 0x200-byte media units, not offset/size.
- The CTR low 8 bytes are the absolute NCA offset / 16, not reset per section; sections therefore need a source over the whole NCA.
- Field labels follow hactool's `nca_fs_header_t`. RomFS data is always IVFC level index 5 (`level_headers[IVFC_MAX_LEVEL - 1]`); `num_levels` reads 7 on real files and must not be used as an index. Byte 0 of an IVFC section is a hash level, not the RomFS header. The IVFC level size is the exact image size; the section size is rounded up to a media unit.
- `AesCtrEx` (patch) sections vary the counter's generation word per subsection; their BKTR tables decrypt with the section's base counter. Sparse sections: the section table extent describes the reassembled section (may lie past EOF); the stored body is at `SparseInfo + 0x20`, and the sparse table is encrypted at its stored offset under its own generation (shifted high, as `NcaSparseInfo::MakeAesCtrUpperIv`). Reassembly happens under decryption since counters follow reassembled positions.
- Compression sits above the hash layer: hashes cover the compressed bytes, so verify before decompressing.
- Wrong keys are undetectable by CTR itself. PFS0/ExeFS sections are checked against the master hash and then every per-block hash (otherwise one bad byte boots and faults later in crt0); verification is skipped for unrecognized geometry rather than failing. RomFS only checks the header's `header_size == 0x50`; full IVFC verification is not implemented.
- An update's Program NCA carries the base title id; only its patch (`AesCtrEx`) RomFS marks it as unbootable alone.
- Container NCAs have hash names, so finding the Program NCA requires decrypting each header (needs `header_key`).
- RomFS is served range-by-range through `romfs_source`; `read_section`/`read_romfs` hold whole sections and are for small ExeFS or native tools only.

## `crates/switch-core/src/crypto.rs`

- Hand-rolled because `switch-core` has no dependencies; verified against NIST SP 800-38A/38E vectors and OpenSSL.
- Table-driven AES (`TE`, 4 KiB) is about 4x the textbook passes; cache-timing leakage is irrelevant since the keys are the user's own. A textbook reference is kept in tests and compared on pseudo-random inputs.
- `RoundKeys` exists so bulk modes expand the key once: per-block expansion was 28% of a Home Menu boot (firmware fonts, 17.7 MB of AES-CTR).
- CTR works on column words and a two-u64 counter to avoid byte scatter/gather and per-block carry walks.
- SHA-256 verifies a decrypted section's hash region against the FS header master hash: the only way to detect a wrong key, which yields plausible garbage.

## `crates/switch-core/src/nso.rs`

- SDK modules start at `.text + NSO_ENTRY_OFFSET` (after `ModulePtr` + `MOD0` header). `rtld` has no `MOD0` header and must start at `.text + 0` to run its base-address bootstrap; skipping it leaves x0 = 0 and a ~4 GiB memset. `entry_offset` detects the signature instead of assuming it.
- rtld handles relocations and BSS itself, so the loader only places decompressed segments.

## `crates/switch-core/src/nro.rs`

- NROs load at 0x08000000 because devkitA64/libnx homebrew is linked against HBL's load address; absolute pointers assume it.
- The environment block is at 0x00100000, outside the image so crt0 BSS zeroing never touches it. HosVersion entry uses 0xFFFFFFFF with the "ATMOSPHR" magic so libnx keeps it and version gates pass. Argv's `argv[0]` must be `sdmc:/switch/homebrew.nro` and that file must exist, because `romfsMountSelf` reopens the running NRO by argv[0]. hbmenu's `launchInit()` fails without the NextLoadPath buffers (0x300 each, as hbloader).
- Self-relocating "HOME BREW" crt0s apply RELR themselves; applying it in the loader too double-relocates. Plain NROs (e.g. the sdl demo) need the loader to do it. hbmenu writes the RELR byte size into DT_RELRCOUNT; the loader takes the larger of RELRSZ/RELRCOUNT, and stops at the first non-monotonic entry (hbmenu's section has trailing garbage).
- `.text` is marked read-only after load so wild guest writes fault instead of corrupting code; `.rodata` stays writable because a self-relocating crt0 may patch `.data.rel.ro` there.
- libtransistor NROs may set `_trn_runconf_heap_mode` to OVERRIDE with a tiny heap; the loader finds the live (possibly strong, not the exported weak) mode pointer by decoding `_sbrk_r` and forces NORMAL so it calls `svcSetHeapSize`.
- Asset parts whose (offset, size) run past the file are dropped individually (half an icon is not an icon). `nro_size` is measured from file start, not the magic, so builds with a boot stub still find the section.

## `crates/switch-core/src/vfs.rs`

- `changed` records only guest-side mutations so the host can persist the card incrementally. Host loads (`write_file`) are deliberately not recorded, else every restored file would be written straight back. `guest_write_file` is for services (e.g. `set:sys`) that keep state in a save and must be persisted.
- `create_file` must fail if the path exists and never truncate: `fsdev` opens existing files by calling `CreateFile`, expecting "already exists". Truncating emptied files on every reopen.
- `normalize` resolves `.` and `..` (clamped at the root): guests build paths by joining, and `hb-appstore` otherwise created literal `.` directories (`/switch/./.get/packages`).
- Save data starts with `Vfs::empty()` (root only) so no unrequested structure is invented.

## `crates/switch-core/src/lib.rs`

- `env_flag!` caches per call site in a `OnceLock`: `std::env::var` scans the whole environment, and these switches sit on per-syscall/IPC/draw paths (`getenv` was 37% of a Home Menu run). `TRACE_*` diagnostics are not env flags; they live in `trace`'s runtime mask, which the environment only seeds, because browsers have no environment.
- `IdHasher` (fxhash-style) replaces SipHash for self-minted keys: SipHash was 18.7% of a Home Menu run. Iteration order was already nondeterministic with the default hasher. `finish` must fold the high half down because one multiply moves entropy upward only and `HashMap` buckets by low bits; without it 4096 page-aligned addresses shared one bucket.
- `FB_BASE`/`INPUT_ADDR` sit above every region a Horizon process is given (see `cpu::GUEST_SPACE_END`); they moved from 0x3F00_0000 when the heap region grew into that space. `switch_fb_snapshot` prefers the GPU framebuffer and falls back to `FB_BASE`.

## `crates/switch-core/src/romfs.rs`

- Header (0x50 bytes, `u64` pairs): 0x00 header size (always 0x50, the only format check), 0x08/0x10 dir hash table off/size, 0x18/0x20 dir metadata off/size, 0x28/0x30 file hash off/size, 0x38/0x40 file metadata off/size, 0x48 file data offset.
- Directory entry: u32 parent, u32 next sibling, u32 first child dir, u32 first file, u32 next in hash bucket, u32 name length, name (UTF-8, padded to 4).
- File entry: u32 parent, u32 next sibling, u64 payload offset (relative to file data), u64 size, u32 next in hash bucket, u32 name length, name (padded to 4).
- Lookups are case-insensitive although RomFS is case-sensitive: the SDK names looked up have been spelled inconsistently by repack tools.
- `RomFsIndex` exists because a title reads its RomFS as raw `IStorage` ranges and walks the tables itself, so the emulator only sees offsets; reading just the tables costs what the guest's own mount costs. Metadata tables are capped (retail ones are a few MB) so a corrupt header cannot drive a huge allocation.
- The cycle guard budgets two reads per directory (once listed by the parent, once walked); one read each rejected valid nested images.

## `crates/switch-core/src/sparse.rs`

- `nn::fssystem` builds the sparse storage over the raw NCA body and layers AES-CTR on top, counting from the section's ordinary (reassembled) offset. Stored bytes are encrypted at their reassembled position and holes decrypt to keystream, not zeroes. The table bytes use their own counter generation (SparseInfo +0x28).

## `crates/switch-core/src/bktr.rs`

- `MAX_TABLE` (64 MiB) is far above real BKTR table sizes (a few hundred KiB); it only stops a corrupt header from requesting an allocation the browser cannot make.
- BKTR tables are flattened at load; the on-disk bucket split is just paging.
- A sparse base or patch section needs no special handling here: `section_source` reassembles it before the relocation table sees it.
- The RomFS header check after composition catches a mismatched base/update pair early instead of a title that boots and then cannot find files.

## `crates/switch-core/src/npdm.rs`

- `system_resource_size` decides the memory layout: Just Dance 2019 declares 0 (plain heap), Just Dance 2023 declares 16 MiB (VAMM). `nnSdk` reads the same figure from `svcGetInfo`, so both must agree.
- Mario Kart 8 Deluxe (`0100152000022000`) is AArch32: its `rtld` opens with `b #+8`, which decodes as `ANDS x0, x0, x0` in A64 and then runs the `MOD0` offset word.
- A system applet like Data Erase gets core 3 alone; refusing it that core panics.
- Missing NPDM means 64-bit (homebrew NROs have none).
