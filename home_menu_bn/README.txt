qlaunch (Home Menu, 0100000000001000) decrypted for static analysis.

  qlaunch_main_08000000.bin   open as RAW, arch aarch64, base 0x08000000
  raw_main                    the original NSO, if Binary Ninja has an NSO loader
  main.bin                    the same flat image with .bss padding (59 MB)
  raw_main.npdm               the process metadata

Segment layout at that base:
  .text    0x08000000 .. 0x089d9d7c
  .rodata  0x089da000 .. 0x08e7d4c4
  .data    0x08e7e000 .. 0x08f78130
  .bss     0x08f78130 .. 0x0b82c000

Every address in the emulator traces is in these terms, so a function at
0x089ba910 in Binary Ninja is the same 0x089ba910 the traces print.

What I would like named first:
  0x089ba910   the applet framework's per-frame function, V(0xe0) on the
               object at guest 0x9149360 (vtable 0x08f240d0). It runs every
               frame and never reaches a draw.
  0x089ba8c0   the loop that calls it: `V(0xf0)(); state=2; while (!obj[0x236]) V(0xe0)();`
  0x089bae04   V(0xf0), the phase start
  0x08326528 / 0x083261c0   the per-frame update it spends its time in
  0x089c0c94 / 0x089c1218   the resource load (succeeds, returns 1)
