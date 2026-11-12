// Placement experiment knob (CMake -DNWGRAD_PLACEMENT_SHIFT=<bytes>; not built by default).
//
// Emits <bytes> of dead code (int3) at the very front of the extension's text: in a
// .text.unlikely.* input section, which GNU ld's default script puts first in .text,
// and in the first object on the link line.  Everything linked after it -- every kernel,
// the bindings, the batch engine -- moves by exactly that many bytes (keep it a multiple
// of 64 to preserve every function's 64-byte phase), while no instruction changes.  The
// timing difference between two shifts is therefore pure code placement.
//
// The pad itself starts on a page boundary, so shift N puts the code after it at
// N mod 4096 plus a fixed offset, whatever precedes it (.plt, .init): a sweep of N over
// 0..4095 visits every placement once.
// NWGRAD_PLACEMENT_SHIFT_AT=text (CMake) puts the pad in a plain .text.* section instead,
// first among the ordinary functions: ahead of the bindings, but behind anything the
// layout script (nwgrad_text.ld) places before .text -- i.e. "the bindings grew by N".
// "R" (SHF_GNU_RETAIN, binutils >= 2.36) keeps the unreferenced section through
// --gc-sections.  ELF only; on other object formats the knob is a no-op.
#if defined(NWGRAD_PLACEMENT_SHIFT) && NWGRAD_PLACEMENT_SHIFT >= 0 && defined(__ELF__)
#define NWGRAD_PS_STR2(x) #x
#define NWGRAD_PS_STR(x) NWGRAD_PS_STR2(x)
#if defined(NWGRAD_PLACEMENT_SHIFT_TEXT)
#define NWGRAD_PS_SECTION ".text.nwgrad_placement_shift"
#else
#define NWGRAD_PS_SECTION ".text.unlikely.nwgrad_placement_shift"
#endif
__asm__(".pushsection " NWGRAD_PS_SECTION ",\"axR\",@progbits\n"
        ".p2align 12\n"
        "nwgrad_placement_shift_pad:\n"
        ".fill " NWGRAD_PS_STR(NWGRAD_PLACEMENT_SHIFT) ",1,0xcc\n"
        ".popsection\n");
#endif
