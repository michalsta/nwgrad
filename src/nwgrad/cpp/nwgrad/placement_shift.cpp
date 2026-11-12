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
// "R" (SHF_GNU_RETAIN, binutils >= 2.36) keeps the unreferenced section through
// --gc-sections.  ELF only; on other object formats the knob is a no-op.
#if defined(NWGRAD_PLACEMENT_SHIFT) && NWGRAD_PLACEMENT_SHIFT >= 0 && defined(__ELF__)
#define NWGRAD_PS_STR2(x) #x
#define NWGRAD_PS_STR(x) NWGRAD_PS_STR2(x)
__asm__(".pushsection .text.unlikely.nwgrad_placement_shift,\"axR\",@progbits\n"
        ".p2align 12\n"
        "nwgrad_placement_shift_pad:\n"
        ".fill " NWGRAD_PS_STR(NWGRAD_PLACEMENT_SHIFT) ",1,0xcc\n"
        ".popsection\n");
#endif
