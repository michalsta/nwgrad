/* Kernel code layout for GNU ld (ELF), passed with -Wl,-T: see "Code placement" in
   CMakeLists.txt.  INSERT augments the default script instead of replacing it.

   Each per-ISA-level unit's functions (-ffunction-sections: one .text._Z* input section
   each) go into one output section ahead of .text, every unit starting on a 4 KiB
   boundary.  A unit's code then sits at the same address mod 4096 whatever is linked
   before it -- the bindings, the other levels -- so a change elsewhere cannot move its
   kernels' hot loops onto different front-end sets.  The cold splits (.text.unlikely.*)
   stay in .text with everything else.  The experiment knob's pad (placement_shift.cpp)
   is placed first, ahead of every anchor. */
SECTIONS
{
  .text.nwgrad_kernels ALIGN(4096) :
  {
    KEEP(*(.text.unlikely.nwgrad_placement_shift))
    . = ALIGN(4096); *level_avx2_inter.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_avx2.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_avx2_soft.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_avx512_inter.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_avx512.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_avx512_soft.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_sse4_inter.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_sse4.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_sse4_soft.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_baseline_inter.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_baseline.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_baseline_soft.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_neon_inter.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_neon.cpp.o(.text._Z*)
    . = ALIGN(4096); *level_neon_soft.cpp.o(.text._Z*)
    . = ALIGN(4096);
  }
}
INSERT BEFORE .text;
