// WFA2-lib called from C: its headers misbehave when included into C++ (the aligner
// crashed in the extend kernel from a C++ TU, ran fine from C), so the driver goes
// through these four functions.
#include <string.h>
#include "wavefront/wavefront_align.h"

// metric: 0 edit, 1 gap_linear, 2 gap_affine; scope: 0 score, 1 alignment;
// memory: 0 high, 1 med, 2 low, 3 ultralow (BiWFA).  Penalties in WFA's sign convention.
void* wfa_shim_new(int metric, int match, int mismatch, int go, int ge, int scope, int memory) {
    wavefront_aligner_attr_t attr = wavefront_aligner_attr_default;
    attr.heuristic.strategy = wf_heuristic_none;
    attr.alignment_scope = scope ? compute_alignment : compute_score;
    if (metric == 0) {
        attr.distance_metric = edit;
    } else if (metric == 1) {
        attr.distance_metric = gap_linear;
        attr.linear_penalties.match = match;
        attr.linear_penalties.mismatch = mismatch;
        attr.linear_penalties.indel = ge;
    } else {
        attr.distance_metric = gap_affine;
        attr.affine_penalties.match = match;
        attr.affine_penalties.mismatch = mismatch;
        attr.affine_penalties.gap_opening = go;
        attr.affine_penalties.gap_extension = ge;
    }
    attr.memory_mode = memory == 0 ? wavefront_memory_high : memory == 1 ? wavefront_memory_med
                     : memory == 2 ? wavefront_memory_low : wavefront_memory_ultralow;
    return wavefront_aligner_new(&attr);
}

int wfa_shim_align(void* w, const char* a, int la, const char* b, int lb) {
    wavefront_aligner_t* wf = (wavefront_aligner_t*)w;
    wavefront_align(wf, a, la, b, lb);
    return wf->cigar->score;
}

void wfa_shim_delete(void* w) { wavefront_aligner_delete((wavefront_aligner_t*)w); }
