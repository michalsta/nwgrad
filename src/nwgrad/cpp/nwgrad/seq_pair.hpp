#pragma once

// SeqPairT<T> — one sequence pair: a standalone pair, or a view of pair i of a
// SeqPairBatchT.  Defined with the batch in seq_pair_batch.hpp (each refers to the
// other); this header exists so `#include "seq_pair.hpp"` keeps working.
#include "seq_pair_batch.hpp"
