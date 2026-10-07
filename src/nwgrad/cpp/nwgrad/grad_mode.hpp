#pragma once

// What a pair's alignment computes besides its score.
//   Hard — Viterbi DP; score = max-path score; grad = subgradient (pair counts on the path)
//   Soft — forward-backward; score = log Z (times T); grad = expected counts
//   None — score only (plus the Viterbi path); no gradient
enum class GradMode { None, Hard, Soft };
