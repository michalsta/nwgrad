#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "grad_mode.hpp"

// Temporary 0.7 compatibility; delete with the deprecated pair API in 0.8.
// Modern fills always publish their results. Only old facade calls use this helper.
struct PairApiCompat {
    template<class P> static void full(P& p) {
        p.align();
        p.batch().visit([&](auto& e) { hide(e, p.index()); });
    }
    template<class P> static void banded(P& p, int bandwidth) {
        if (!p.path_valid())
            throw std::logic_error("nwgrad: call align_full() before realign_banded()");
        p.banded_align(bandwidth);
        p.batch().visit([&](auto& e) { hide(e, p.index()); });
    }
    template<class P> static void publish(P& p) {
        p.batch().visit([&](auto& e) { release(e, p.index()); });
    }
    template<class E> static void hide(E& e, size_t i) noexcept {
        e.flags_[i] &= static_cast<uint8_t>(~E::kGrad);
    }
    template<class E> static void release(E& e, size_t i) {
        if (e.grad_valid(i)) return;
        if (!e.score_valid(i))
            throw std::logic_error("nwgrad: call align_full() or realign_banded() first");
        if (!e.path_stored(i))
            throw std::logic_error("nwgrad: DP tables have been dropped; call align_full() or realign_banded() first");
        if (e.grad_mode() == GradMode::None)
            throw std::logic_error("nwgrad: grad_mode is None; construct with Hard or Soft to enable gradients");
        e.flags_[i] |= E::kGrad;
    }
    template<class E> static void release_for_sum(E& e) {
        for (size_t i = 0; i < e.size(); ++i) release(e, i);
    }
};
