#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Alphabet: the single owner of the character <-> index mapping.
//
// An Alphabet is immutable, and instances are interned: Alphabet::get() returns
// the same instance for the same symbol string, so two alphabets are equal iff
// their addresses are equal.  Every SubstMatrix, AlignParams and encoded
// sequence carries a `const Alphabet*`, and compatibility checks are pointer
// comparisons.
//
// Interned alphabets are never destroyed.  That is deliberate: it makes a raw
// `const Alphabet*` impossible to dangle, so nothing has to pay for shared_ptr
// refcounting on the copy-heavy AlignParams path.  Only genuinely distinct
// alphabets accumulate (a few hundred bytes each), so the registry cannot grow
// to any size that matters.
//
// Scope: legality and ordering only.  An Alphabet says which characters are
// permitted and what index each gets.  It says nothing about scores -- those
// live entirely in the user's SubstMatrix.
//
// Case is significant.  'd' is not 'D'; feeding lowercase to a uppercase
// alphabet throws exactly as any other out-of-alphabet character does.  Callers
// reading soft-masked FASTA must upper-case at the boundary.

class Alphabet {
public:
    // Return the interned Alphabet for `symbols`, creating it if new.
    // Symbols must be unique and non-empty.  Thread-safe.
    static const Alphabet& get(std::string_view symbols);

    // ── Named alphabets ──────────────────────────────────────────────────────
    // Extensions append at the end, so the canonical 20 amino acids keep
    // indices 0-19 and an existing 20x20 matrix (BLOSUM62, PAM, ...) embeds as
    // the top-left block of any extended one.
    static const Alphabet& dna()          { return get("ACGT");   }
    static const Alphabet& dna_n()        { return get("ACGTN");  }
    static const Alphabet& rna()          { return get("ACGU");   }
    static const Alphabet& rna_n()        { return get("ACGUN");  }
    static const Alphabet& protein()      { return get(AA20);     }
    static const Alphabet& protein_x()    { return get(std::string(AA20) + "X");   }
    static const Alphabet& protein_uo()   { return get(std::string(AA20) + "UO");  }
    static const Alphabet& protein_uox()  { return get(std::string(AA20) + "UOX"); }

    // The alphabets the matrices shipped in nwgrad.matrices are actually
    // expressed in.  These are *different orderings* from protein()/dna() above,
    // not extensions of them: NCBI orders its columns ARNDCQEG..., not
    // alphabetically, so a BLOSUM62 does not embed in protein_x().  Combining a
    // matrix over one with a gradient over the other throws, which is the point.
    static const Alphabet& ncbi_protein() { return get("ARNDCQEGHILKMFPSTWYVBZX"); }
    static const Alphabet& iupac_dna()    { return get("ATGCSWRYKMBVHDN"); }

    int size() const noexcept { return n_; }
    const std::string& symbols() const noexcept { return symbols_; }

    // Index of `c`, or -1 if it is not in this alphabet.  Branch-free.
    int index_of(char c) const noexcept {
        return lut_[static_cast<unsigned char>(c)];
    }

    bool contains(char c) const noexcept { return index_of(c) >= 0; }

    // The symbol at `i`.  No bounds check: callers hold indices this Alphabet
    // produced.
    char symbol_at(int i) const noexcept { return symbols_[static_cast<size_t>(i)]; }

    // Validate and encode `s` to indices.  Throws std::invalid_argument naming
    // the offending character and its position.
    std::vector<uint8_t> encode(std::string_view s) const {
        std::vector<uint8_t> out;
        out.reserve(s.size());
        for (size_t k = 0; k < s.size(); ++k) {
            int i = index_of(s[k]);
            if (i < 0) throw_bad_char(s[k], k);
            out.push_back(static_cast<uint8_t>(i));
        }
        return out;
    }

    std::string decode(const std::vector<uint8_t>& idx) const {
        std::string out;
        out.reserve(idx.size());
        for (uint8_t i : idx) out.push_back(symbol_at(static_cast<int>(i)));
        return out;
    }

    // Alphabets are interned, so identity is address identity.
    bool operator==(const Alphabet& o) const noexcept { return this == &o; }
    bool operator!=(const Alphabet& o) const noexcept { return this != &o; }

    Alphabet(const Alphabet&)            = delete;
    Alphabet& operator=(const Alphabet&) = delete;

private:
    static constexpr std::string_view AA20 = "ACDEFGHIKLMNPQRSTVWY";

    explicit Alphabet(std::string_view symbols)
        : symbols_(symbols), n_(static_cast<int>(symbols.size()))
    {
        if (symbols.empty())
            throw std::invalid_argument("nwgrad: alphabet must be non-empty");
        if (symbols.size() > 127)
            throw std::invalid_argument(
                "nwgrad: alphabet may have at most 127 symbols");

        for (int i = 0; i < 256; ++i) lut_[i] = -1;
        for (int i = 0; i < n_; ++i) {
            char        ch = symbols[static_cast<size_t>(i)];
            auto        c  = static_cast<unsigned char>(ch);

            // Symbols must be single-byte ASCII.  Python hands us std::string as
            // UTF-8, so a non-ASCII character would arrive as two bytes and be
            // silently split into two separate symbols.
            if (c >= 128)
                throw std::invalid_argument(
                    "nwgrad: alphabet symbols must be ASCII; got a byte >= 0x80 "
                    "(a non-ASCII character encoded as UTF-8?)");

            // '-' is how aligned() renders a gap; a '-' symbol would make its
            // output ambiguous and break guide_j_from_aligned().
            if (ch == '-')
                throw std::invalid_argument(
                    "nwgrad: '-' cannot be an alphabet symbol; it is reserved as "
                    "the gap marker in aligned sequences");

            if (lut_[c] >= 0)
                throw std::invalid_argument(
                    std::string("nwgrad: duplicate symbol '") + ch +
                    "' in alphabet \"" + symbols_ + "\"");
            lut_[c] = static_cast<int16_t>(i);
        }
    }

    [[noreturn]] void throw_bad_char(char c, size_t pos) const {
        std::string msg = "nwgrad: character '";
        msg += c;
        msg += "' (0x";
        const char* hex = "0123456789abcdef";
        auto u = static_cast<unsigned char>(c);
        msg += hex[u >> 4];
        msg += hex[u & 0xf];
        msg += ") at position " + std::to_string(pos) +
               " is not in alphabet \"" + symbols_ + "\"";
        throw std::invalid_argument(msg);
    }

    std::string symbols_;
    int         n_;
    // -1 = not in this alphabet.  int16_t, not int8_t: an alphabet may have up
    // to 255 symbols, and indices 128..254 do not fit in a signed byte -- they
    // would wrap negative and the alphabet would reject its own symbols.
    int16_t     lut_[256];
};

// The registry owns every Alphabet ever created, forever.  Defined inline so
// the library stays header-only; `inline` gives one shared instance across all
// translation units.
inline const Alphabet& Alphabet::get(std::string_view symbols) {
    static std::mutex mu;

    // The registry itself is deliberately leaked, not merely its contents.  A
    // plain function-local static map would be destroyed during static
    // teardown, which would orphan every Alphabet it owns -- and an Alphabet*
    // held by some other static (a SubstMatrix at namespace scope, say) may
    // still be dereferenced at that point, with destruction order between them
    // unspecified.  Immortality is the contract this class advertises, so it
    // has to hold all the way through exit.
    //
    // It is also what LeakSanitizer needs: the map is reachable from a root for
    // the whole run, so the Alphabets it points at are "still reachable" rather
    // than leaked.
    static auto* registry = new std::map<std::string, const Alphabet*, std::less<>>();

    std::lock_guard<std::mutex> lock(mu);
    auto it = registry->find(symbols);
    if (it != registry->end()) return *it->second;

    const Alphabet* a = new Alphabet(symbols);
    registry->emplace(std::string(symbols), a);
    return *a;
}
