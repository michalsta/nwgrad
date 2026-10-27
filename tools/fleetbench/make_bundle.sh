#!/usr/bin/env bash
# Assemble the per-host bundle: nwgrad source (a commit of this repo), this harness, and the
# competitors' sources at the versions measured -- headers and sources only, ~2.6 MB as .tar.xz,
# small enough for metered hosts.  Binary competitors (parasail, Biopython, pyopal, edlib,
# scikit-bio, DeepBLAST) come from PyPI in setup_host.sh instead.
#   make_bundle.sh OUT.tar.xz [NWGRAD_COMMIT]          (run inside the nwgrad repo)
set -euo pipefail
OUT=$(realpath "$1"); REV=${2:-HEAD}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
B=$W/bundle; mkdir -p "$B/ext"
git -C "$REPO" archive --format=tar --prefix=nwgrad/ "$REV" | tar -x -C "$B"
mkdir -p "$B/bench" && cp -r "$HERE"/*.py "$HERE"/*.sh "$HERE"/drivers "$B/bench/"
clone() {   # url ref dir
    git clone -q --depth 1 --branch "$2" "$1" "$W/src/$3" 2>/dev/null ||
        { git clone -q "$1" "$W/src/$3" && git -C "$W/src/$3" checkout -q "$2"; }
}
clone https://github.com/seqan/seqan3 3.4.2 seqan3
clone https://github.com/seqan/seqan seqan-v2.5.3 seqan2
clone https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library a66636b79ef36ac178122437053be0d8ef345271 ssw
clone https://github.com/lh3/ksw2 289609bd9e5381a13b16239d0a7703f1ff03f9ca ksw2
clone https://github.com/smarco/WFA2-lib v2.3.6 wfa2
mkdir -p "$B/ext/seqan3" "$B/ext/seqan2" "$B/ext/ssw" "$B/ext/ksw2"
cp -r "$W/src/seqan3/include" "$W/src/seqan3/LICENSE.md" "$B/ext/seqan3/"
cp -r "$W/src/seqan2/include" "$B/ext/seqan2/"
cp "$W/src/ssw/src/"{ssw.c,ssw.h,sse2neon.h} "$B/ext/ssw/"
cp "$W/src/ksw2/"{ksw2.h,ksw2_extz2_sse.c,kalloc.c,kalloc.h,LICENSE.txt} "$B/ext/ksw2/"
git -C "$W/src/wfa2" archive --format=tar --prefix=WFA2-lib/ HEAD | tar -x -C "$B/ext"
rm -rf "$B/ext/WFA2-lib/"{tests,docs,img,examples}
tar -C "$W" -cJf "$OUT" bundle
echo "$OUT: $(du -h "$OUT" | cut -f1)"
