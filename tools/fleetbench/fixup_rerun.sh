#!/usr/bin/env bash
# Re-run arms on a FINISHED host after a harness fix:
#   fixup_rerun.sh HOST timeouts    -- every "timeout" record (warm-up allowance fix);
#                                      all C/C++ drivers are rebuilt (they share drv_common.hpp)
# Pushes the harness, drops the records, clears the done/setup markers; the coordinator then
# re-runs setup and a runner, which runs only the missing arms.
h=$1; what=$2
cd "$(dirname "$0")/.."
case $h in
  spot|solace) tar -C bench -cJf work/bench.tar.xz . && scp -q work/bench.tar.xz $h:/tmp/bench.tar.xz && ssh $h 'tar -xJf /tmp/bench.tar.xz -C ~/nwbench/bench' ;;
  *) rsync -a bench/ $h:nwbench/bench/ ;;
esac
ssh $h "cd ~/nwbench && find bin -maxdepth 1 -type f -delete && rm -f .setup_done results/$h.done && cp results/$h.jsonl results/$h.jsonl.bak-$what && venv-gcc/bin/python -I -c '
import json
p=\"results/$h.jsonl\"; L=open(p).readlines()
keep=[l for l in L if json.loads(l)[\"status\"]!=\"timeout\"]
open(p,\"w\").writelines(keep); print(\"$h:\", len(L)-len(keep), \"timeout records dropped\")'"
rm -f "results/$h.done"   # the coordinator syncs without --delete
