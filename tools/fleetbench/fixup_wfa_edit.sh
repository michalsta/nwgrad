#!/usr/bin/env bash
# Re-run the WFA2 edit-metric arms on a finished host after the driver's sign fix:
# push the harness, drop those records, force a wfa2_drv rebuild; the coordinator then
# re-runs setup and a runner, which runs only the missing arms.
h=$1
cd "$(dirname "$0")/.."
case $h in
  spot|solace) tar -C bench -cJf work/bench.tar.xz . && scp -q work/bench.tar.xz $h:/tmp/bench.tar.xz && ssh $h 'tar -xJf /tmp/bench.tar.xz -C ~/nwbench/bench' ;;
  *) rsync -a bench/ $h:nwbench/bench/ ;;
esac
ssh $h "cd ~/nwbench && rm -f bin/wfa2_drv .setup_done results/$h.done && cp results/$h.jsonl results/$h.jsonl.bak && venv-gcc/bin/python -I -c '
import json
p=\"results/$h.jsonl\"; L=open(p).readlines()
keep=[l for l in L if json.loads(l)[\"variant\"]!=\"metric=edit\"]
open(p,\"w\").writelines(keep); print(\"$h:\", len(L)-len(keep), \"records dropped\")'"
rm -f "results/$h.done"   # the coordinator syncs without --delete
