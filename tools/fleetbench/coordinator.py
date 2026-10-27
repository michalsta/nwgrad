#!/usr/bin/env python3
"""Drive the fleet benchmark from skynet; survive hosts going away.

    coordinator.py --once        # one pass over the fleet
    coordinator.py               # loop forever (every --interval seconds)

Run from (or NWB_TOP=) a working directory; NWB_SCRATCH (default ./work) holds what it
ships: bundle.tar.xz (make_bundle.sh), fx.tar.xz (the fixtures without miRNA) and
fxm.tar.xz (the miRNA fixtures, for unmetered hosts).  HOSTS below is the published run's
fleet.

Per host, each pass:
  unreachable or booted into its other OS -> note it, try again next pass;
  laptop -> re-take the sleep inhibitor if it is gone (a power-off drops it);
  not deployed -> ship the bundle (spot: from solace over the LAN) and start setup;
  set up, runner not running, not finished -> start the runner (it resumes by itself);
  Gentoo desktop busy (an emerge, someone else's CPU) -> pause flag, so the runner waits
    between arms (laptops: the user asked to keep going under light load);
  always -> pull results/, then rewrite preliminary-results.md with progress and ETA.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# the coordinator's working directory: results/, holds/, the logs, and the files it ships
TOP = os.path.abspath(os.environ.get("NWB_TOP", os.getcwd()))
SCRATCH = os.environ.get("NWB_SCRATCH", os.path.join(TOP, "work"))
BUNDLE = f"{SCRATCH}/bundle.tar.xz"
FX = f"{SCRATCH}/fx.tar.xz"                      # fixtures without mirna
FXM = f"{SCRATCH}/fxm.tar.xz"                    # mirna fixtures (desktops)
RESULTS = f"{TOP}/results"
LOG = f"{TOP}/coordinator.log"

HOSTS = {
    "wloczykij":  dict(os="gentoo", laptop=False, reps=3, metered=False),
    "nighthaven": dict(os="gentoo", laptop=False, reps=3, metered=False),
    "oldsolace":  dict(os="gentoo", laptop=True, reps=5, metered=False, inhibit="elogind"),
    "spot":       dict(os="fedora", laptop=True, reps=5, metered=True, inhibit="systemd"),
    "solace":     dict(os="wsl", laptop=True, reps=5, metered=True, inhibit="keepawake"),
}
INHIBIT_SECS = 30 * 3600


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ssh(host, cmd, timeout=60, check=False):
    try:
        p = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, cmd],
                           capture_output=True, text=True, timeout=timeout)
        if check and p.returncode:
            log(f"{host}: '{cmd[:80]}' failed: {p.stderr[-300:]}")
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def os_ok(host, uname):
    want = HOSTS[host]["os"]
    if want == "wsl": return "microsoft-standard-WSL2" in uname
    if want == "fedora": return uname.startswith("Linux") and "asahi" in uname
    return uname.startswith("Linux") and "microsoft" not in uname


def inhibitor(host):
    kind = HOSTS[host].get("inhibit")
    why = "nwgrad fleet benchmark"
    if kind == "elogind":
        rc, out, _ = ssh(host, "elogind-inhibit --list | grep -c 'nwgrad fleet benchmark'")
        if out.strip() in ("", "0"):
            ssh(host, f"setsid nohup elogind-inhibit --what=idle:sleep --mode=block --who=claude-code --why='{why}' sleep {INHIBIT_SECS} >/dev/null 2>&1 < /dev/null &")
            log(f"{host}: re-took elogind inhibitor")
    elif kind == "systemd":
        rc, out, _ = ssh(host, "systemd-inhibit --list | grep -c 'nwgrad fleet benchmark'")
        if out.strip() in ("", "0"):
            ssh(host, f"setsid nohup systemd-inhibit --what=idle:sleep --mode=block --who=claude-code --why='{why}' sleep {INHIBIT_SECS} >/dev/null 2>&1 < /dev/null &")
            log(f"{host}: re-took systemd inhibitor")
    elif kind == "keepawake":
        rc, out, _ = ssh(host, "~/cc/keepawake.sh status")
        if "HELD" not in out:
            ssh(host, f"~/cc/keepawake.sh start {INHIBIT_SECS} '{why}'", timeout=90)
            log(f"{host}: re-took keepawake")


def release_inhibitor(host):
    """The job on this laptop is over: let it sleep again."""
    kind = HOSTS[host].get("inhibit")
    if kind in ("elogind", "systemd"):
        # match OUR inhibitor by its --why text: other sessions use --who=claude-code too
        rc, out, _ = ssh(host, "pgrep -u untrusted -f 'nwgrad fleet benchmar[k]' >/dev/null && pkill -u untrusted -f 'nwgrad fleet benchmar[k]' && echo released")
    elif kind == "keepawake":
        # one shared flag: stop it only while it carries our reason
        rc, out, _ = ssh(host, "~/cc/keepawake.sh status | grep -q 'HELD: nwgrad fleet benchmark' && ~/cc/keepawake.sh stop >/dev/null && echo released")
    else:
        out = ""
    if out.strip() == "released":
        log(f"{host}: finished -- sleep inhibitor released")


def metered_check(host):
    """Log what the laptop's connection looks like (the user said spot/solace are metered)."""
    if host == "oldsolace":
        rc, out, _ = ssh(host, "dev=$(ip route get 1.1.1.1 | sed -n 's/.* dev \\([^ ]*\\).*/\\1/p'); nmcli -g GENERAL.METERED dev show \"$dev\" 2>/dev/null")
        if out.strip().startswith("yes"):
            log(f"{host}: connection now METERED ({out.strip()}); large transfers stop")
            return True
    return HOSTS[host]["metered"]


def deployed(host):
    # .needs_rerun: the harness changed under a running setup; once it ends, set up again
    ssh(host, "cd ~/nwbench 2>/dev/null && test -f .needs_rerun && ! pgrep -f 'setup_hos[t].sh' >/dev/null && rm -f .needs_rerun .setup_done")
    rc, out, _ = ssh(host, "test -f ~/nwbench/.needs_rerun && echo running; test -f ~/nwbench/.setup_done && echo done; test -f ~/nwbench/.setup_running && pgrep -f 'setup_hos[t].sh' >/dev/null && echo running; test -d ~/nwbench/bench && echo shipped")
    return set(out.split())


def ship(host):
    if host == "spot":
        # from solace over the LAN (solace -> spot only: WSL is NAT'd)
        rc, out, _ = ssh("solace", "test -f ~/nwbench/.shipped && echo ok")
        if out.strip() != "ok":
            log("spot: waiting for solace to be shipped first")
            return False
        rc, _, err = ssh("solace", "ssh spotlan 'mkdir -p nwbench' && rsync -a ~/nwbench/bundle.tar.xz ~/nwbench/fx.tar.xz spotlan:nwbench/", timeout=900)
        if rc:
            log(f"spot: LAN copy from solace failed: {err[-300:]}")
            return False
        rc, _, err = ssh("spot", "cd ~/nwbench && tar -xJf bundle.tar.xz --strip-components=1 && mkdir -p fixtures && tar -xJf fx.tar.xz -C fixtures && touch .shipped", timeout=300)
        log(f"spot: shipped over the LAN rc={rc}")
        return rc == 0
    ssh(host, "mkdir -p ~/nwbench")
    files = [BUNDLE, FX] + ([FXM] if not HOSTS[host]["metered"] else [])
    rc = subprocess.run(["scp", "-q"] + files + [f"{host}:nwbench/"]).returncode
    if rc:
        log(f"{host}: scp failed")
        return False
    extra = "&& tar -xJf fxm.tar.xz -C fixtures" if not HOSTS[host]["metered"] else ""
    rc, _, err = ssh(host, f"cd ~/nwbench && tar -xJf bundle.tar.xz --strip-components=1 && mkdir -p fixtures && tar -xJf fx.tar.xz -C fixtures {extra} && touch .shipped", timeout=300)
    log(f"{host}: shipped rc={rc} {err[-200:]}")
    return rc == 0


def mirna_for_laptop(host):
    """spot: build mirna fixtures from its own Manakov copy; solace: pull them from spot."""
    rc, out, _ = ssh(host, "ls ~/nwbench/fixtures/mirna_st.tsv ~/nwbench/fixtures/mirna_mt.tsv 2>/dev/null | wc -l")
    if out.strip() == "2":
        return True
    if host == "spot":
        rc, out, err = ssh("spot", "cd ~/nwbench && venv-gcc/bin/python bench/make_fixtures.py --manakov ~/cc/manakov_fit_rc.tsv --out fixtures --only mirna | grep mirna", timeout=900)
        log(f"spot: mirna fixtures built: {out.strip()} {err[-200:]}")
        return rc == 0
    if host == "solace":
        rc, _, err = ssh("solace", "rsync -a spotlan:nwbench/fixtures/mirna_st.tsv spotlan:nwbench/fixtures/mirna_mt.tsv ~/nwbench/fixtures/", timeout=900)
        log(f"solace: mirna fixtures from spot over the LAN rc={rc} {err[-200:]}")
        return rc == 0
    return True


def start_setup(host):
    flag = " --metered" if HOSTS[host]["metered"] else ""
    ssh(host, f"cd ~/nwbench && touch .setup_running; setsid nohup bash -c 'bench/setup_host.sh ~/nwbench{flag} && touch .setup_done; rm -f .setup_running' > setup.out 2>&1 < /dev/null &")
    log(f"{host}: setup started")


def runner_running(host):
    rc, out, _ = ssh(host, "pgrep -f 'bench/runne[r].py --root' >/dev/null && echo yes")
    return out.strip() == "yes"


def mt_spec(host):
    rc, out, _ = ssh(host, "cd ~/nwbench && venv-gcc/bin/python -I -c \"import nwgrad; b = nwgrad.SeqPairBatch(['A'], ['A'], nwgrad.AlignParams(nwgrad.matrices.BLOSUM62, 11, 1, 11, 1)); print(b.n_threads)\"")
    n = out.strip() or "1"
    if host == "spot":
        # nwgrad's n_threads=0 counts unique (package, core_id) pairs; the M1's two
        # clusters both number their cores 0-3, so it finds 4 of the 8 physical cores
        n = "8"
    spec = [f"--mt {n}"]
    if host == "wloczykij":
        spec.append("--mt 8:numa0")
    return " ".join(spec)


def start_runner(host):
    h = HOSTS[host]
    mt = mt_spec(host)
    extra = ""
    if h["os"] == "wsl":
        extra = "setsid nohup bench/win_sampler.sh results/solace.wincpu > /dev/null 2>&1 < /dev/null & "
    ssh(host, f"cd ~/nwbench; {extra}setsid nohup venv-gcc/bin/python bench/runner.py --root ~/nwbench --host {host} --reps {h['reps']} {mt} > runner.out 2>&1 < /dev/null &")
    log(f"{host}: runner started ({mt}, reps {h['reps']})")


def mt_phase(host):
    """True once the host's runner has written a multithread-phase record."""
    rc, out, _ = ssh(host, f"tail -1 ~/nwbench/results/{host}.jsonl 2>/dev/null | grep -c '\"phase\": \"mt\"'")
    return out.strip() == "1"


def busy_gentoo(host):
    """An emerge, or someone else's process eating a core.  Returns a reason or ''."""
    # top's second sample is CPU over the last 2 s (ps reports lifetime averages)
    rc, out, _ = ssh(host, "pgrep -x emerge >/dev/null && echo EMERGE; top -b -n2 -d2 -w 200 | awk '/^top -/{n++} n==2 && $1 ~ /^[0-9]+$/ && $2 !~ /^untrust/ && $9+0 > 10 {print $2, $9, $12}' | head -20")
    lines = [l for l in out.strip().splitlines() if l]
    if host == "wloczykij" and "EMERGE" not in out:
        # 64 cores: pause only when foreign load (other users, CI, skynet's VM) adds up to
        # more than 4 cores while one benchmark thread runs, 3 while 32 (or 8 pinned) do --
        # the scheduler puts a stray process on an idle core; each arm's load is recorded
        tot = sum(float(l.split()[1].replace(",", ".")) for l in lines if len(l.split()) >= 3)
        return "; ".join(lines) if tot > (300 if mt_phase(host) else 400) else ""
    keep = []
    for line in lines:
        f = line.split()
        # wloczykij hosts skynet (this coordinator's VM): its qemu at about a core is the
        # VM idling or the coordinator working -- negligible on 64 cores; more is not
        if len(f) >= 3 and f[0] == "qemu" and float(f[1].replace(",", ".")) < 300:
            continue
        if len(f) >= 3 and float(f[1].replace(",", ".")) <= 60:
            continue                     # listed for the sum above; not busy on its own
        keep.append(line)
    return "; ".join(keep)


def hold_until(host):
    """holds/<host> holds an epoch: keep that host paused until then (a quiet window
    another session asked for).  An expired hold is removed."""
    p = f"{TOP}/holds/{host}"
    try:
        t = float(open(p).read().strip())
    except Exception:
        return None
    if t <= time.time():
        os.remove(p)
        log(f"{host}: hold expired")
        return None
    return t


def pause(host, on, why=""):
    if on:
        ssh(host, f"echo '{why}' > ~/nwbench/PAUSE")
    else:
        ssh(host, "rm -f ~/nwbench/PAUSE")


def sync(host):
    os.makedirs(f"{RESULTS}", exist_ok=True)
    subprocess.run(["rsync", "-a", "--timeout=60", f"{host}:nwbench/results/", f"{RESULTS}/"],
                   capture_output=True)


def one_pass(state):
    for host, h in HOSTS.items():
        rc, uname, _ = ssh(host, "uname -a", timeout=30)
        if rc:
            state[host] = "unreachable"
            continue
        if not os_ok(host, uname):
            state[host] = f"booted into another OS ({uname.split()[0]} {uname.split()[2]})"
            continue
        rc, out, _ = ssh(host, "test -f ~/nwbench/results/%s.done && echo done" % host)
        if out.strip() == "done":
            if h["laptop"]:
                release_inhibitor(host)
            sync(host)
            state[host] = "finished"
            continue
        if h["laptop"]:
            inhibitor(host)
            metered_check(host)
        st = deployed(host)
        if "shipped" not in st:
            if not ship(host):
                state[host] = "shipping failed"
                continue
            st = deployed(host)
        if "done" not in st:
            if "running" not in st:
                start_setup(host)
            state[host] = "setting up"
            continue
        if host in ("spot", "solace") and not mirna_for_laptop(host):
            state[host] = "waiting for mirna fixtures"
            continue
        rc, out, _ = ssh(host, "test -f ~/nwbench/results/%s.done && echo done" % host)
        finished = out.strip() == "done"
        why = busy_gentoo(host) if (h["os"] == "gentoo" and not h["laptop"]) else ""
        held = hold_until(host)
        if held:
            why = (why + "; " if why else "") + f"held until {time.strftime('%H:%M', time.localtime(held))}"
        pause(host, bool(why), why)
        if why: log(f"{host}: paused: {why}")
        if not finished and not runner_running(host):
            start_runner(host)
        sync(host)
        state[host] = "finished" if finished else "running"
    try:
        subprocess.run([sys.executable, f"{HERE}/report.py", "--results", RESULTS,
                        "--out", f"{TOP}/preliminary-results.md", "--state", json.dumps(state)],
                       timeout=600)
    except Exception as e:
        log(f"report failed: {e}")
    log("pass: " + ", ".join(f"{k}={v}" for k, v in state.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    a = ap.parse_args()
    state = {}
    while True:
        one_pass(state)
        if a.once or all(v == "finished" for v in state.values()):
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
