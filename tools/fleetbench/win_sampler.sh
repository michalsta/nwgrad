#!/usr/bin/env bash
# solace under WSL: WSL's own load average does not see Windows' processes.  Append
# "<epoch> <Windows total CPU %>" every ~5 s to $1, while the runner runs.
OUT=$1
cd /mnt/c || exit 1
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
while pgrep -f 'bench/runner.py --root' >/dev/null || [ -z "$started" ]; do
    started=1
    $PS -NoProfile -Command "Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 5 -MaxSamples 12 | ForEach-Object { '{0} {1:N1}' -f ([DateTimeOffset]\$_.Timestamp).ToUnixTimeSeconds(), \$_.CounterSamples[0].CookedValue }" 2>/dev/null | tr -d '\r' >> "$OUT"
done
