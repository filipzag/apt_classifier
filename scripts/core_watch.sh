#!/usr/bin/env bash
# Run a command; kill it if available memory drops below 700 MB. Prints peak RSS (KB) at the end.
"$@" & pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null || echo 0)
  [ "${rss:-0}" -gt "$peak" ] && peak=$rss
  if [ "$avail" -lt 700 ]; then kill $pid; echo "KILLED: only ${avail} MB available" >&2; fi
  sleep 1
done
wait $pid; rc=$?
echo "peak_rss_mb=$((peak/1024)) exit=$rc" >&2
exit $rc
