#!/usr/bin/env bash
set -e
OUT=$(mktemp -d)
printf "%-28s %6s %10s %10s %10s %10s\n" KERNEL CALLS AVG_NS MIN_NS MAX_NS STDDEV
printf "%s\n" "----------------------------------------------------------------------------------"
for BIN in "$@"; do
  D="$OUT/$(basename "$BIN")"
  rocprofv3 --kernel-trace --stats --output-format csv \
    --output-directory "$D" -- "$BIN" > /dev/null 2>&1
  python3 - "$D" <<'PY'
import csv, glob, sys
for f in glob.glob(sys.argv[1] + "/*/*_kernel_stats.csv"):
    for r in csv.DictReader(open(f)):
        name = r["Name"].split("(")[0][:28]
        print("%-28s %6s %10.1f %10s %10s %10.1f" % (
            name, r["Calls"], float(r["AverageNs"]),
            r["MinNs"], r["MaxNs"], float(r["StdDev"])))
PY
done
