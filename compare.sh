#!/bin/bash
prof() {
  rm -rf /tmp/p; rocprofv3 --kernel-trace --stats --output-format csv \
    --output-directory /tmp/p -- "$1" > /dev/null 2>&1
  awk -F, 'NR==2{print $(NF-2)}' /tmp/p/*/*_kernel_stats.csv
}
B=$(prof /tmp/softmax_run/build/softmax_baseline)
C=$(prof /tmp/softmax_run/build/softmax_candidate)
echo
echo "  rocprofv3 - independent, outside the harness"
echo "  ------------------------------------------------"
printf "  baseline    %s ns\n" "$B"
printf "  candidate   %s ns\n" "$C"
echo
