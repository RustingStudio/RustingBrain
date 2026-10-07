#!/usr/bin/env bash
# Throughput regression check. Runs bench_train_step (CUDA) and bench_decode
# (CPU) five times each, takes the best tokens/s, and fails if either is more
# than 3% below its row in docs/baseline.md ("Regression check" section).
# CI has no GPU: run this locally, on an idle card, before every release.
#
#   scripts/bench_check.sh            compare against the baseline
#   scripts/bench_check.sh --record   print the rows to paste into the baseline
set -euo pipefail
cd "$(dirname "$0")/.."

others=$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | grep -v gnome-clocks || true)
if [[ -n "$others" ]]; then
    echo "warning: the GPU is shared, numbers will be low:" >&2
    echo "$others" >&2
fi

# bench_decode runs on the CPU, and a parallel build halves it.
read -r load _ < /proc/loadavg
if (( ${load%.*} >= 2 )); then
    echo "warning: load average $load, bench_decode will be low" >&2
fi

cargo build --release --quiet --features cuda --example bench_train_step --example bench_decode

# Best tokens/s over five runs: one run spreads about 3.5%, the best of five
# much less, and a regression moves the best too.
best() {
    for _ in 1 2 3 4 5; do "$@"; done | sed -E 's/.* ([0-9]+) tokens\/s$/\1/' | sort -n | tail -1
}

status=0
check() {
    local name=$1; shift
    local measured expected
    measured=$(best "$@")
    if [[ $record == 1 ]]; then
        echo "| $name | $measured |"
        return
    fi
    expected=$(grep -F "| $name |" docs/baseline.md | sed -E 's/.*\| ([0-9]+) \|$/\1/')
    if [[ -z "$expected" ]]; then
        echo "$name: no baseline row in docs/baseline.md" >&2
        status=1
        return
    fi
    if (( measured * 100 < expected * 97 )); then
        echo "FAIL $name: $measured tokens/s, baseline $expected (more than 3% slower)"
        status=1
    else
        echo "ok   $name: $measured tokens/s, baseline $expected"
    fi
}

record=0
[[ ${1:-} == --record ]] && record=1
check "bench_train_step 4 1024 cuda" ./target/release/examples/bench_train_step 4 1024 cuda
check "bench_decode 512 128" ./target/release/examples/bench_decode 512 128
exit $status
