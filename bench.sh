#!/bin/bash
#
# bench.sh — Resource usage benchmark for nv-monitor vs top vs htop
#
# Run on the target device:
#   chmod +x bench.sh
#   ./bench.sh
#
# Outputs a summary to stdout. No files sent, just paste the results.

set -e

NV_MONITOR="./nv-monitor"
DURATION=10
CLK=$(getconf CLK_TCK)

echo "========================================"
echo " nv-monitor resource benchmark"
echo "========================================"
echo ""
echo "--- System ---"
uname -m
cat /sys/firmware/devicetree/base/model 2>/dev/null || grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2
echo ""
echo "CPUs: $(nproc)"
free -h | head -2
echo ""
echo "Binary sizes:"
ls -lh "$NV_MONITOR" 2>/dev/null | awk '{print "  nv-monitor:", $5}'
ls -lh $(which top 2>/dev/null) 2>/dev/null | awk '{print "  top:       ", $5}'
ls -lh $(which htop 2>/dev/null) 2>/dev/null | awk '{print "  htop:      ", $5}'
echo ""

measure() {
    local NAME="$1"
    shift
    "$@" &>/dev/null &
    local PID=$!
    sleep 2

    # Memory
    local RSS=$(grep VmRSS /proc/$PID/status 2>/dev/null | awk '{print $2}')
    local PRIVATE=$(grep Private_Dirty /proc/$PID/smaps_rollup 2>/dev/null | awk '{print $2}')
    local SHARED=$(grep Shared_Clean /proc/$PID/smaps_rollup 2>/dev/null | awk '{print $2}')
    local THREADS=$(grep Threads /proc/$PID/status 2>/dev/null | awk '{print $2}')

    # CPU over measurement period
    local CPU1=$(cat /proc/$PID/stat 2>/dev/null | awk '{print $14+$15}')
    sleep $DURATION
    local CPU2=$(cat /proc/$PID/stat 2>/dev/null | awk '{print $14+$15}')
    local CPU_PCT=$(echo "scale=2; ($CPU2-$CPU1)*100/$CLK/$DURATION" | bc 2>/dev/null)

    kill $PID 2>/dev/null; wait $PID 2>/dev/null

    printf "%-35s RSS: %6s KB  Private: %6s KB  Shared: %6s KB  Threads: %s  CPU: %s%%\n" \
        "$NAME" "${RSS:-?}" "${PRIVATE:-?}" "${SHARED:-?}" "${THREADS:-?}" "${CPU_PCT:-?}"
}

echo "--- Measurements (${DURATION}s sample) ---"
echo ""

if [ -x "$NV_MONITOR" ]; then
    measure "nv-monitor (headless)" "$NV_MONITOR" -n -l /dev/null
    measure "nv-monitor (headless+prometheus)" "$NV_MONITOR" -n -l /dev/null -p 9198
fi

if command -v top &>/dev/null; then
    measure "top (batch mode)" top -b -d 1
fi

if command -v htop &>/dev/null; then
    measure "htop" script -qc "htop -d 10" /dev/null
fi

echo ""
echo "--- NVML memory breakdown ---"
if [ -x "$NV_MONITOR" ]; then
    "$NV_MONITOR" -n -l /dev/null &>/dev/null &
    NV_PID=$!
    sleep 2
    echo "NVML mapped regions:"
    grep nvidia /proc/$NV_PID/maps 2>/dev/null | while read line; do
        RANGE=$(echo "$line" | awk '{print $1}')
        START=$(echo "$RANGE" | cut -d- -f1)
        END=$(echo "$RANGE" | cut -d- -f2)
        SIZE=$(python3 -c "print(f'{(int(\"$END\",16)-int(\"$START\",16))/1024:.0f} KB')" 2>/dev/null || echo "? KB")
        PERMS=$(echo "$line" | awk '{print $2}')
        FILE=$(echo "$line" | awk '{print $NF}')
        echo "  $PERMS  $SIZE  $FILE"
    done
    if ! grep -q nvidia /proc/$NV_PID/maps 2>/dev/null; then
        echo "  (no NVML loaded)"
    fi
    kill $NV_PID 2>/dev/null; wait $NV_PID 2>/dev/null
fi

echo ""
echo "--- Notes ---"
echo "RSS includes shared library pages mapped by other GPU processes."
echo "Private_Dirty is the unique memory cost of each process."
echo "CPU% measured over ${DURATION}s at default refresh rate."
echo ""
echo "Done."
