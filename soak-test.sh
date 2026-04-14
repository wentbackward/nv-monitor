#!/bin/bash
#
# soak-test.sh — Memory leak and stability soak test for nv-monitor
#
# Runs nv-monitor at maximum collection frequency while hammering the
# Prometheus endpoint. Monitors RSS over time to detect memory growth.
#
# Usage: ./soak-test.sh [duration_minutes]
#        Default: 10 minutes

set -e

DURATION_MIN=${1:-10}
DURATION_SEC=$((DURATION_MIN * 60))
NV_MONITOR="./nv-monitor"
PROM_PORT=9177
LOG_FILE="/tmp/soak-test-data.csv"
SCRAPE_INTERVAL=0.5  # seconds between prometheus scrapes
RSS_SAMPLE_INTERVAL=5  # seconds between RSS samples

echo "========================================"
echo " nv-monitor soak test"
echo "========================================"
echo " Duration:  ${DURATION_MIN} minutes"
echo " Log file:  ${LOG_FILE}"
echo " Prom port: ${PROM_PORT}"
echo " Log interval: 100ms (maximum collection rate)"
echo " Scrape interval: ${SCRAPE_INTERVAL}s"
echo "========================================"
echo ""

# Clean up on exit
cleanup() {
    echo ""
    echo "Stopping..."
    kill $NV_PID $SCRAPER_PID 2>/dev/null
    wait $NV_PID $SCRAPER_PID 2>/dev/null
    rm -f "$LOG_FILE"
}
trap cleanup EXIT

# Start nv-monitor: headless, fastest logging, prometheus enabled
"$NV_MONITOR" -n -l "$LOG_FILE" -i 100 -p $PROM_PORT &>/dev/null &
NV_PID=$!
sleep 2

if ! kill -0 $NV_PID 2>/dev/null; then
    echo "FAIL: nv-monitor failed to start"
    exit 1
fi

echo "nv-monitor running (PID $NV_PID)"

# Get initial RSS
RSS_START=$(grep VmRSS /proc/$NV_PID/status 2>/dev/null | awk '{print $2}')
echo "Initial RSS: ${RSS_START} KB"
echo ""

# Prometheus scraper — hits /metrics as fast as configured
(
    while kill -0 $NV_PID 2>/dev/null; do
        curl -s "http://localhost:${PROM_PORT}/metrics" > /dev/null 2>&1
        sleep $SCRAPE_INTERVAL
    done
) &
SCRAPER_PID=$!

# Monitor RSS over time
echo "Time       RSS (KB)    CSV lines   Prom scrapes   Delta RSS"
echo "---------- ----------- ----------- -------------- ---------"

SCRAPE_COUNT=0
START_TIME=$(date +%s)

while true; do
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ $ELAPSED -ge $DURATION_SEC ]; then
        break
    fi

    if ! kill -0 $NV_PID 2>/dev/null; then
        echo ""
        echo "FAIL: nv-monitor crashed after ${ELAPSED}s"
        exit 1
    fi

    RSS_NOW=$(grep VmRSS /proc/$NV_PID/status 2>/dev/null | awk '{print $2}')
    CSV_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)

    # Count prometheus scrapes (approximate from curl loop)
    SCRAPE_COUNT=$(echo "$ELAPSED / $SCRAPE_INTERVAL" | bc 2>/dev/null || echo "?")

    DELTA=$((RSS_NOW - RSS_START))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))

    printf "%3dm %02ds    %-11s %-11s %-14s %+d KB\n" \
        $MINS $SECS "$RSS_NOW" "$CSV_LINES" "$SCRAPE_COUNT" "$DELTA"

    sleep $RSS_SAMPLE_INTERVAL
done

# Final stats
echo ""
echo "========================================"
echo " Results after ${DURATION_MIN} minutes"
echo "========================================"
RSS_END=$(grep VmRSS /proc/$NV_PID/status 2>/dev/null | awk '{print $2}')
CSV_FINAL=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
CSV_SIZE=$(du -h "$LOG_FILE" 2>/dev/null | awk '{print $1}')
DELTA=$((RSS_END - RSS_START))

echo "RSS start:      ${RSS_START} KB"
echo "RSS end:        ${RSS_END} KB"
echo "RSS delta:      ${DELTA} KB"
echo "CSV rows:       ${CSV_FINAL}"
echo "CSV size:       ${CSV_SIZE}"
echo "Prom scrapes:   ~$((DURATION_SEC * 2))"
echo ""

if [ $DELTA -gt 1024 ]; then
    echo "WARNING: RSS grew by ${DELTA} KB — possible memory leak"
    exit 1
elif [ $DELTA -gt 256 ]; then
    echo "NOTE: RSS grew by ${DELTA} KB — minor, likely kernel page cache"
    exit 0
else
    echo "PASS: RSS stable (delta ${DELTA} KB)"
    exit 0
fi
