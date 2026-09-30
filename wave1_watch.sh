#!/bin/bash
# Poll until all 70 Wave-1 jobs have left the queue, then summarise.
cd "$(dirname "$0")"
LOG=slurm_logs/wave1_watch.log
TARGET=70
for i in $(seq 1 300); do   # 300 x 5min = 25h safety bound
    RUN=$(squeue -u "$USER" -h -t RUNNING -o "%j" | grep -c '^w1_')
    PEND=$(squeue -u "$USER" -h -t PENDING -o "%j" | grep -c '^w1_')
    DONE=$(sacct -n -X -u "$USER" --starttime=2026-08-28T04:00 --format=JobName%40,State \
           | awk '$1 ~ /^w1_/ && $2=="COMPLETED" {print $1}' | sort -u | wc -l)
    BAD=$(sacct -n -X -u "$USER" --starttime=2026-08-28T04:00 --format=JobName%40,State \
          | awk '$1 ~ /^w1_/ && $2!="COMPLETED" && $2!="RUNNING" && $2!="PENDING" {print $1" "$2}' | sort -u)
    echo "$(date '+%F %T') done=$DONE/$TARGET running=$RUN pending=$PEND" >> "$LOG"
    if [ "$RUN" -eq 0 ] && [ "$PEND" -eq 0 ] && [ -f slurm_logs/wave1_dripfeed_done.marker ]; then
        echo "$(date '+%F %T') WAVE1 QUEUE EMPTY — done=$DONE" >> "$LOG"
        echo "WAVE1_FINISHED completed=$DONE/$TARGET"
        [ -n "$BAD" ] && { echo "NON-COMPLETED JOBS:"; echo "$BAD"; }
        exit 0
    fi
    sleep 300
done
echo "WATCHER_TIMEOUT after 25h"; exit 1
