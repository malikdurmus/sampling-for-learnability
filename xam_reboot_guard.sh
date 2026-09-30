#!/bin/bash
# Reboot guard for the released xam_* XLand jobs (Malik 2026-09-04).
# Cluster reboots Sunday ~04:40; a run takes <=15:45 (limit now 17:00), so any
# job still PENDING on Saturday 10:30 could no longer finish safely and is
# held; after the reboot the held jobs are released again.
# Usage: xam_reboot_guard.sh hold|release   (scheduled via `at`)
LIST=$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability/slurm_logs/xam_held_for_reboot.txt
LOG=$HOME/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability/slurm_logs/xam_reboot_guard.log
case "$1" in
  hold)
    : > "$LIST"
    for J in $(squeue -u "$USER" -h -o '%i %j %T' | awk '$2 ~ /^xam_/ && $3=="PENDING"{print $1}'); do
      scontrol hold "$J" && echo "$J" >> "$LIST" && echo "$(date '+%F %T') held $J" >> "$LOG"
    done
    echo "$(date '+%F %T') hold pass done ($(wc -l < "$LIST") held)" >> "$LOG";;
  release)
    [ -s "$LIST" ] || { echo "$(date '+%F %T') release pass: nothing held" >> "$LOG"; exit 0; }
    # Wait until the reboot has actually happened (an Abaki node reports a
    # BootTime from today) so a late reboot cannot kill freshly started jobs;
    # give up on the boot check after 4h and release anyway.
    TODAY=$(date '+%Y-%m-%d')
    for i in $(seq 1 48); do
      scontrol show node abakus11 2>/dev/null | grep -q "BootTime=${TODAY}" && break
      echo "$(date '+%F %T') release pass: no post-reboot BootTime yet (try $i)" >> "$LOG"
      sleep 300
    done
    # Release with retries: a one-shot scontrol against a mid-reboot
    # controller would fail silently and leave the jobs held forever.
    for i in $(seq 1 36); do
      PEND=0
      while read -r J; do
        scontrol release "$J" >> "$LOG" 2>&1 || PEND=1
      done < "$LIST"
      STILL=$(squeue -u "$USER" -h -o '%r' 2>/dev/null | grep -c JobHeldUser || true)
      echo "$(date '+%F %T') release try $i: still-held=$STILL scontrol-errors=$PEND" >> "$LOG"
      [ "$PEND" -eq 0 ] && [ "${STILL:-1}" -eq 0 ] && { : > "$LIST"; echo "$(date '+%F %T') release pass complete" >> "$LOG"; exit 0; }
      sleep 300
    done
    echo "$(date '+%F %T') release pass GAVE UP after 3h of retries - release manually: scontrol release $(tr '\n' ' ' < "$LIST")" >> "$LOG";;
esac
