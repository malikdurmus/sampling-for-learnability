#!/usr/bin/env bash
# find_free_gpu_nodes.sh -- probe cluster nodes over SSH and report which ones
# have an idle GPU (no CUDA compute process running on it).
#
# Slurm's view (sinfo/squeue) says nothing about actual GPU occupancy on these
# CIP machines, so we ask each node directly with nvidia-smi.
#
# Prints the names of free nodes, one per line, on stdout (pipe-friendly);
# all progress/diagnostics go to stderr.
#
#   ./find_free_gpu_nodes.sh                 # NvidiaAll partition, names only
#   ./find_free_gpu_nodes.sh -v              # full table of every node
#   ./find_free_gpu_nodes.sh -p All -v       # different partition
#   ./find_free_gpu_nodes.sh -n dolomit,chert
#   ssh "$(./find_free_gpu_nodes.sh | head -1)"

set -uo pipefail

PARTITION="NvidiaAll"
NODES=""
JOBS=24
TIMEOUT=10
MEM_MB=0          # ignore compute procs using less than this much VRAM
VERBOSE=0
WATCH=0

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    cat >&2 <<'EOF'

Options:
  -p PARTITION   slurm partition to scan            (default: NvidiaAll)
  -n LIST        comma/space separated node list, overrides -p
  -j N           parallel ssh probes                (default: 24)
  -t SEC         ssh connect timeout in seconds     (default: 10)
  -m MB          treat compute procs smaller than MB as noise (default: 0)
  -v             verbose: print a table of all nodes, not just free ones
  -w SEC         watch: re-scan every SEC seconds until a free node shows up
  -h             this help
EOF
    exit "${1:-0}"
}

while getopts "p:n:j:t:m:w:vh" opt; do
    case "$opt" in
        p) PARTITION="$OPTARG" ;;
        n) NODES="$OPTARG" ;;
        j) JOBS="$OPTARG" ;;
        t) TIMEOUT="$OPTARG" ;;
        m) MEM_MB="$OPTARG" ;;
        w) WATCH="$OPTARG" ;;
        v) VERBOSE=1 ;;
        h) usage 0 ;;
        *) usage 2 ;;
    esac
done

# ---------------------------------------------------------------- node list --
if [[ -n "$NODES" ]]; then
    node_list=$(tr ', ' '\n\n' <<<"$NODES" | sed '/^$/d' | sort -u)
else
    command -v sinfo >/dev/null 2>&1 || { echo "sinfo not found; use -n" >&2; exit 1; }
    node_list=$(sinfo -h -p "$PARTITION" -N -o "%N" 2>/dev/null | sort -u)
fi
[[ -n "$node_list" ]] || { echo "no nodes to probe (partition '$PARTITION')" >&2; exit 1; }
n_nodes=$(wc -l <<<"$node_list")

# ------------------------------------------------------------ remote probe --
# Emits one line:  STATUS|nprocs|used_MiB|total_MiB|util%|gpu_name|proc_details
read -r -d '' REMOTE <<'EOF'
if ! command -v nvidia-smi >/dev/null 2>&1; then echo "NOGPU|||||"; exit 0; fi
gpu=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,name \
        --format=csv,noheader,nounits 2>/dev/null) || { echo "NO-NVIDIA|||||"; exit 0; }
[ -n "$gpu" ] || { echo "NO-NVIDIA|||||"; exit 0; }
used=0; total=0; util=0; name=""
while IFS=, read -r u t g n; do
    u=${u// /}; t=${t// /}; g=${g// /}; n=$(echo "$n" | sed 's/^ *//;s/ *$//')
    used=$((used + u)); total=$((total + t))
    [ "$g" -gt "$util" ] 2>/dev/null && util=$g
    [ -z "$name" ] && name=$n
done <<GPUS
$gpu
GPUS
procs=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null)
n=0; details=""
while IFS=, read -r pid mem; do
    pid=${pid// /}; mem=${mem// /}
    [ -n "$pid" ] || continue
    n=$((n + 1))
    who=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    cmd=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')
    details="$details ${who:-?}:${cmd:-?}:${mem}MiB"
done <<PROCS
$procs
PROCS
echo "OK|$n|$used|$total|$util|$name|$details"
EOF

probe() {
    local node="$1" out status
    out=$(timeout $((TIMEOUT + 10)) ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
              -o ConnectTimeout="$TIMEOUT" -o LogLevel=ERROR "$node" "$REMOTE" 2>/dev/null | tail -1)
    status=$?
    [[ -n "$out" ]] || out="UNREACHABLE|||||"
    printf '%s|%s\n' "$node" "$out"
}
export -f probe
export REMOTE TIMEOUT

# ------------------------------------------------------------ scan + report -
scan() {
echo "probing $n_nodes node(s) in parallel (${JOBS} at a time)..." >&2
results=$(xargs -P "$JOBS" -I{} bash -c 'probe {}' <<<"$node_list" | sort)

free_nodes=()
n_gpu=0
if (( VERBOSE )); then
    printf '%-14s %-12s %5s %14s %6s  %s\n' NODE STATUS PROCS "MEM(MiB)" UTIL DETAILS >&2
    printf '%s\n' "-------------------------------------------------------------------------------" >&2
fi

while IFS='|' read -r node status nproc used total util name details; do
    [[ -n "$node" ]] || continue
    label="$status"
    if [[ "$status" == "OK" ]]; then
        n_gpu=$((n_gpu + 1))
        # count only processes above the noise threshold
        big=0
        for d in $details; do
            m=${d##*:}; m=${m%MiB}
            [[ "$m" =~ ^[0-9]+$ ]] && (( m >= MEM_MB )) && big=$((big + 1))
        done
        if (( big == 0 )); then
            label="FREE"; free_nodes+=("$node")
        else
            label="BUSY"
        fi
    fi
    (( VERBOSE )) && printf '%-14s %-12s %5s %14s %6s  %s\n' \
        "$node" "$label" "${nproc:--}" "${used:--}/${total:--}" "${util:--}" \
        "${details:-${name:-}}" >&2
done <<<"$results"

echo >&2
note=""; (( MEM_MB > 0 )) && note=" larger than ${MEM_MB} MiB"
echo "$(date '+%H:%M:%S'): ${#free_nodes[@]} of $n_gpu NVIDIA node(s) have no GPU compute process${note}:" >&2
printf '%s\n' "${free_nodes[@]+"${free_nodes[@]}"}"
(( ${#free_nodes[@]} > 0 ))
}

if [[ "$WATCH" == "0" ]]; then
    scan
else
    while ! scan; do
        echo "no free node -- re-scanning in ${WATCH}s (Ctrl-C to stop)" >&2
        sleep "$WATCH"
    done
fi
