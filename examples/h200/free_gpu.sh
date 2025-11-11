#!/usr/bin/env bash
set -euo pipefail

# Utility to inspect and clean GPU processes across all hosts in a hostfile.
# Default mode only prints VRAM usage; --clean kills all GPU compute processes.

usage() {
  cat <<'EOF'
Usage: bash free_gpu.sh [--hostfile FILE] [--status|--clean|--reset] [--dry-run] [-y]

Options:
  --hostfile FILE   Hostfile to read (defaults to hostfile_nodes.txt next to this script)
  --status          Only print GPU status (default)
  --clean           Kill every process currently holding GPU memory on each host
  --reset           Report GPUs that still have VRAM allocated but no compute process
  --dry-run         Print the commands that would run remotely without executing them
  -y, --yes         Do not prompt before killing processes
  -h, --help        Show this help

Environment overrides:
  SSH_USER          Username to prepend (e.g. SSH_USER=bob -> bob@HOST)
  SSH_PASSWORD      If passwordless SSH is not ready, export DS_SSH_PASSWORD/SSH_PASSWORD
                    and run with sshpass outside this script.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${SCRIPT_DIR}/hostfile_4nodes.txt"
MODE="status"
DRY_RUN=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hostfile)
      HOSTFILE="$2"
      shift 2
      ;;
    --status)
      MODE="status"
      shift
      ;;
    --clean)
      MODE="clean"
      shift
      ;;
    --reset)
      MODE="reset"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -y|--yes)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$HOSTFILE" ]]; then
  echo "Hostfile not found: $HOSTFILE" >&2
  exit 1
fi

mapfile -t HOSTS < <(awk '{print $1}' "$HOSTFILE" | sed 's/#.*//' | sed '/^$/d')
if [[ ${#HOSTS[@]} -eq 0 ]]; then
  echo "Hostfile $HOSTFILE does not contain any hosts." >&2
  exit 1
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes)
if [[ -n "${SSH_USER:-}" ]]; then
  for i in "${!HOSTS[@]}"; do
    HOSTS[$i]="${SSH_USER}@${HOSTS[$i]}"
  done
fi

run_remote() {
  local host="$1"
  shift
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[$host] DRY RUN: $*"
    return 0
  fi
  ssh "${SSH_OPTS[@]}" "$host" "$@"
}

print_status_cmd=$(cat <<'EOF'
set -eo pipefail
echo "=== $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || echo "No compute processes."
EOF
)

clean_cmd=$(cat <<'EOF'
set -eo pipefail
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " ")
if [[ -z "$PIDS" ]]; then
  echo "No GPU processes to kill."
  exit 0
fi
echo "Killing GPU PIDs: $PIDS"
for pid in $PIDS; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" || true
  fi
done
sleep 1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || echo "GPU processes cleared."
EOF
)

reset_cmd=$(cat <<'EOF'
set -eo pipefail
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3/python is required for --reset mode but not found." >&2
  exit 1
fi
"$PYTHON_BIN" <<'PY'
import subprocess, sys

def run(cmd, allow_failure=False):
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0 and not allow_failure:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc.stdout.strip()

try:
    gpu_raw = run(["nvidia-smi",
                   "--query-gpu=index,uuid,memory.used,memory.total",
                   "--format=csv,noheader,nounits"])
except RuntimeError as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)

if not gpu_raw:
    print("No GPUs reported by nvidia-smi.")
    sys.exit(0)

gpu_info = []
for line in gpu_raw.splitlines():
    parts = [field.strip() for field in line.split(",")]
    if len(parts) < 4:
        continue
    idx, uuid, used, total = parts[:4]
    try:
        used_val = int(float(used))
        total_val = int(float(total))
    except ValueError:
        continue
    gpu_info.append((idx, uuid, used_val, total_val))

proc_raw = run(["nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader"], allow_failure=True)
active = {line.strip() for line in proc_raw.splitlines() if line.strip()}

orphan = [
    info for info in gpu_info
    if info[2] > 0 and info[1] not in active
]

if not orphan:
    print("No orphaned VRAM usage detected.")
    sys.exit(0)

print("WARNING: VRAM is allocated with no compute process attached.")
print("Suggested action: ensure the GPU is idle, then run 'sudo nvidia-smi -i <gpu_id> --gpu-reset'.")
for idx, uuid, used, total in orphan:
    pct = (used / total * 100.0) if total else 0.0
    prefix = uuid[:12]
    print(f"  GPU {idx} ({prefix}...) -> {used}/{total} MiB ({pct:.1f}%) still allocated")
PY
EOF
)

if [[ "$MODE" == "clean" && $FORCE -ne 1 ]]; then
  echo "About to kill every GPU process listed by nvidia-smi on ${#HOSTS[@]} hosts:"
  printf '  %s\n' "${HOSTS[@]}"
  read -r -p "Continue? [y/N] " reply
  if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
  fi
fi

for host in "${HOSTS[@]}"; do
  echo "------ $host ------"
  if [[ "$MODE" == "status" ]]; then
    if ! run_remote "$host" "$print_status_cmd"; then
      echo "Failed to query $host" >&2
    fi
  elif [[ "$MODE" == "clean" ]]; then
    if ! run_remote "$host" "$clean_cmd"; then
      echo "Failed to clean $host" >&2
    fi
  else
    if ! run_remote "$host" "$reset_cmd"; then
      echo "Failed to scan orphaned VRAM on $host" >&2
    fi
  fi
done
