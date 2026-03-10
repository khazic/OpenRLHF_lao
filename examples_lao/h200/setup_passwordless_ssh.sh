#!/usr/bin/env bash
set -euo pipefail

# One-shot script to enable passwordless SSH for all nodes in hostfile_nodes.txt.
# Usage: bash setup_passwordless_ssh.sh [ssh_password]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${SCRIPT_DIR}/hostfile_nodes.txt"

if [[ ! -f "$HOSTFILE" ]]; then
  echo "Hostfile not found: $HOSTFILE" >&2
  exit 1
fi

PASSWORD="${1:-${SSH_PASSWORD:-1}}"
if [[ -z "$PASSWORD" ]]; then
  echo "SSH password not provided. Pass it as the first argument or export SSH_PASSWORD." >&2
  exit 1
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "sshpass is required but not found. Please install it first." >&2
  exit 1
fi

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

KEY_PATH="$HOME/.ssh/id_ed25519"
PUB_KEY="${KEY_PATH}.pub"

if [[ ! -f "$PUB_KEY" ]]; then
  echo "Generating new ed25519 key pair at $KEY_PATH ..."
  ssh-keygen -t ed25519 -N "" -f "$KEY_PATH"
else
  echo "Using existing public key: $PUB_KEY"
fi

mapfile -t HOSTS < <(awk '{print $1}' "$HOSTFILE" | sed 's/#.*//' | sed '/^$/d')

if [[ "${#HOSTS[@]}" -eq 0 ]]; then
  echo "No hosts found in hostfile: $HOSTFILE" >&2
  exit 1
fi

for host in "${HOSTS[@]}"; do
  echo "Installing key on $host ..."
  sshpass -p "$PASSWORD" ssh-copy-id -i "$PUB_KEY" -o StrictHostKeyChecking=no "$host"
done

echo "Verifying passwordless SSH:"
for host in "${HOSTS[@]}"; do
  echo -n "  $host -> "
  if ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" hostname; then
    echo "OK"
  else
    echo "FAILED" >&2
  fi
done

echo "All done. You can now run .sh without sshpass."
