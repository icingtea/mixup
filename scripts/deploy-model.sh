#!/usr/bin/env bash
# deploy-model.sh — copy the mixup model weights to a device (the Pi "lambda", etc.).
#
# Usage:
#   ./scripts/deploy-model.sh [node] [remote_root]
#
#   node         one of: lambda | pi-inch-nails  (default: lambda)
#   remote_root  model directory on the device, relative to $HOME
#                (default: mixup/models)
#
# Reads SSH credentials from .ssh-credentials (gitignored, never committed). SSHes in,
# creates the directory, and rsyncs the weights to <remote_root>/granite-3.1-1b-a400m-instruct.
# That directory is what the container mounts as /usr/local/app/models (read-only).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CREDS="$REPO_ROOT/.ssh-credentials"
[ -f "$CREDS" ] || { echo "error: missing $CREDS (gitignored credentials file)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CREDS"

NODE="${1:-lambda}"
REMOTE_ROOT="${2:-mixup/models}"

case "$NODE" in
  lambda)
    HOST="$lambda_host"; SSH_USER="$lambda_user"; SSH_PASS="$lambda_password" ;;
  pi-inch-nails|pi)
    HOST="$pi_inch_nails_host"; SSH_USER="$pi_inch_nails_user"; SSH_PASS="$pi_inch_nails_password" ;;
  *)
    echo "error: unknown node '$NODE' (known: lambda, pi-inch-nails)" >&2
    exit 1
    ;;
esac

MODEL_NAME="granite-3.1-1b-a400m-instruct"
LOCAL_MODEL="models/$MODEL_NAME"
REMOTE_MODEL="$REMOTE_ROOT/$MODEL_NAME"

[ -d "$LOCAL_MODEL" ] || { echo "error: local model dir not found: $LOCAL_MODEL" >&2; exit 1; }

# Locate sshpass for password auth (prefer PATH, else any nix store copy).
SSHPASS_BIN="$(command -v sshpass || true)"
if [ -z "$SSHPASS_BIN" ]; then
  SSHPASS_BIN="$(ls -1 /nix/store/*sshpass*/bin/sshpass 2>/dev/null | head -1 || true)"
fi

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

# run_ssh <cmd...> — run a command on the device (password or key auth).
run_ssh() {
  if [ -n "$SSHPASS_BIN" ]; then
    SSHPASS="$SSH_PASS" "$SSHPASS_BIN" -e ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST" "$@"
  else
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST" "$@"
  fi
}

# 1. Create the remote directory (the container will mount this).
echo "==> [$NODE] creating remote dir '$REMOTE_ROOT' on $SSH_USER@$HOST"
run_ssh "mkdir -p '$REMOTE_ROOT'"

# 2. rsync the model (skip the transformers .cache — the device rebuilds it).
echo "==> [$NODE] rsyncing model to $SSH_USER@$HOST:$REMOTE_MODEL/ (2.5 GB, may take a while)"
RSH="ssh ${SSH_OPTS[*]}"
if [ -n "$SSHPASS_BIN" ]; then
  RSH="$SSHPASS_BIN -e ssh ${SSH_OPTS[*]}"
fi
SSHPASS="$SSH_PASS" rsync -avz --progress --exclude='.cache' \
  -e "$RSH" \
  "$LOCAL_MODEL/" "$SSH_USER@$HOST:$REMOTE_MODEL/"

echo
echo "==> done: model is at $SSH_USER@$HOST:$REMOTE_MODEL"
echo "    mount it into a container with:"
echo "      -v /home/$SSH_USER/$REMOTE_ROOT:/usr/local/app/models:ro"
