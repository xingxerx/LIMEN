#!/usr/bin/env bash
# Copyright (C) 2026 Jemone McCubbin / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
#
# Stand up a LIMEN coordination node (e.g. LIMEN-B) on a remote machine
# with one command: ships the local checkout over rsync/ssh, creates a
# venv, installs the distributed extra, optionally mints a self-signed TLS
# cert, and starts `python -m limen.distributed.server` under systemd.
#
# Usage:
#   scripts/deploy_node.sh <user@host> --node-id limen-b [options]
#
# Required:
#   <user@host>            SSH destination, e.g. ubuntu@node-b.example.com
#   --node-id ID           Unique node identifier (LIMEN_NODE_ID)
#
# Options:
#   --port PORT            gRPC port to bind (default: 50051)
#   --peers HOST:PORT,...  Comma-separated peer addresses to register with
#                          on startup (LIMEN_KNOWN_PEERS)
#   --extras EXTRAS        Comma-separated pip extras beyond 'distributed',
#                          e.g. "dwave,braket" (default: none)
#   --tls                  Generate a self-signed TLS cert/key on the
#                          remote host and bind the server with TLS
#   --remote-dir DIR       Remote checkout directory
#                          (default: ~/limen-<node-id>)
#   --ssh-key PATH         Identity file passed to ssh/rsync as -i PATH
#   --no-restart           Skip stopping/restarting if the service is
#                          already running (still installs/updates code)
#
# Example — bring up LIMEN-B and point it at LIMEN-A:
#   scripts/deploy_node.sh ubuntu@node-b \
#       --node-id limen-b --port 50051 \
#       --peers limen-a.internal:50051 --tls
#
# Then on LIMEN-A's side, pass server_addresses=["node-b:50051"] to
# limen.run_pipeline() — see examples/distributed_two_node.py.

set -euo pipefail

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

DEST="$1"
shift

NODE_ID=""
PORT=50051
PEERS=""
EXTRAS=""
USE_TLS=0
REMOTE_DIR=""
SSH_KEY_OPT=()
RESTART=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-id) NODE_ID="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --peers) PEERS="$2"; shift 2 ;;
        --extras) EXTRAS="$2"; shift 2 ;;
        --tls) USE_TLS=1; shift ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --ssh-key) SSH_KEY_OPT=(-i "$2"); shift 2 ;;
        --no-restart) RESTART=0; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ -z "$NODE_ID" ]]; then
    echo "ERROR: --node-id is required" >&2
    usage
fi

REMOTE_DIR="${REMOTE_DIR:-~/limen-${NODE_ID}}"
EXTRA_SPEC="distributed"
if [[ -n "$EXTRAS" ]]; then
    EXTRA_SPEC="distributed,${EXTRAS}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="limen-node-${NODE_ID}"

echo "[1/4] Syncing ${REPO_ROOT} -> ${DEST}:${REMOTE_DIR} ..."
ssh "${SSH_KEY_OPT[@]}" "$DEST" "mkdir -p ${REMOTE_DIR}"
rsync -az --delete \
    --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    --exclude='target' --exclude='results' \
    -e "ssh ${SSH_KEY_OPT[*]}" \
    "${REPO_ROOT}/" "${DEST}:${REMOTE_DIR}/"

echo "[2/4] Installing LIMEN (extras: ${EXTRA_SPEC}) in a remote venv ..."
ssh "${SSH_KEY_OPT[@]}" "$DEST" bash -s -- "$REMOTE_DIR" "$EXTRA_SPEC" <<'REMOTE_INSTALL'
set -euo pipefail
remote_dir="$1"
extra_spec="$2"
cd "$remote_dir"
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e ".[${extra_spec}]"
REMOTE_INSTALL

TLS_ENV=""
if [[ "$USE_TLS" -eq 1 ]]; then
    echo "[3/4] Generating self-signed TLS cert on the remote host ..."
    HOST_FOR_CERT="${DEST#*@}"
    ssh "${SSH_KEY_OPT[@]}" "$DEST" bash -s -- "$REMOTE_DIR" "$HOST_FOR_CERT" <<'REMOTE_TLS'
set -euo pipefail
remote_dir="$1"
cn="$2"
cert_dir="${remote_dir}/.tls"
mkdir -p "$cert_dir"
if [[ ! -f "${cert_dir}/server.crt" ]]; then
    openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
        -keyout "${cert_dir}/server.key" -out "${cert_dir}/server.crt" \
        -subj "/CN=${cn}" >/dev/null 2>&1
fi
REMOTE_TLS
    TLS_ENV="Environment=\"LIMEN_TLS_CERT=${REMOTE_DIR}/.tls/server.crt\"
Environment=\"LIMEN_TLS_KEY=${REMOTE_DIR}/.tls/server.key\""
else
    echo "[3/4] Skipping TLS (pass --tls to enable)."
fi

echo "[4/4] Writing and starting systemd unit '${SERVICE_NAME}' ..."
ssh "${SSH_KEY_OPT[@]}" "$DEST" sudo bash -s -- \
    "$SERVICE_NAME" "$REMOTE_DIR" "$NODE_ID" "$PORT" "$PEERS" "$RESTART" "$TLS_ENV" <<'REMOTE_UNIT'
set -euo pipefail
service_name="$1"; remote_dir="$2"; node_id="$3"; port="$4"; peers="$5"; restart="$6"; tls_env="$7"
remote_dir_expanded="$(eval echo "$remote_dir")"

cat > "/etc/systemd/system/${service_name}.service" <<UNIT
[Unit]
Description=LIMEN coordination node (${node_id})
After=network.target

[Service]
WorkingDirectory=${remote_dir_expanded}
ExecStart=${remote_dir_expanded}/.venv/bin/python -m limen.distributed.server
Environment="LIMEN_NODE_ID=${node_id}"
Environment="LIMEN_NODE_HOST=0.0.0.0"
Environment="LIMEN_NODE_PORT=${port}"
Environment="LIMEN_KNOWN_PEERS=${peers}"
${tls_env}
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable "${service_name}"
if [[ "$restart" -eq 1 ]]; then
    systemctl restart "${service_name}"
else
    systemctl start "${service_name}" || true
fi
sleep 1
systemctl --no-pager status "${service_name}"
REMOTE_UNIT

echo
echo "LIMEN node '${NODE_ID}' is running on ${DEST} as systemd unit '${SERVICE_NAME}', port ${PORT}."
echo "Point the other node's run_pipeline(server_addresses=[...]) at this host:${PORT}."
