#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir="${repo_dir}/build"
binary="${build_dir}/BPGNN"
server_pid=""

cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ ! -x "${binary}" ]]; then
    echo "BPGNN is not built. Configure and build the project first." >&2
    exit 1
fi

cd "${build_dir}"

local_ip="127.0.0.1"
if [[ $# -gt 0 && "$1" != --* ]]; then
    local_ip=$1
    shift
fi
extra_args=("$@")

echo "[1/3] Generating offline key material..."
"${binary}" 1 "${extra_args[@]}"

echo "[2/3] Starting server 0..."
"${binary}" 2 "${extra_args[@]}" >server-0.log 2>&1 &
server_pid=$!

echo "[3/3] Running server 1/client..."
"${binary}" 3 "${local_ip}" "${extra_args[@]}"

wait "${server_pid}"
server_pid=""
echo "Local BPGNN example completed. Server output: ${build_dir}/server-0.log"
