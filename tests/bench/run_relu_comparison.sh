#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-cora}"
server_ip="${2:-127.0.0.1}"
binary="${3:-./relu_comparison}"

case "${dataset}" in
  cora|citeseer|pubmed) ;;
  *) echo "dataset must be cora, citeseer, or pubmed" >&2; exit 2 ;;
esac

log_dir="relu_comparison_logs_${dataset}"
mkdir -p "${log_dir}"

# Party 1 produces the dealer material and exits. Parties 2 and 3 then execute
# the online phase concurrently. The client log contains the final table.
"${binary}" 1 "${server_ip}" "${dataset}" >"${log_dir}/dealer.log" 2>&1
"${binary}" 2 "${server_ip}" "${dataset}" >"${log_dir}/server.log" 2>&1 &
server_pid=$!
"${binary}" 3 "${server_ip}" "${dataset}" >"${log_dir}/client.log" 2>&1
wait "${server_pid}"

cat "${log_dir}/client.log"
