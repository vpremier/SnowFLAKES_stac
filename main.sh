#!/usr/bin/env bash
set -euo pipefail

# Main workflow entry point. Each processing stage can be added as a function
# below while keeping all user settings in the same JSON configuration.
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 path/to/config.json" >&2
    exit 1
fi

CONFIG_PATH="$1"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file does not exist: ${CONFIG_PATH}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

run_query() {
    echo "[1/1] Querying available satellite scenes"
    python3 "${SCRIPT_DIR}/data_download/query_available.py" \
        --config "${CONFIG_PATH}"
}

run_query
