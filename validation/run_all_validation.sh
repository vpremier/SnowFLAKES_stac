#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_ROOT="${PROJECT_ROOT}/config/validation"
MAIN_SCRIPT="${PROJECT_ROOT}/main.sh"

if [[ ! -f "${MAIN_SCRIPT}" ]]; then
    echo "SnowFLAKES launcher not found: ${MAIN_SCRIPT}" >&2
    exit 1
fi

if [[ ! -d "${CONFIG_ROOT}" ]]; then
    echo "Validation configuration directory not found: ${CONFIG_ROOT}" >&2
    exit 1
fi

mapfile -d '' CONFIGS < <(
    find "${CONFIG_ROOT}" -type f -name 'config_*.json' -print0 | sort -zV
)

if (( ${#CONFIGS[@]} == 0 )); then
    echo "No validation configuration files found in ${CONFIG_ROOT}" >&2
    exit 1
fi

completed=0
failed=0
failed_configs=()
total=${#CONFIGS[@]}

for config_path in "${CONFIGS[@]}"; do
    current=$((completed + failed + 1))
    relative_path="${config_path#"${PROJECT_ROOT}/"}"
    echo
    echo "[${current}/${total}] Running ${relative_path}"

    if bash "${MAIN_SCRIPT}" "${config_path}"; then
        completed=$((completed + 1))
    else
        status=$?
        failed=$((failed + 1))
        failed_configs+=("${relative_path} (exit ${status})")
        echo "Validation run failed: ${relative_path} (exit ${status})" >&2
    fi
done

echo
echo "Validation batch finished: ${completed} succeeded, ${failed} failed."

if (( failed > 0 )); then
    echo "Failed configurations:" >&2
    printf '  %s\n' "${failed_configs[@]}" >&2
    exit 1
fi
