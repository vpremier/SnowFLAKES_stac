#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${PROJECT_ROOT}/main.sh"
CONFIG_ROOT="${PROJECT_ROOT}/config/Sentinel2-l2a"
dry_run=false

if (( $# == 1 )) && [[ "$1" == "--dry-run" ]]; then
    dry_run=true
elif (( $# != 0 )); then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

cd "${PROJECT_ROOT}" || exit 1

if [[ ! -f "${MAIN_SCRIPT}" ]]; then
    echo "SnowFLAKES launcher not found: ${MAIN_SCRIPT}" >&2
    exit 1
fi

if [[ ! -d "${CONFIG_ROOT}" ]]; then
    echo "Configuration directory not found: ${CONFIG_ROOT}" >&2
    exit 1
fi
mapfile -d '' CONFIGS < <(
    find "${CONFIG_ROOT}/snow" "${CONFIG_ROOT}/snowfree" -type f -name 'config_*.json' -print0 | sort -zV
)
if (( ${#CONFIGS[@]} == 0 )); then
    echo "No snow or snow-free configuration files found in ${CONFIG_ROOT}" >&2
    exit 1
fi

completed=0
failed=0
failed_configs=()
total=${#CONFIGS[@]}
current=0

for config_path in "${CONFIGS[@]}"; do
    current=$((current + 1))
    relative_path="${config_path#"${PROJECT_ROOT}/"}"
    echo
    echo "[${current}/${total}] ${relative_path}"
    if [[ "${dry_run}" == true ]]; then
        continue
    fi

    if bash "${MAIN_SCRIPT}" "${config_path}"; then
        completed=$((completed + 1))
    else
        status=$?
        failed=$((failed + 1))
        failed_configs+=("${relative_path} (exit ${status})")
        echo "SnowFLAKES run failed: ${relative_path} (exit ${status})" >&2
    fi
done

echo
if [[ "${dry_run}" == true ]]; then
    echo "Dry run: ${total} configs listed; no processing started."
else
    echo "Snow and snow-free batch finished: ${completed} succeeded, ${failed} failed."
fi

if (( failed > 0 )); then
    echo "Failed configurations:" >&2
    printf '  %s\n' "${failed_configs[@]}" >&2
    exit 1
fi
