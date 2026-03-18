#!/bin/bash

MAX_RETRIES=20
RETRY_DELAY=120  # seconds (2 minutes)
ENV_NAME="snowmap_cdse"

# ---- INIT MICROMAMBA ----
# This is REQUIRED for activation to work in scripts
eval "$(micromamba shell hook --shell bash)"

# Activate environment
micromamba activate "$ENV_NAME"

attempt=1

while true; do
    echo "Running main.py (attempt $attempt)..."

    # Run script and capture output + exit code
    output=$(python main.py 2>&1)
    exit_code=$?

    # If successful → exit loop
    if [ $exit_code -eq 0 ]; then
        echo "✅ Script finished successfully!"
        break
    fi

    echo "❌ Script failed."

    # Check if error is related to STAC / connection
    if echo "$output" | grep -qiE "stac|connection|timeout|temporarily unavailable"; then
        if [ $attempt -ge $MAX_RETRIES ]; then
            echo "🚫 Max retries reached. Exiting."
            exit 1
        fi

        echo "🌐 Detected STAC/connection issue. Retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY
        attempt=$((attempt + 1))
    else
        echo "💥 Non-retryable error detected:"
        echo "$output"
        exit 1
    fi
done
