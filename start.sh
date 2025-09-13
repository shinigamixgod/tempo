#!/bin/bash
# start.sh

# Function to run cache every hour
run_cache_loop() {
  while true; do
    echo "$(date) - Generating cache..."
    python /app/generate_cache.py
    echo "$(date) - Cache generated. Waiting 1 hour..."
    sleep 3600
  done
}

# Start the cache loop in the background
run_cache_loop &

# Start FastAPI
exec uvicorn main:app --host 0.0.0.0 --port 3000