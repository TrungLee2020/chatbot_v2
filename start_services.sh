#!/bin/bash

set -e

echo "Starting services..."

# Start Xinference server
xinference-local -H 0.0.0.0 --port 9997 &

# Wait for Xinference to be ready
until curl -s http://xinference:9997/health; do
  echo "Waiting for Xinference to be ready..."
  sleep 5
done

echo "Xinference is ready. Registering custom model..."

# Run the registration script
python3 /app/register_xinference_model.py

echo "Custom model registered. Attempting to launch the model..."

# Launch the model programmatically
python3 /app/launch_xinference_model.py

echo "Init chat history DB..."

# Launch the model programmatically
python3 /app/database.py

# Start your other services with hot-reloading
echo "Starting backend server with hot-reloading..."
uvicorn query_server:app --host 0.0.0.0 --port 8338 --reload &

echo "Starting API server with hot-reloading..."
uvicorn app:app --host 0.0.0.0 --port 6868 --reload --log-level debug &

echo "All services started. Waiting..."

# Keep the container running
wait

# Trap SIGTERM and SIGINT
trap 'kill $(jobs -p)' SIGTERM SIGINT

# Wait for all background processes to complete
wait