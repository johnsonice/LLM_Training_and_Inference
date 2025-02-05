#!/bin/bash

# Configuration variables
PORT=8100

# Print out arguments
echo "Running with arguments:"
echo "  --port: $PORT"

# Run the Python script with arguments passed directly
python create_batch_task.py --port $PORT # --test-run

