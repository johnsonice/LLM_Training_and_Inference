#!/bin/bash

# Configuration variables
PORT=8800
BATCH_SIZE=1024

# Print out arguments
echo "Running with arguments:"
echo "  --port: $PORT"
echo "  --batch-size: $BATCH_SIZE"

# Run the Python script with arguments passed directly
python async_inference.py --port $PORT --batch-size $BATCH_SIZE #--test-run

