#!/bin/bash

# Configuration variables
PORT=8800
BATCH_SIZE=1024
NUM_PROCESSES=4

# Calculate total number of files
total_files=$(ls /ephemeral/home/xiong/data/Fund/Climate/infer_res_2label/*.csv | grep -v "results_" | grep -v "^\." | wc -l)
files_per_process=$((total_files / NUM_PROCESSES))

echo "Total files found: $total_files"
echo "Files per process: $files_per_process"
echo "Running with arguments:"
echo "  --port: $PORT"
echo "  --batch-size: $BATCH_SIZE"
echo "Starting $NUM_PROCESSES processes..."

# Start processes with different ranges
for i in $(seq 0 $((NUM_PROCESSES-1))); do
    start_idx=$((i * files_per_process))
    # For the last process, make sure to include any remaining files
    if [ $i -eq $((NUM_PROCESSES-1)) ]; then
        end_idx=$total_files
    else
        end_idx=$(((i + 1) * files_per_process))
    fi
    
    echo "Starting process $((i+1))/$NUM_PROCESSES (files $start_idx to $end_idx)"
    python async_inference.py \
        --port $PORT \
        --batch-size $BATCH_SIZE \
        --start-idx $start_idx \
        --end-idx $end_idx &
    
    # Wait for 1 minute before starting the next process
    if [ $i -lt $((NUM_PROCESSES-1)) ]; then
        echo "Waiting 60 seconds before starting next process..."
        sleep 60
    fi
done

# Wait for all processes to complete
echo "Waiting for all processes to complete..."
wait

echo "All processes completed!" 