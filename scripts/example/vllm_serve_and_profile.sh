#!/bin/bash

# Configuration
MODEL_NAME="meta-llama/Llama-2-7b-hf"  # Change this to your desired model
GPU_MEMORY_UTILIZATION=0.9
MAX_NUM_SEQS=256
PORT=8000

# Test configurations
INPUT_LENGTHS=(64 128 256 512 1024)
OUTPUT_LENGTHS=(64 128 256 512)
SEQUENCE_LENGTHS=(2048 4096 8192)

# Create results directory
RESULTS_DIR="vllm_profile_results"
mkdir -p $RESULTS_DIR

# Function to generate test prompts
generate_prompt() {
    local length=$1
    # Generate a prompt of specified length using repeated words
    python3 -c "
import random
words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
prompt = ' '.join(random.choices(words, k=$length))
print(prompt)
"
}

# Function to run profiling test
run_profile_test() {
    local input_len=$1
    local output_len=$2
    local seq_len=$3
    
    echo "Running test with input_len=$input_len, output_len=$output_len, seq_len=$seq_len"
    
    # Generate test prompt
    local prompt=$(generate_prompt $input_len)
    
    # Run the test using curl and measure time
    local start_time=$(date +%s.%N)
    
    # Make request to vLLM server
    response=$(curl -s -X POST "http://localhost:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_NAME\",
            \"prompt\": \"$prompt\",
            \"max_tokens\": $output_len,
            \"temperature\": 0.0,
            \"stream\": true
        }")
    
    # Process streaming response and measure latencies
    local first_token_time=0
    local last_token_time=0
    local token_count=0
    
    while IFS= read -r line; do
        if [[ $line == data:* ]]; then
            local current_time=$(date +%s.%N)
            if [[ $first_token_time == 0 ]]; then
                first_token_time=$current_time
            fi
            last_token_time=$current_time
            ((token_count++))
        fi
    done <<< "$response"
    
    local end_time=$(date +%s.%N)
    
    # Calculate metrics
    local ttft=$(echo "$first_token_time - $start_time" | bc)
    local total_time=$(echo "$end_time - $start_time" | bc)
    local avg_token_latency=$(echo "($last_token_time - $first_token_time) / ($token_count - 1)" | bc)
    
    # Save results
    echo "input_len,output_len,seq_len,ttft,avg_token_latency,total_time,token_count" >> "$RESULTS_DIR/profile_results.csv"
    echo "$input_len,$output_len,$seq_len,$ttft,$avg_token_latency,$total_time,$token_count" >> "$RESULTS_DIR/profile_results.csv"
}

# Start vLLM server
echo "Starting vLLM server..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    --port $PORT &

# Wait for server to start
sleep 10

# Create results file with header
echo "input_len,output_len,seq_len,ttft,avg_token_latency,total_time,token_count" > "$RESULTS_DIR/profile_results.csv"

# Run tests
for input_len in "${INPUT_LENGTHS[@]}"; do
    for output_len in "${OUTPUT_LENGTHS[@]}"; do
        for seq_len in "${SEQUENCE_LENGTHS[@]}"; do
            run_profile_test $input_len $output_len $seq_len
            sleep 2  # Small delay between tests
        done
    done
done

# Kill the vLLM server
pkill -f "vllm.entrypoints.openai.api_server"

echo "Profiling complete. Results saved in $RESULTS_DIR/profile_results.csv"
