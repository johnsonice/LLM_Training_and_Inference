"""
Asynchronous API Performance Testing Script

This script benchmarks the performance of LLM API calls (specifically Netmind API) 
under concurrent load conditions. It measures:
- Response time for each API call
- Token processing rates
- Overall throughput
- Success/failure rates

The test simulates multiple concurrent requests to evaluate how the API 
performs under various levels of load, which is useful for:
- Capacity planning
- Performance optimization
- Identifying bottlenecks in high-throughput scenarios
- Comparing different models or API configurations
"""

#%%
import asyncio
import time
import os
from openai import AsyncOpenAI
import sys
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path='../../../.env')
#%%
NETMIND_API_BASE_URL = "https://api.netmind.ai/inference-api/openai/v1"
NETMIND_API_KEY = os.environ.get("NETMIND_API_KEY")
NUM_CONCURRENT_CALLS = 40
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
TEST_PROMPT = "Write a short story about a robot learning to dream."*1000
MAX_TOKENS_PER_CALL = 50000

#%%
async def call_netmind_api(client: AsyncOpenAI, call_id: int):
    """
    Makes a single asynchronous call to the Netmind API and measures performance.

    Args:
        client: The AsyncOpenAI client instance.
        call_id: An identifier for the concurrent call.

    Returns:
        A tuple containing (call_id, duration, prompt_tokens, completion_tokens) for successful calls,
        or (call_id, duration, 0, None) if an error occurred or no tokens were generated.
    """
    start_time = time.perf_counter()
    duration = 0.0 # Initialize duration
    prompt_tokens = 0 # Initialize prompt tokens
    completion_tokens = None # Initialize completion tokens to None (indicating potential failure)
    try:
        print(f"Starting call {call_id}...")
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_TOKENS_PER_CALL,
            stream=False # Ensure we get the full response for token counting
        )
        end_time = time.perf_counter()
        duration = end_time - start_time

        # Extract token counts safely
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens if response.usage.prompt_tokens is not None else 0
            completion_tokens = response.usage.completion_tokens if response.usage.completion_tokens is not None else 0
        else:
             prompt_tokens = 0
             completion_tokens = 0 # Treat as 0 if usage info is missing

        if completion_tokens is not None and completion_tokens > 0:
            tokens_per_second = completion_tokens / duration if duration > 0 else 0
            print(f"Call {call_id} finished in {duration:.2f}s, Input Tokens: {prompt_tokens}, Output Tokens: {completion_tokens}, Speed: {tokens_per_second:.2f} tokens/sec")
            return call_id, duration, prompt_tokens, completion_tokens
        elif completion_tokens == 0:
             print(f"Call {call_id} finished in {duration:.2f}s, Input Tokens: {prompt_tokens}, but no Output Tokens generated.")
             return call_id, duration, prompt_tokens, 0 # Return 0 completion tokens
        else: # Should not happen with current logic, but good practice
             print(f"Call {call_id} finished in {duration:.2f}s, Input Tokens: {prompt_tokens}. Unexpected state for completion_tokens.")
             return call_id, duration, prompt_tokens, 0 # Return 0 completion tokens

    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Call {call_id} failed after {duration:.2f}s: {e}")
        # Return None for completion_tokens to signal failure, but duration and 0 prompt tokens
        return call_id, duration, 0, None

async def benchmark():
    """
    Runs the benchmark by sending multiple concurrent API calls.
    """
    if not NETMIND_API_KEY:
        print("Error: NETMIND_API_KEY environment variable not set.")
        print("Please set the NETMIND_API_KEY in your .env file or environment.")
        return

    print(f"Starting benchmark with {NUM_CONCURRENT_CALLS} concurrent calls to {MODEL_NAME}...")
    print(f"Base URL: {NETMIND_API_BASE_URL}")
    print(f"Prompt length: {len(TEST_PROMPT)} characters")
    print(f"Max tokens per call: {MAX_TOKENS_PER_CALL}\\n")

    # Initialize the AsyncOpenAI client
    client = AsyncOpenAI(
        base_url=NETMIND_API_BASE_URL,
        api_key=NETMIND_API_KEY,
    )

    # Create tasks
    tasks = [call_netmind_api(client, i) for i in range(NUM_CONCURRENT_CALLS)]

    # Measure time for all concurrent calls
    overall_start_time = time.perf_counter()
    results = await asyncio.gather(*tasks)
    overall_end_time = time.perf_counter()
    total_duration = overall_end_time - overall_start_time

    # Process results
    successful_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0 # Renamed from total_generated_tokens for clarity
    total_individual_tokens_per_second = 0.0 # Sum of individual call speeds for averaging

    print("\n--- Benchmark Results ---")
    # Unpack results including prompt_tokens
    for call_id, duration, prompt_tokens, completion_tokens in results:
        # Check if the call was successful (completion_tokens is not None)
        if completion_tokens is not None:
            successful_calls += 1
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            # Calculate individual speed only if call generated tokens and took time
            if duration > 0 and completion_tokens > 0:
                individual_speed = completion_tokens / duration
                total_individual_tokens_per_second += individual_speed
            # Individual call details are printed within call_netmind_api
        else:
            # Failed call details are printed within call_netmind_api
            pass # No need to print failure again here

    print("\n--- Summary ---")
    print(f"Total time for {NUM_CONCURRENT_CALLS} concurrent calls: {total_duration:.2f} seconds")
    print(f"Successful calls: {successful_calls}/{NUM_CONCURRENT_CALLS}")
    # Use total_completion_tokens in the summary print statement
    print(f"Total completion tokens generated (successful calls): {total_completion_tokens}")

    if successful_calls > 0:
        # Calculate average tokens
        average_prompt_tokens = total_prompt_tokens / successful_calls
        average_completion_tokens = total_completion_tokens / successful_calls
        print(f"Average prompt tokens per successful call: {average_prompt_tokens:.2f}")
        print(f"Average completion tokens per successful call: {average_completion_tokens:.2f}")

        # Average speed based on individual call durations and tokens
        average_tokens_per_second = total_individual_tokens_per_second / successful_calls
        print(f"Average tokens per second (based on individual successful calls): {average_tokens_per_second:.2f}")

        # Overall speed based on total tokens and total time for the batch
        if total_duration > 0:
            # Use total_completion_tokens for overall speed calculation
            overall_tokens_per_second = total_completion_tokens / total_duration
            print(f"Overall tokens per second (total completion tokens / total time): {overall_tokens_per_second:.2f}")
        else:
            print("Overall tokens per second: N/A (Total duration was zero)")
    else:
        # Print N/A for all averages if no calls were successful
        print("Average prompt tokens per successful call: N/A")
        print("Average completion tokens per successful call: N/A")
        print("Average tokens per second: N/A (No successful calls)")
        print("Overall tokens per second: N/A (No successful calls)")

#%%
if __name__ == "__main__":
    # Ensure asyncio event loop runs correctly depending on the environment
    # (e.g., Jupyter notebooks might have a running loop)
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If in an environment like Jupyter, create a task
            loop.create_task(benchmark())
        else:
            # Otherwise, run normally
            asyncio.run(benchmark())
    except RuntimeError:
        # No running event loop, run normally
        asyncio.run(benchmark())
