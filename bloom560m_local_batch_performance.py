#!/usr/bin/env python3
"""
Local performance test script for BLOOM-560m model
Compares with Petals distributed inference results
"""

import asyncio
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configuration
MODEL_NAME = "bigscience/bloom-560m"

# Test configuration
BATCH_SIZES = [1, 2, 4, 8, 16]  # Configurable batch sizes to test

async def test_single_inference():
    """Test single inference performance (batch size 1)"""
    print("=== Single Inference Test (Batch Size 1) ===")
    
    # Load model and tokenizer
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    base_prompt = "The future of artificial intelligence is"
    
    print(f"Base prompt: '{base_prompt}'")
    
    # Tokenize single prompt
    inputs = tokenizer([base_prompt], return_tensors="pt", padding=True)
    
    print(f"Input shapes:")
    print(f"  input_ids: {inputs['input_ids'].shape}")
    print(f"  attention_mask: {inputs['attention_mask'].shape}")
    print(f"  Original prompt length: {len(base_prompt)} characters")
    print(f"  Tokenized length: {inputs['input_ids'].shape[1]} tokens")
    
    # Generate tokens
    start_time = time.time()
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,  # Add repetition penalty to prevent loops
    )
    end_time = time.time()
    
    # Decode and print output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nComplete text (input + generated):")
    print(f"'{generated_text}'")
    
    # Also show just the generated part for clarity
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    generated_part = generated_text[len(input_text):]
    print(f"\nGenerated part only:")
    print(f"'{generated_part}'")
    
    # Calculate metrics
    total_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    time_taken = end_time - start_time
    tokens_per_sec = total_tokens / time_taken
    
    print(f"\nPerformance:")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Time taken: {time_taken:.2f} seconds")
    print(f"  Tokens per second: {tokens_per_sec:.2f}")
    
    return model, tokenizer, tokens_per_sec

async def test_batch_inference(model, tokenizer, batch_size):
    """Test batch inference performance with same prompt repeated"""
    print(f"\n=== Batch Inference Test (Batch Size: {batch_size}) ===")
    
    base_prompt = "The future of artificial intelligence is"
    
    # Create batch of identical prompts (same prompt repeated batch_size times)
    prompts = [base_prompt] * batch_size  # This creates [prompt, prompt, prompt, ...] batch_size times
    
    print(f"Creating batch with {batch_size} identical prompts")
    print(f"Base prompt: '{base_prompt}'")
    
    # Tokenize the batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    
    print(f"Input shapes:")
    print(f"  input_ids: {inputs['input_ids'].shape}")
    print(f"  attention_mask: {inputs['attention_mask'].shape}")
    print(f"  Original prompt length: {len(base_prompt)} characters")
    print(f"  Tokenized length: {inputs['input_ids'].shape[1]} tokens")
    
    # Generate tokens
    start_time = time.time()
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,  # Add repetition penalty to prevent loops
    )
    end_time = time.time()
    
    # Decode and print outputs
    print(f"\nGenerated texts:")
    for i in range(min(3, batch_size)):  # Show first 3 outputs to avoid spam
        generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        print(f"  Output {i+1} (complete): '{generated_text}'")
        
        # Also show just the generated part
        print()
    
    if batch_size > 3:
        print(f"  ... and {batch_size - 3} more outputs")
    
    # Calculate metrics
    total_tokens = (outputs.shape[1] - inputs["input_ids"].shape[1]) * batch_size
    time_taken = end_time - start_time
    tokens_per_sec = total_tokens / time_taken
    tokens_per_sec_per_prompt = tokens_per_sec / batch_size
    
    print(f"\nPerformance:")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Time taken: {time_taken:.2f} seconds")
    print(f"  Total tokens per second: {tokens_per_sec:.2f}")
    print(f"  Tokens per second per prompt: {tokens_per_sec_per_prompt:.2f}")
    
    return {
        'total_tokens': total_tokens,
        'time': time_taken,
        'tokens_per_sec': tokens_per_sec,
        'tokens_per_sec_per_prompt': tokens_per_sec_per_prompt
    }

async def main():
    print("Starting BLOOM-560m Local Performance Test")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    # Single inference (batch size 1)
    try:
        model, tokenizer, single_tokens_sec = await test_single_inference()
    except Exception as e:
        print(f"Error during single inference: {e}")
        return

    # Batch inference
    batch_results = {}
    
    for batch_size in BATCH_SIZES[1:]:  # Skip batch size 1 since we already tested it
        try:
            result = await test_batch_inference(model, tokenizer, batch_size)
            batch_results[batch_size] = result
        except Exception as e:
            print(f"Error during batch inference (size {batch_size}): {e}")
            continue

    # Add single inference result to batch_results for consistency
    batch_results[1] = {
        'total_tokens': 50,  # Approximate
        'time': 50 / single_tokens_sec,
        'tokens_per_sec': single_tokens_sec,
        'tokens_per_sec_per_prompt': single_tokens_sec
    }

    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'Total Tokens/sec':<18} {'Tokens/sec/prompt':<20} {'Time (s)':<10}")
    print("-" * 60)
    
    for batch_size in BATCH_SIZES:
        if batch_size in batch_results:
            result = batch_results[batch_size]
            print(f"{batch_size:<12} {result['tokens_per_sec']:<18.2f} {result['tokens_per_sec_per_prompt']:<20.2f} {result['time']:<10.2f}")

    # Plot
    plt.figure(figsize=(8, 5))
    batch_sizes = list(batch_results.keys())
    tokens_sec = [batch_results[bs]['tokens_per_sec'] for bs in batch_sizes]
    tokens_sec_per_prompt = [batch_results[bs]['tokens_per_sec_per_prompt'] for bs in batch_sizes]
    
    plt.plot(batch_sizes, tokens_sec, 'o-', label='Total tokens/sec (throughput)', linewidth=2, markersize=8)
    plt.plot(batch_sizes, tokens_sec_per_prompt, 's--', label='Tokens/sec per prompt', linewidth=2, markersize=8)
    plt.xlabel('Batch size')
    plt.ylabel('Tokens/sec')
    plt.title('BLOOM-560m Local Inference Performance')
    plt.xticks(batch_sizes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bloom560m_local_batch_performance.png', dpi=300, bbox_inches='tight')
    print(f"\nPerformance graph saved as 'bloom560m_local_batch_performance.png'")

if __name__ == "__main__":
    asyncio.run(main()) 