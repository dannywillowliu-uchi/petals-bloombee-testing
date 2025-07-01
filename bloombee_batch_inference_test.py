#!/usr/bin/env python3
"""
BloomBee Batch Inference Test Script
Tests batch inference performance with huggyllama/llama-7b model
"""

import time
import torch
from transformers import AutoTokenizer
from bloombee import AutoDistributedModelForCausalLM
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "huggyllama/llama-7b"
DHT_PREFIX = "huggyllama/llama-7b-bloombee"
INITIAL_PEERS = [
    "/ip4/127.0.0.1/tcp/42879/p2p/12D3KooWKcTAZ6je9aDWir26WG4TxBxS6PFswU4sQPAeFWMuTb6k",
    "/ip4/172.28.0.12/tcp/42879/p2p/12D3KooWKcTAZ6je9aDWir26WG4TxBxS6PFswU4sQPAeFWMuTb6k"
]

# Test prompts for batch inference
TEST_PROMPTS = [
    "The future of artificial intelligence is",
    "Machine learning algorithms can",
    "Deep learning models are capable of",
    "Natural language processing enables",
    "Computer vision systems can",
    "Reinforcement learning allows agents to",
    "Neural networks process information by",
    "The transformer architecture revolutionized",
    "Attention mechanisms help models focus on",
    "Transfer learning enables models to"
]

def create_batch_prompts(num_batches=5, batch_size=4):
    """Create multiple batches of prompts for testing"""
    all_batches = []
    for i in range(num_batches):
        batch = TEST_PROMPTS[i * batch_size:(i + 1) * batch_size]
        if len(batch) == batch_size:  # Only add complete batches
            all_batches.append(batch)
    return all_batches

async def test_bloombee_batch_inference():
    """Test batch inference with BloomBee"""
    logger.info("Starting BloomBee batch inference test...")
    
    try:
        # Initialize tokenizer
        logger.info(f"Loading tokenizer for {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize BloomBee model
        logger.info("Initializing BloomBee model...")
        model = AutoDistributedModelForCausalLM.from_pretrained(
            MODEL_NAME,
            initial_peers=INITIAL_PEERS,
            dht_prefix=DHT_PREFIX,
            torch_dtype=torch.float32,
            max_retries=3,
            timeout=30.0
        )
        
        logger.info("Model loaded successfully!")
        
        # Create test batches
        test_batches = create_batch_prompts(num_batches=3, batch_size=3)
        logger.info(f"Created {len(test_batches)} test batches")
        
        # Test parameters
        max_new_tokens = 50
        temperature = 0.7
        do_sample = True
        
        total_tokens_generated = 0
        total_time = 0
        
        # Run batch inference tests
        for batch_idx, batch_prompts in enumerate(test_batches):
            logger.info(f"\n--- Batch {batch_idx + 1}/{len(test_batches)} ---")
            logger.info(f"Prompts: {batch_prompts}")
            
            # Tokenize batch
            start_time = time.time()
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            
            # Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            total_time += generation_time
            
            # Decode and analyze results
            responses = []
            for i, output in enumerate(outputs):
                # Remove input tokens to get only generated text
                generated_tokens = output[inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(response)
                
                # Count tokens
                num_tokens = len(generated_tokens)
                total_tokens_generated += num_tokens
                
                logger.info(f"Response {i+1}: {response[:100]}... ({num_tokens} tokens)")
            
            # Calculate batch metrics
            tokens_per_second = total_tokens_generated / total_time if total_time > 0 else 0
            logger.info(f"Batch {batch_idx + 1} completed in {generation_time:.2f}s")
            logger.info(f"Current throughput: {tokens_per_second:.2f} tokens/sec")
        
        # Final statistics
        logger.info("\n" + "="*50)
        logger.info("BATCH INFERENCE TEST RESULTS")
        logger.info("="*50)
        logger.info(f"Total batches processed: {len(test_batches)}")
        logger.info(f"Total prompts processed: {len(test_batches) * len(test_batches[0])}")
        logger.info(f"Total tokens generated: {total_tokens_generated}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average throughput: {total_tokens_generated / total_time:.2f} tokens/sec")
        logger.info(f"Average time per batch: {total_time / len(test_batches):.2f} seconds")
        logger.info(f"Average tokens per batch: {total_tokens_generated / len(test_batches):.1f} tokens")
        
        return {
            'total_batches': len(test_batches),
            'total_prompts': len(test_batches) * len(test_batches[0]),
            'total_tokens': total_tokens_generated,
            'total_time': total_time,
            'throughput': total_tokens_generated / total_time,
            'avg_batch_time': total_time / len(test_batches),
            'avg_tokens_per_batch': total_tokens_generated / len(test_batches)
        }
        
    except Exception as e:
        logger.error(f"Error during batch inference test: {e}")
        raise

async def test_single_inference_comparison():
    """Compare single vs batch inference performance"""
    logger.info("\n" + "="*50)
    logger.info("SINGLE vs BATCH INFERENCE COMPARISON")
    logger.info("="*50)
    
    try:
        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoDistributedModelForCausalLM.from_pretrained(
            MODEL_NAME,
            initial_peers=INITIAL_PEERS,
            dht_prefix=DHT_PREFIX,
            torch_dtype=torch.float32,
            max_retries=3,
            timeout=30.0
        )
        
        test_prompts = TEST_PROMPTS[:4]  # Use first 4 prompts
        
        # Test single inference
        logger.info("Testing single inference...")
        single_start_time = time.time()
        single_tokens = 0
        
        for prompt in test_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            single_tokens += len(generated_tokens)
        
        single_time = time.time() - single_start_time
        
        # Test batch inference
        logger.info("Testing batch inference...")
        batch_start_time = time.time()
        
        inputs = tokenizer(test_prompts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        batch_time = time.time() - batch_start_time
        batch_tokens = sum(len(output[inputs['input_ids'].shape[1]:]) for output in outputs)
        
        # Compare results
        logger.info(f"Single inference: {single_time:.2f}s for {single_tokens} tokens")
        logger.info(f"Batch inference: {batch_time:.2f}s for {batch_tokens} tokens")
        logger.info(f"Speedup: {single_time / batch_time:.2f}x")
        logger.info(f"Single throughput: {single_tokens / single_time:.2f} tokens/sec")
        logger.info(f"Batch throughput: {batch_tokens / batch_time:.2f} tokens/sec")
        
    except Exception as e:
        logger.error(f"Error during comparison test: {e}")
        raise

def main():
    """Main function to run all tests"""
    logger.info("BloomBee Batch Inference Test Suite")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"DHT Prefix: {DHT_PREFIX}")
    logger.info(f"Initial Peers: {INITIAL_PEERS}")
    
    try:
        # Run batch inference test
        results = asyncio.run(test_bloombee_batch_inference())
        
        # Run comparison test
        asyncio.run(test_single_inference_comparison())
        
        logger.info("\nAll tests completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise

if __name__ == "__main__":
    main() 