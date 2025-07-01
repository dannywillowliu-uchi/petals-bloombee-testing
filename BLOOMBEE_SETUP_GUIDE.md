# BloomBee Batch Inference Testing Setup Guide

This guide will help you set up BloomBee to test batch inferencing capabilities.

## Prerequisites

- Python 3.8+
- BloomBee installed: `pip install bloombee`
- Sufficient GPU memory for Llama-7b model blocks

## Step 1: Start the DHT Server

The DHT (Distributed Hash Table) server acts as the coordinator for the BloomBee network.

```bash
# Start the DHT server
python -m bloombee.cli.run_dht --host_maddrs /ip4/0.0.0.0/tcp/31340
```

**Look for output like:**
```
To connect other peers to this one, use --initial_peers /ip4/YOUR_IP_ADDRESS/tcp/31340/p2p/QmefxzDL1DaJ7TcrZjLuz7Xs9sUVKpufyg7f5276ZHFjbQ
```

**Save this address** - you'll need it for the next steps.

## Step 2: Start BloomBee Workers

BloomBee workers hold different parts of the model. You need multiple workers to handle the full model.

### Worker 1 (Blocks 0-15)
```bash
python -m bloombee.cli.run_server huggyllama/llama-7b \
  --initial_peers /ip4/YOUR_IP_ADDRESS/tcp/31340/p2p/QmefxzDL1DaJ7TcrZjLuz7Xs9sUVKpufyg7f5276ZHFjbQ \
  --num_blocks 16
```

### Worker 2 (Blocks 16-31)
```bash
python -m bloombee.cli.run_server huggyllama/llama-7b \
  --initial_peers /ip4/YOUR_IP_ADDRESS/tcp/31340/p2p/QmefxzDL1DaJ7TcrZjLuz7Xs9sUVKpufyg7f5276ZHFjbQ \
  --num_blocks 16
```

**Note:** Replace `/ip4/YOUR_IP_ADDRESS/tcp/31340/p2p/QmefxzDL1DaJ7TcrZjLuz7Xs9sUVKpufyg7f5276ZHFjbQ` with the actual address from Step 1.

## Step 3: Update Test Script

Edit `test_bloombee_simple.py` and update the `initial_peers` list:

```python
# Connect to BloomBee (you'll need to update this with your DHT server address)
initial_peers = ["/ip4/YOUR_IP_ADDRESS/tcp/31340/p2p/QmefxzDL1DaJ7TcrZjLuz7Xs9sUVKpufyg7f5276ZHFjbQ"]
```

## Step 4: Run Batch Inference Test

```bash
python test_bloombee_simple.py
```

## Expected Results

The test will compare:
1. **Sequential Processing**: 3 separate inference calls
2. **Batch Processing**: 3 prompts processed together

### True Batching Detection

If batch inference is working, you should see:
- ✅ **TRUE BATCHING DETECTED**: Batch is 2-4x faster than sequential
- Batch time significantly less than sequential total time
- Efficiency factor > 1.5x

### No Batching Detection

If batch inference is not working, you'll see:
- ❌ **NO BATCHING DETECTED**: Batch time similar to sequential
- Efficiency factor close to 1.0x

## Troubleshooting

### Connection Issues
- Ensure DHT server is running and accessible
- Check that initial_peers address is correct
- Verify network connectivity and firewall settings

### Model Loading Issues
- Ensure sufficient GPU memory for model blocks
- Check that workers are running and connected to DHT
- Verify model name and DHT prefix match

### Performance Issues
- Monitor GPU utilization during inference
- Check for network bottlenecks
- Ensure workers are on machines with adequate compute resources

## Alternative: Use BloomBee Benchmark

You can also use the built-in benchmark script:

```bash
python benchmarks/benchmark_inference.py \
  --model huggyllama/llama-7b \
  --initial_peers /ip4/YOUR_IP_ADDRESS/tcp/31340/p2p/QmefxzDL1DaJ7TcrZjLuz7Xs9sUVKpufyg7f5276ZHFjbQ \
  --torch_dtype float32 \
  --seq_len 128
```

## Understanding Batch Inference in BloomBee

BloomBee uses a distributed architecture where:
- **DHT Server**: Coordinates the network and routes requests
- **Workers**: Hold different parts of the model (transformer blocks)
- **Client**: Sends batch requests to the distributed model

True batch inference occurs when:
1. Multiple prompts are sent in a single request
2. All prompts are processed together through the distributed model
3. The total time is significantly less than processing prompts sequentially

## Performance Expectations

Based on the [BloomBee repository](https://github.com/ai-decentralized/BloomBee), you should expect:
- **Good batching efficiency**: 2-4x improvement over sequential
- **Scalable throughput**: Performance improves with batch size
- **Distributed processing**: Workload shared across multiple workers

## Next Steps

Once you have BloomBee running:
1. Test different batch sizes (1, 2, 4, 8, 16)
2. Compare with local model performance
3. Analyze scaling patterns
4. Test with different model sizes and configurations 