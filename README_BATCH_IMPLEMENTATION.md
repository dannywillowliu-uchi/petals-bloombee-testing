# Petals True Batch Inference Implementation Plan

## Executive Summary

**Current State**: Petals supports "batch processing" but processes batches **sequentially** (one after another)

**Target State**: Implement **true parallel batch inference** where multiple prompts are processed simultaneously

## Problem Analysis (Based on Your Research)

### Current Limitations

1. **Sequential Processing**: 
   - Current: Processes batches one by one via `async def sequential_forward`
   - Issue: No true parallelism, just sequential execution

2. **Missing Parallel Dimension**: 
   - Current: `[batch_size, seq_length, hidden_size]` (2D effective processing)
   - Needed: `[parallel_batch, batch_size, seq_length, hidden_size]` (3D parallel processing)

3. **Token-Based Batching Limits**: 
   - Current: `MAX_TOKENS_IN_BATCH = 1024` limits total tokens, not parallel streams

## Implementation Requirements

### Primary Goals
1. ✅ **Batch 1 prompt 16 times** - Process same prompt with different parameters
2. ✅ **Add 3rd dimension** - True parallel batch processing  
3. ✅ **Performance testing** - Measure parallel vs sequential performance
4. ✅ **LLaMA backend modification** - Start with LLaMA models

### Performance Targets
- Process N identical prompts in parallel (not sequentially)
- Achieve 2-4x speedup for batch processing
- Maintain memory efficiency (<50% overhead)
- Support different batch sizes dynamically

## Core Changes Required

### 1. RemoteSequential Parallel Processing
**File**: `petals/client/remote_sequential.py`

**Add parallel processing method:**
```python
def forward(self, inputs: torch.Tensor, prompts: Optional[torch.Tensor] = None, 
           parallel_batch_size: int = 1, **kwargs):
    
    if parallel_batch_size > 1:
        # New: 4D parallel processing
        assert inputs.ndim == 4, "Expected [parallel_batch, batch_size, seq_length, hidden_size]"
        return self._parallel_forward(inputs, prompts, parallel_batch_size, **kwargs)
    else:
        # Existing: 3D sequential processing  
        assert inputs.ndim == 3, "Expected [batch_size, seq_length, hidden_size]"
        return self._sequential_forward(inputs, prompts, **kwargs)

def _parallel_forward(self, inputs: torch.Tensor, prompts: Optional[torch.Tensor], 
                     parallel_batch_size: int, **kwargs):
    """Process multiple batches in TRUE PARALLEL instead of sequentially"""
    parallel_batch, batch_size, seq_length, hidden_size = inputs.shape
    
    # Reshape for parallel processing
    inputs_flat = inputs.view(-1, seq_length, hidden_size)
    
    # KEY CHANGE: Process all batches simultaneously instead of sequentially
    outputs_flat = self.sequence_manager.forward_parallel(inputs_flat, prompts, **kwargs)
    
    # Reshape back to parallel structure
    outputs = outputs_flat.view(parallel_batch, batch_size, seq_length, hidden_size)
    return outputs
```

### 2. LLaMA Model Parallel Support  
**File**: `petals/models/llama/model.py`

**Update forward pass for 4D tensors:**
```python
def forward(self, input_ids: Optional[torch.LongTensor] = None, 
           parallel_batch_size: int = 1, **kwargs):
    
    # Handle input processing
    if input_ids is not None:
        inputs_embeds = self.embed_tokens(input_ids)
    
    # Support both 3D and 4D inputs
    if parallel_batch_size > 1 and inputs_embeds.ndim == 3:
        # Convert to 4D: [1, batch_size, seq_length, hidden_size]
        inputs_embeds = inputs_embeds.unsqueeze(0)
    
    # Process prompts for parallel batching
    if use_prompts:
        if inputs_embeds.ndim == 4:  # Parallel mode
            parallel_batch, batch_size, seq_length, hidden_size = inputs_embeds.shape
            prompts, intermediate_prompts = self.get_prompt_parallel(parallel_batch, batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=2)
        else:  # Sequential mode (existing)
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
    
    # Process through transformer layers with parallel support
    hidden_states = self.h(inputs_embeds, parallel_batch_size=parallel_batch_size, **kwargs)
    
    return BaseModelOutputWithPast(last_hidden_state=hidden_states)
```

## Expected Performance Results

```
Parallel Batch Performance Results:
============================================================
Batch Size   Sequential   Parallel     Speedup
------------------------------------------------------------
1            0.850        0.850        1.00x (baseline)
2            1.600        0.950        1.68x  
4            3.100        1.200        2.58x
8            6.000        1.800        3.33x
16           11.500       2.500        4.60x

Average Speedup: 2.64x
Memory Overhead: <50%
```

## Implementation Timeline

### Week 1: Core Infrastructure  
- [ ] Implement `ParallelSequentialAutograd`
- [ ] Modify `RemoteSequential._parallel_forward()`
- [ ] Add 4D tensor dimension handling
- [ ] Basic unit tests

### Week 2: LLaMA Integration
- [ ] Update `DistributedLlamaModel.forward()`
- [ ] Implement `get_prompt_parallel()`
- [ ] Add `generate()` with `parallel_batch_size`
- [ ] Integration tests

### Week 3: Server Optimization
- [ ] Update backend memory estimation
- [ ] Modify server parallel batch limits
- [ ] Optimize caching for parallel processing
- [ ] Performance tuning

### Week 4: Testing & Documentation
- [ ] Comprehensive test suite
- [ ] Performance benchmarking
- [ ] Memory usage analysis  
- [ ] Usage documentation

## Key Files to Modify

| File | Changes Required |
|------|------------------|
| `petals/client/remote_sequential.py` | Add `_parallel_forward()` method |
| `petals/client/sequential_autograd.py` | Add `ParallelSequentialAutograd` class |
| `petals/models/llama/model.py` | 4D tensor support, `get_prompt_parallel()` |
| `petals/models/llama/modeling_llama.py` | Enhanced `generate()` with parallel support |
| `petals/server/backend.py` | Parallel memory estimation |
| `petals/server/server.py` | Parallel batch configuration |

## Success Criteria

1. **Functionality**: Process 1 prompt 16 times in parallel
2. **Performance**: 2-4x speedup vs sequential processing  
3. **Memory**: <50% additional memory overhead
4. **Compatibility**: Zero breaking changes
5. **Testing**: Comprehensive test coverage 