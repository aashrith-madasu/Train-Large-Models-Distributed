## Todos
* checkpointing - save and load
* train Lora model
* train QLora model
* mixed precision
* bfloat?
* Quantizations - GGUF
* Train recipiecs - Iterative Curriculum Learning, 
* RL for LLMs - GRPO, PPO
* Distillation - from reasoning models (good project)
* MOE architectures
* Unsloth - for fast training
* vLLM - for fast inference
* LLM roles 
* special tokens (for reasoning)



## Notes
- Accelerator alone uses DDP backend in which only data is sharded and model is replicated - can give OOM for large models
- Use FSDP/Deepspeed backend for models sharding/parallelism along with data parallel 
- On each GPU, the corresponding model shard run the data shard it recieves
- Reshard after forward is te
- Activation Checkpointing (works complimentarily to model sharding) reduces memory footprint further by not 
    storing intermediate activations and thus re-computing it during backward (more processing time)
- CPU-Offload offloads parameters to CPU thus reducing memory footprint.
- FSDP Info: https://github.com/facebookresearch/fairseq/blob/main/examples/fully_sharded_data_parallel/README.md



## Typical Issues

- CPU Offloading not working 
    - requires mapping the model first to CPU before moving it to CUDA devices, IDK how to do it properly.

- Lora + Activation Checkpointing (AC) can give errors 
    - maybe because a lot of params of the base model are frozen and AC may require require_grad=True

- 

## Misc
For live GPU memory usage:
```
watch -n 1 nvidia-smi
```

## Tests

- Working:
    - context_length=2k, batch_size=1, model (Lora, mp-bf16), FSDP (AC, RAM-Eff, mp-bf16)
    - context_length=4k, batch_size=1, model (Lora, mp-bf16), FSDP (AC, RAM-Eff, mp-bf16)
    - context_length=6k, batch_size=1, model (Lora, mp-bf16), FSDP (AC, RAM-Eff, mp-bf16)
    - context_length=8k, batch_size=1, model (Lora, mp-bf16), FSDP (AC, RAM-Eff, mp-bf16)
    - context_length=8k, batch_size=1, model (Lora, Quant-4bit + mp-bf16), DDP (mp-bf16)

- Testing:

- Todo:
    - NOT WORKING !!! context_length=8k, batch_size=1, model (Lora, Quant-4bit + mp-bf16), FSDP (AC, RAM-Eff, mp-bf16)



## Future Work
- Quant + FSDP 
