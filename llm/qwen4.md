# Qwen4-Exp in MiniSGlang project

Qwen3.8-Flash-Next （Qwen4-Exp）是Qwen4 model 的预览版本，在Qwen3.8系列，引入了多项模型架构的更新：

- 基于 QSA + GDN 的 Hybrid Attention

- PLE (2/3-Embeddding)
  
- GatedRedisual Hyper Connection


## N-gram embedding

LLM初始的hidden states来源于token的embedding lookup，每一个embedding vector对应的是这个词的语义。如果需要获取某一个词在上下文中的更多信息语义，则需要通过Attention机制，查询与前文的关系/attention，从而获取对应的token在当前的上下文中的语义。

N-gram embedding 不同于采用Attention计算上下文信息，而是把常见的局部 token 组合直接 hash 到一个超大的 embedding table；通过一个Embedding Lookup查询[t-n, t]的token组合的embedding信息，从而获取更多对应token位置的更多信息。

N-gram Embedding 用 local context 做 lookup，以很小计算量扩展模型容量，而且大表可以放 host memory，通过 async prefetch 和主模型计算 overlap

在实际推理过程中，这个embedding table存储在host cpu memory中，lookup可以与主模型的transformer block 的计算overlap，从而实现并不显著扩大主模型的size同时，实现scaling。

Qwen3.8-Flash-Next的计算流程如下


```plain

token IDs
   │
   ├─ bigram  (t-1,t)
   └─ trigram (t-2,t-1,t)
           │
           ▼
     multi-head hash
     8 heads / ngram
           │
           ▼
16 embedding row IDs
           │
           ▼
huge N-gram embedding table
           │
      16 × 160 dim
           │ concat
           ▼
       2560 dim
           │
      ┌────┴─────┐
      ▼          ▼
   Key proj   Value proj
      │          │
      │       V_ngram
      ▼
 compare with current
 4-way residual state
      │
      ▼
 sigmoid gate
      │
      ▼
 gate × V_ngram
      │
 dilated depthwise causal conv
      │
      ▼
add into residual stream

```

vllm/sglang中的计算是类似的,采用的是 two stream overlap的方式。


Qwen3.8-Flash-Next 中，需要


类似DeepSeek提供的N Gram Embedding，使用2/3 gram embedding提供更多静态的context给到Hidden States


## TODOs


- 参考SGLang里的逻辑, 实现embedding的overlap


- 



## Questions:

- 如何支持 PLE 与 model forward 并行逻辑？


```plain

Main Stream                                 PLE Stream

input
  │
compute ngram IDs
  │
  ├──────── launch ────────────────────────> UVA lookup
  │                                            │
Layer 1                                        │
Attention / GDN                                │
MoE                                            │
  │                                            │
  │                                            ▼
  └──────────── Layer 2 ──────────────── wait_stream()
                                             │
                                        lookup finished
                                             │
                           <─────────────────┘
  │
PLE key/value projection
PLE query(hidden)
PLE gating
PLE short conv
  │
Layer 2 normal compute

```

采用triton kernel，使用UVA的方式load 访问NGram Embedding Table.


- 所以这里的PLE 的lookup 还是一个GPU Kernel 对吗？采用Triton实现？ 才可以用一个CUDA stream 来submit 对应kernel/wait对应kernel， fetch到的embeddings 是直接到 GPU的？

- 这里为什么能够支持Kernel能够直接访问对应的CPU上的Embedding？虽然是Pin memory，但是Triton 不是访问的GPU memory吗？这里所以的 UVA（Unified Virtual Addressing） ? 是否直接pin memory就能够支持 UVA？Triton Kernel访问时，怎么知道对应的memory是在GPU还是CPU上的？

- 这里的算子实现具体是如何做的？通过ngram ids 是怎么fetch到对应的embedding的？这里是否需要load完整的 embedding 到 CUDA？还是一个batch的访存操作？


- 数值敏感吗，是否适合量化的策略？适合什么量化的策略？


- PLE 的Embedding具体是如何计算的？


- 基于QSA/Linear Attention的计算

- Two Batch Overlap 为什么与PLE并不兼容



