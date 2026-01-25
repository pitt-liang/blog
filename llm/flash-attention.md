# Attention

## Attention Mechanism

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

$$
Q \in \mathbb{R}^{T \times D}
$$

$$
K^T \in \mathbb{R}^{D \times S}
$$

$$
V \in \mathbb{R}^{S \times D}
$$

$$
O \in \mathbb{R}^{T \times D}
$$


## FlashAttention

## Tile in GEMM


![tiled-gemm](resources/tiled-gemm.png)

```python

for m in range(0, M, M_blk):
    for n in range(0, N, N_blk):
		C_tile = initialize_zero(C_tile)
        for k in range(0, K, K_blk):
            A_tile = A[m:m+M_blk, k:k+K_blk]
            B_tile = B[k:k+K_blk, n:n+N_blk]
            C_tile = C[m:m+M_blk, n:n+N_blk]
            C_tile += A_tile @ B_tile

```


## Flash Attention

![](resources/flash-attention-memory-access.png)




## Question:


- 启动的Kernel情况?


- How does KV Cache works？




```mermaid

%% 3D tiling visualization for matrix multiplication
%% Q [T x D] * K^T [D x S] -> I [T x S]
%% 一个输出 tile 是 T_blk x S_blk，内部通过 D_blk 累加完成
flowchart TB
    subgraph Cube["输出 I_tile (T_blk x S_blk)"]
        direction TB
        subgraph Ddim["沿 D 维分块"]
            i1["部分结果 (d0..d0+D_blk)"]
            i2["部分结果 (d1..d1+D_blk)"]
            i3["..."]
        end
    end
    i1 --> isum["累加"]
    i2 --> isum
    i3 --> isum
    isum --> Itile["最终 I_tile (T_blk x S_blk)"]



```


## References:

- [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)