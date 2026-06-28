# Image-Text Token Attention

在 Diffusion Model 中，prompt 文本会作为条件指导图片生成。每一轮去噪时，当前的 image latent token 和 prompt text token 会一起进入 Transformer Block，由 block 更新 image token，使它逐步变成更清晰、也更符合 prompt 的图像 latent。

这篇文章主要看两个问题：

1. image token 和 text token 如何做 joint attention。
2. 在这个 joint attention 中，RoPE 如何同时支持 image/text 两类 token 的位置编码。

下面以 Qwen-Image 的 Transformer block 为例说明。

## 1. Image/Text Token 的 Joint Attention

在 text-to-image diffusion 中，Transformer block 的输入通常可以分成两路：

- image stream：当前 timestep 的 noisy latent tokens，记作 $X_i \in R^{B \times S_i \times D}$。
- text stream：prompt encoder 输出的条件 tokens，记作 $X_t \in R^{B \times S_t \times D}$。

Qwen-Image 采用 double-stream / joint attention 结构。image/text 两路先各自计算 Q/K/V：

$$
Q_i = X_i W_Q^i,\quad K_i = X_i W_K^i,\quad V_i = X_i W_V^i
$$

$$
Q_t = X_t W_Q^t,\quad K_t = X_t W_K^t,\quad V_t = X_t W_V^t
$$

然后在 sequence 维度拼接：

$$
Q = [Q_t; Q_i],\quad K = [K_t; K_i],\quad V = [V_t; V_i]
$$

再做一次 non-causal attention：

$$
O = \text{softmax}\left(\frac{QK^T}{\sqrt d} + M\right)V
$$

如果按 text / image 两类 token 分块，attention score 可以写成：

$$
\begin{bmatrix}
Q_tK_t^T & Q_tK_i^T \\
Q_iK_t^T & Q_iK_i^T
\end{bmatrix}
$$

其中 $Q_iK_t^T$ 是 image token attend prompt token，也是 prompt 影响 image latent 更新的主要路径；$Q_tK_i^T$ 表示 text token 也可以读取当前 image token 状态。相比单向 cross-attention，joint attention 让 image/text 两路在同一个 block 中双向交互。

attention 之后，输出会再拆回 text part 和 image part，分别写回两路 stream：

```python
joint_query = concat([txt_query, img_query], dim=seq)
joint_key = concat([txt_key, img_key], dim=seq)
joint_value = concat([txt_value, img_value], dim=seq)

joint_output = attention(joint_query, joint_key, joint_value)

txt_out = joint_output[:, :text_seq_len]
img_out = joint_output[:, text_seq_len:]
```

这就是 Qwen-Image block 中 image/text token 交互的主干。

## 2. Joint Attention 中如何支持 RoPE

Joint attention 把 image token 和 text token 拼成同一个序列，但两类 token 的位置结构并不一样：

- text token 是一维序列位置。
- image token 来自 latent grid，更自然的位置是 frame / height / width。

所以这里不能只问“attention 怎么拼接”，还要问：**Q/K 在进入 joint attention 前，如何带上各自的位置？**

### 2.1 RoPE 为什么能够 Work

RoPE 的核心做法是：不把 position embedding 加到 hidden states 上，而是在 attention 前对 Q/K 做旋转。

对位置 $p$ 上的 query/key，可以抽象成：

$$
\tilde q_p = R_p q,\quad \tilde k_p = R_p k
$$

其中 $R_p$ 是由位置 $p$ 决定的旋转矩阵。attention score 变成：

$$
\tilde q_m^T \tilde k_n
=
q^T R_m^T R_n k
=
q^T R_{n-m} k
$$

也就是说，两个 token 的 attention score 会自然包含相对位置信息 $n-m$。这也是 RoPE 常用于 attention 的原因：它不改变 V，只修改 Q/K，让位置关系直接进入 $QK^T$。

对于二维或三维 token，也可以把 head dim 切成几段，分别给不同轴使用。例如 image token 的位置可以写成 $(f, h, w)$，RoPE 可以分别为 frame、height、width 生成频率，再 concat 成一个完整的 rotary embedding。

### 2.2 Qwen-Image 中的 RoPE

Qwen-Image 的 RoPE 可以理解成两步：

1. 为 image token 和 text token 分别生成 RoPE 频率。
2. 在 concat 做 joint attention 之前，分别把 RoPE 应用到 image/text 的 Q/K 上。

在实现中，image token 使用 3D RoPE。默认配置里 `axes_dims_rope=(16, 56, 56)`，对应 frame / height / width 三个轴的 rotary 维度。对于 image latent grid，模型会根据 `img_shapes=(frame, height, width)` 生成每个 image token 的 3D 位置频率：

```python
img_freqs = rope(frame, height, width)
```

text token 则使用同一套频率表中的一段连续 1D 位置。Qwen-Image 会先根据 image 的 height/width 计算 `max_vid_index`，然后让 text position 从这个 index 后面开始：

```python
txt_freqs = pos_freqs[max_vid_index : max_vid_index + text_seq_len]
```

这样 image token 和 text token 都有自己的 RoPE position，并且位置范围不会简单重叠。

在 attention processor 里，RoPE 的应用顺序大致是：

```python
img_query = apply_rope(img_query, img_freqs)
img_key = apply_rope(img_key, img_freqs)

txt_query = apply_rope(txt_query, txt_freqs)
txt_key = apply_rope(txt_key, txt_freqs)

joint_query = concat([txt_query, img_query], dim=seq)
joint_key = concat([txt_key, img_key], dim=seq)
joint_value = concat([txt_value, img_value], dim=seq)
```

注意这里 RoPE 是在 concat 前分别作用到 image/text 的 Q/K 上；concat 后的 joint attention 直接使用已经带有位置信息的 Q/K。

对于 image-image attention，RoPE 表达的是 latent grid 中 token 的空间相对关系；对于 text-text attention，RoPE 表达的是 prompt token 的序列关系；对于 image-text attention，RoPE 提供了两类 token 在同一个 rotary space 下的相对位置偏置，模型再通过训练学习如何利用这种跨模态位置关系。

## 3. 简单总结

Qwen-Image 中 image-text token attention 的核心流程是：

1. image/text 两路分别计算 Q/K/V。
2. 在 Q/K 上分别应用 RoPE：image 使用 3D RoPE，text 使用单独的 1D position range。
3. 将 text 和 image 的 Q/K/V concat 成 joint sequence。
4. 做一次 non-causal joint attention。
5. attention output 再 split 回 image stream 和 text stream。

所以这套机制可以概括为：**RoPE 先为两类 token 注入各自的位置结构，joint attention 再让两类 token 在同一个 attention 空间里交互。**

## References

- [Qwen-Image Blog](https://qwenlm.github.io/blog/qwen-image/)
- [QwenImageTransformer2DModel in Diffusers](https://huggingface.co/docs/diffusers/en/api/models/qwenimage_transformer2d)
- [Diffusers Qwen-Image Transformer Source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage.py)
