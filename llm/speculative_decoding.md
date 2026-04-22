## 0. Speculative decoding

标准 speculative decoding（也叫 speculative sampling）就是：

1. **drafter（草稿模型）** 先一次性猜一串未来 token
2. **target（大模型）** 用并行方式对这串 token 做验证（算出每一步真实分布）
3. 逐个 token 按接受规则“能收就收，不能收就停”，保证 **最终采样分布不变（lossless）**

EAGLE 系列、DFlash 都是在改进 **“草稿怎么更快更准地产生”** 以及 **“怎么提高一次验证能接受的长度（acceptance length）”**。

---

## 1) EAGLE-1（原始 EAGLE）：为什么要用“倒数第二层特征”

### 1.1 “倒数第二层特征”到底是哪一层？

EAGLE 论文里把 **LM head（词表线性层 + softmax）**看作最顶层，所以它说的 **second-to-top-layer feature**，就是：

* **LM head 之前的隐藏向量**（hidden state before LM head）
* 也就是“最后一个 Transformer block 输出（通常再接一个 RMSNorm）得到的向量”

论文的符号定义非常直接：输入 token 序列 $T_{1:j}$ 经过 embedding 和主干网络得到 feature 序列 $F_{1:j}$，并且 **LM head 把 $f_j$ 映射成下一个 token 的分布 $p_{j+1}$**：
$$p_{j+1} = \text{LMHead}(f_j)$$
再从 $p_{j+1}$ 采样得到 $t_{j+1}$。 ([arXiv][1])

> 所以：**”倒数第二层特征 $f_j$”本质上就是”用于预测下一个 token 的那根隐藏向量”。**

### 1.2 为啥在 feature 空间做 draft 更“好学”？

EAGLE 的观察是：token 是离散符号，直接多步预测容易错；而 feature 是连续向量序列，规律性更强，因此“在 feature 上做自回归，再用 LM head 映射回 token”更稳定。论文在引言/预备知识里明确把 feature 定义为 “second-to-top-layer features… before the LM head”，并强调这种 feature-level autoregression 更容易。 ([arXiv][1])

---

## 2) EAGLE-1 的关键：**“提前一步的 token 序列”解决 feature 不确定性**

### 2.1 不确定性来自哪？

如果你只看 $f_i$，下一步 token 可能采样出不同结果（比如 “am” 或 “always”），那 **下一步 feature $f_{i+1}$** 也会分叉成不同轨迹。论文用图例直接说明：**feature 是连续的，没法像 token 那样”分叉采样”来覆盖所有可能**，因此”只用 $f_i$ 去预测 $f_{i+1}$”天生不确定。 ([arXiv][1])

### 2.2 EAGLE 怎么解：用“提前一步的 token 序列”做条件

EAGLE 的 drafting 逻辑（非常关键）是：

* 想预测 $f_{i+1}$，你需要知道位置 $i+1$ 上到底喂进了哪个 token（因为 $f_{i+1}$ 是”读入 $t_{i+1}$ 后”的隐藏状态）。
* 所以 EAGLE 先从 $p_{i+1}=\text{LMHead}(f_i)$ **采样出 $t_{i+1}$**
* 然后用 **$(f_{1:i})$ + “提前一步的 token 序列” $(t_{2:i+1})$** 去预测 $f_{i+1}$

论文把这点写得很直白：EAGLE 用 **feature 序列 $(f_1,f_2)$** 和 **提前一步的 token 序列 $(t_2,t_3)$** 来预测 $f_3$，然后由 $p_4=\text{LMHead}(f_3)$ 采样 $t_4$，再继续滚动。 ([arXiv][1])

### 2.3 具体计算图（按实现拆开）

EAGLE 的 draft model 由三部分组成，其中 embedding 和 LM head 直接复用 target（冻结）： ([arXiv][1])

对每个时间步（或每次要外推下一步 feature）大致是：

1. **取 target 的特征序列**：$F_{1:i}$（每个 $f_k\in\mathbb{R}^d$）
2. **把提前一步 token 序列** $T_{2:i+1}$ 过 embedding 得到 $E_{2:i+1}\in\mathbb{R}^{d}$
3. **拼接融合**：$[F_{1:i}; E_{2:i+1}] \in \mathbb{R}^{2d}$
4. 过一个 **FC 降维回 $d$**，再过一个 **decoder layer**（自回归头），输出 $\hat f_{i+1}$ ([arXiv][1])
5. 用 target 的 LM head 得到下一 token 分布：$\hat p_{i+2}=\text{Softmax}(\text{LMHead}(\hat f_{i+1}))$ ([arXiv][1])

### 2.4 训练目标长啥样（核心公式）

EAGLE 同时做：

* **feature 回归（Smooth L1）**：让 $\hat f_{i+1}$ 靠近真 $f_{i+1}$
* **token 分布对齐（交叉熵）**：让 $\text{LMHead}(\hat f_{i+1})$ 给出的分布靠近 $\text{LMHead}(f_{i+1})$

论文给了明确公式：
$$L_{\text{reg}}=\text{SmoothL1}(f_{i+1}, \text{DraftModel}(T_{2:i+1},F_{1:i}))$$
以及
$$p_{i+2}=\text{Softmax}(\text{LMHead}(f_{i+1})), \quad \hat p_{i+2}=\text{Softmax}(\text{LMHead}(\hat f_{i+1}))$$
$$L_{\text{cls}}=\text{CrossEntropy}(p_{i+2},\hat p_{i+2}), \quad \text{总损失 } L=L_{\text{reg}}+w_{\text{cls}}L_{\text{cls}}$$
([arXiv][1])

> 这就是你问的“倒数第二层特征的计算逻辑”：
> **target 负责给真 feature；draft head 学会在“已采样到的下一 token”条件下外推下一 feature；LM head 把 feature 变回 token 分布。**

---

## 3) 有 EAGLE-2 吗？有，而且它主要改“树形 draft 分配”

有的，EAGLE-2 叫 **Dynamic Draft Trees**。它的关键发现是：**token 被接受不仅和位置有关，还强烈依赖上下文**，因此应该动态决定“把草稿预算花在哪些分支上”。 ([arXiv][2])

它在树里给每个节点（token）定义”全局可接受价值”近似为沿路径接受率乘积：
$$V_i \approx \prod_{t_j\in Path(root,t_i)} p_j$$
并用 draft model 的置信度去近似接受率，从而选择 top-k 节点扩展，做到”更聪明的 draft tree”。 ([arXiv][2])

---

## 4) EAGLE-3：为什么说它“去掉 feature 约束”，并做多层特征融合

EAGLE-3 论文总结了两大改动：

1. **移除 feature prediction constraint（不再强制回归 feature）**，改为通过 **training-time test** 在训练时就模拟多步生成，从而直接优化“多步 token 草稿质量”。 ([arXiv][3])
2. 不再只复用“top-layer feature”，而是 **融合低/中/高层特征**，获取更丰富信息。 ([arXiv][3])

它报告的总体加速区间大概 **3.0×–6.5×**，并且指出 training-time test 让接受率在“输入包含更多 draft 自己生成的内容”时更稳定。 ([arXiv][3])

---

## 5) DFlash：用“块扩散（block diffusion）”把 drafting 并行化

DFlash（2026-02 的论文）核心是：**现有 speculative 的 drafter 多数还是自回归（串行）**，所以它用一个很小的 **block diffusion drafter**，在**一次 forward**里并行生成一个 token block（masked positions 一起 denoise），降低 drafting latency。 ([arXiv][4])

它还强调两点工程/结构设计（和 EAGLE-3形成对比）：

* **从 target 抽多层 hidden feature**（浅到深均匀采样），拼接后过轻量投影，得到“target context feature”来条件化 drafter。 ([arXiv][4])
* 不只是把这个特征当“输入”，而是把它**注入到每一层 draft model 的 K/V 投影里（KV injection）**，并缓存复用，使条件信息不被层数加深而稀释。 ([arXiv][4])

论文声称在多任务上做到 **>6× lossless acceleration**，并说相对 EAGLE-3 可到 **2.5× 更高 speedup**（具体取决于模型/任务/实现）。 ([arXiv][4])

---

## 6) DeepSeek-V3 的 MTP（Multiple Token Prediction）和 EAGLE-1/2/3 是啥关系？

DeepSeek-V3 的技术报告把关系点名了：

* 它的原则是 **”maintaining the causal chain”**，并说这个原则 **类似 EAGLE**，但 **EAGLE 的主要目标是 speculative decoding**，而 DeepSeek 的 MTP **主要用于改进训练**。 ([arXiv][5])

### 6.1 MTP 的计算公式（和 EAGLE 的”提前一步 token”非常像）

DeepSeek-V3 的 MTP 用 $D$ 个串联模块去预测额外的 $D$ 个未来 token。第 $k$ 个模块把：

* 上一深度的表示 $h^{k-1}_i$
* 以及 **未来第 $i+k$ 个 token 的 embedding $\text{Emb}(t_{i+k})$**

做 RMSNorm 后拼接，再过投影矩阵：
$$
h^{\prime k}_i = M_k[\text{RMSNorm}(h^{k-1}_i);\ \text{RMSNorm}(\text{Emb}(t_{i+k}))]
$$
([arXiv][5])

然后再过一个 Transformer block 得到 $h^k_i$，并用共享输出头预测第 $k$ 个额外 token 的分布：
$$P^k_{i+k+1} = \text{OutHead}(h^k_i)$$
([arXiv][5])

> 你可以把它理解成：
> **EAGLE：用“未来 token（已采样）”来消除 feature 外推的不确定性；**
> **MTP：用“未来 token（训练时 teacher forcing 的真值）”来构造更密集的多步预测训练信号，并保持因果链。**

### 6.2 MTP 在推理时能不能用来加速？

报告写得也很直接：

* 推理时可以 **直接丢掉 MTP 模块**，主模型照常工作
* 也可以 **把 MTP 模块 repurpose 成 speculative decoding 的 drafter** 来降低生成时延 ([arXiv][5])

### 6.3 和 EAGLE-1/2/3 的”对应关系”总结

* **共同点**：都在做”多步 lookahead”，并且都在强调**因果链/一步偏移**（EAGLE 用 $T_{2:i+1}$；MTP 用 $\text{Emb}(t_{i+k})$）来让多步预测在逻辑上可落地。 ([arXiv][1])
* **不同点**：

  * EAGLE-1 的核心变量是 **feature（LM head 前隐藏向量）**，再映射到 token；MTP 是**直接为多个未来 token 建额外预测路径/模块**。 ([arXiv][1])
  * EAGLE-2/3 是明确的 speculative decoding 系统改造（树、training-time test、多层特征融合）；MTP 本身更像”**训练时的多步辅助头**”，只是”可被改造成”drafter。 ([arXiv][2])

---

如果你愿意我继续往下挖，我可以把 **EAGLE-3 的 training-time test** 用更接近代码执行的方式（训练时怎样 roll-out、怎样构造输入特征融合、为何能缓解“训练-推理不一致”）拆成步骤讲；也可以把 **“EAGLE 的倒数第二层 feature”在实际框架里对应哪几个张量（含 RMSNorm 前后、KV cache 复用边界）**按常见实现（HF/vLLM/SGLang）给你对齐。

[1]: https://arxiv.org/pdf/2401.15077.pdf "https://arxiv.org/pdf/2401.15077.pdf"
[2]: https://arxiv.org/pdf/2406.16858 "https://arxiv.org/pdf/2406.16858"
[3]: https://arxiv.org/pdf/2503.01840 "https://arxiv.org/pdf/2503.01840"
[4]: https://arxiv.org/pdf/2602.06036 "https://arxiv.org/pdf/2602.06036"
[5]: https://arxiv.org/pdf/2412.19437 "https://arxiv.org/pdf/2412.19437"
