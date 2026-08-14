> Source: https://jax-ml.github.io/scaling-book/transformers/ — Austin et al., "How To Scale Your Model", Google DeepMind, 2025

---

# All the Transformer Math You Need to Know

Part 4 of [How To Scale Your Model](/scaling-book) ([Part 3:
Sharding](../sharding) \| [Part 5: Training](../training))

Here we'll do a quick review of the Transformer architecture,
specifically how to calculate FLOPs, bytes, and other quantities of
interest.

### Authors

### 

### Affiliation

<a href="https://www.jacobaustin.org/" class="name"
rel="external nofollow noopener" target="_blank">Jacob Austin</a>

<span class="affiliation">Google DeepMind</span>

<a href="https://x.com/_sholtodouglas" class="name"
rel="external nofollow noopener" target="_blank">Sholto Douglas</a>

<span class="affiliation"></span>

<a href="https://cs.stanford.edu/~rfrostig/" class="name"
rel="external nofollow noopener" target="_blank">Roy Frostig</a>

<span class="affiliation"></span>

<a href="https://anselmlevskaya.com/" class="name"
rel="external nofollow noopener" target="_blank">Anselm Levskaya</a>

<span class="affiliation"></span>

<a href="https://x.com/charliexychen" class="name"
rel="external nofollow noopener" target="_blank">Charlie Chen</a>

<span class="affiliation"></span>

<a href="https://sharadvikram.com/" class="name"
rel="external nofollow noopener" target="_blank">Sharad Vikram</a>

<span class="affiliation"></span>

<a href="https://fedelebron.com/" class="name"
rel="external nofollow noopener" target="_blank">Federico Lebron</a>

<span class="affiliation"></span>

<a href="https://x.com/pchoy95" class="name"
rel="external nofollow noopener" target="_blank">Peter Choy</a>

<span class="affiliation"></span>

<a href="https://x.com/vinayramasesh" class="name"
rel="external nofollow noopener" target="_blank">Vinay Ramasesh</a>

<span class="affiliation"></span>

<a href="https://representation.ai/" class="name"
rel="external nofollow noopener" target="_blank">Albert Webson</a>

<span class="affiliation"></span>

<a href="https://x.com/reinerpope" class="name"
rel="external nofollow noopener" target="_blank">Reiner
Pope<sup>*</sup></a>

<span class="affiliation"></span>

### Published

Feb. 4, 2025

### Contents

[Counting Dots](#counting-dots)

[](#)

- [Forward and reverse FLOPs](#forward-and-reverse-flops)

[Transformer Accounting](#transformer-accounting)

[Global FLOPs and Params
Calculation](#global-flops-and-params-calculation)

[Miscellaneous Math](#miscellaneous-math)

[](#)

- [Sparsity and Mixture-of-Experts](#sparsity-and-mixture-of-experts)
- [Gradient checkpointing](#gradient-checkpointing)
- [Key-Value (KV) caching](#key-value-kv-caching)

[What Should You Take Away from this
Section?](#what-should-you-take-away-from-this-section)

[A Few Problems to Work](#a-few-problems-to-work)

[Appendix](#appendix)

[](#)

- [Appendix A: How does Flash Attention
  work?](#appendix-a-how-does-flash-attention-work)

## Counting Dots

Let’s start with vectors \\x\\, \\y\\ and matrices \\A\\, \\B\\ of the
following shapes:

\\\def \red#1{\textcolor{red}{#1}} \def \green#1{\textcolor{green}{#1}}
\def \blue#1{\textcolor{blue}{#1}} \def
\purple#1{\textcolor{purple}{#1}} \def \orange#1{\textcolor{orange}{#1}}
\def \gray#1{\textcolor{gray}{#1}} \begin{array}{cc} \textrm{array} &
\textrm{shape} \\ \hline x & \textrm{\[P\]} \\ y & \textrm{\[P\]} \\ A &
\textrm{\[N P\]} \\ B & \textrm{\[P M\]} \\ \hline \end{array}\\

- A dot product of \\x \cdot y\\ requires \\P\\ *adds* and *multiplies*,
  or \\2P\\ floating-point operations total.
- A matrix-vector product \\Ax\\ does \\N\\ dot-products along the rows
  of \\A\\, for \\2NP\\ FLOPs.
- A matrix-matrix product \\AB\\ does a matrix-vector product for each
  of the \\M\\ columns of \\B\\, for \\2NPM\\ FLOPs total.
- In general, if we have two higher-dimensional arrays \\C\\ and \\D\\,
  where some dimensions are <span style="color:red">CONTRACTING</span>
  and some are <span style="color:blue">BATCHING</span> (e.g.
  \\C\[\blue{GH}IJ\red{KL}\], D\[\blue{GH}MN\red{KL}\]\\), then the
  FLOPs cost of this contraction is two times the product of all of the
  \\C\\ and \\D\\ dimensions where the batch and contraction dimensions
  are only counted once (e.g. \\2\blue{GH}IJMN\red{KL}\\). Note that a
  dimension is only batching if it occurs in both multiplicands. (Note
  also that the factor of 2 won’t apply if there are no contracting
  dimensions and this is just an elementwise product.)**Contracting**
  dimensions are axes that are summed over during the operation (they
  appear in both inputs but not in the output), like the inner dimension
  in a matrix multiply. **Batching** dimensions are shared axes that
  appear in both inputs and are carried unchanged to the output; they
  index independent subproblems and aren't multiplied together in FLOP
  counts. In einsum terms: labels present on both inputs and the output
  are batching; labels present on both inputs but absent from the output
  are contracting.

\\\begin{array}{ccc} \textrm{Operation} & \textrm{FLOPs} & \textrm{Data}
\\ \hline x \cdot y & 2P & 2P \\ A x & 2NP & NP + P \\ AB & 2NPM & NP +
PM \\ \[c_0,...,c_N\] \cdot \[d_0,...,d_N\] & 2 \prod c_i \times
\prod\_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}}
d_j & \prod c_i + \prod d_j \\ \hline \end{array}\\

Make note of the fact that for a matrix-matrix multiply, the *compute*
scales cubically \\O(N^3)\\ while the data transfer only scales
quadratically \\O(N^2)\\ — this means that as we scale up our matmul
size, it becomes *easier* to hit the compute-saturated limit. This is
extremely unusual, and explains in large part why we use architectures
dominated by matrix multiplication — they’re amenable to being scaled!

<figure>
<img src="/scaling-book/assets/img/matmul-flops.gif" class="img-fluid"
style="width:100.0%" loading="lazy"
onerror="this.onerror=null; $(&#39;.responsive-img-srcset&#39;).remove();" />
</figure>

### Forward and reverse FLOPs

During training, we don’t particularly care about the result of a given
matrix multiply; we really care about its derivative. It turns out that
calculating that derivative costs about 3x more than just doing the
matmul itself.

If we imagine **B** is just one matrix in a larger network and **A** are
our input activations with **C = A B**, the derivative of the loss **L**
with respect to **B** is given by the chain rule:

\\\frac{\partial L}{\partial B} = \frac{\partial L}{\partial
C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial
C}\right)\\

which requires \$2NPM\$ FLOPs to compute (since it contracts over the
\$N\$ dimension). Likewise, the derivative of the loss with respect to
**A** is

\\\frac{\partial L}{\partial A} = \frac{\partial L}{\partial
C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial
C}\right) B^T\\

which is again \$2NPM\$ FLOPs since **dL/dC** is a matrix of size \\\[N,
M\]\\. While this quantity isn’t the derivative w.r.t. a parameter, it’s
used to compute derivatives for previous layers of the network (e.g.
just as dL/dC is used to compute dL/dB above).

Adding these up, we see that **during training, we have a total of 6NPM
FLOPs**, compared to 2NPM during inference: 2NPM in the forward pass,
4NPM in the backward pass. Since PM is the number of parameters in the
matrix, this is the simplest form of the famous \\6 \* \text{num
parameters} \* \text{num tokens}\\ approximation of Transformer FLOPs
during training: each token requires \\6 \* \text{num parameters}\\
FLOPs. We’ll show a more correct derivation below.

## Transformer Accounting

Transformers are the future. Well, they’re the present at least. Maybe a
few years ago, they were one of many architectures. But today, it’s
worth knowing pretty much every detail of the architecture. We won’t
reintroduce the architecture, but
<a href="https://jalammar.github.io/illustrated-transformer/"
rel="external nofollow noopener" target="_blank">this blog</a> and the
<a href="https://arxiv.org/abs/1706.03762"
rel="external nofollow noopener" target="_blank">original Transformer
paper</a> may be helpful references.

Here’s a basic diagram of the Transformer decoder architecture:

<figure>
<img src="/scaling-book/assets/img/transformer-diagram.png"
class="img-fluid" style="width:100.0%" loading="lazy"
onerror="this.onerror=null; $(&#39;.responsive-img-srcset&#39;).remove();" />
<figcaption><strong>Figure:</strong> this diagram shows one layer of a
standard Transformer and flows from top-to-bottom. We use a
single-letter convention to describe the shapes and layouts of arrays in
a Transformer, again showing contracting dimensions in red, and batched
dimensions in blue. In a given operation, the input shape is given on
top-left and the parameter shape is given on the top-right, with the
resulting shape below, e.g. BTD is the input shape for the gating einsum
and DF is the weight shape.</figcaption>
</figure>

**Note \[gating einsum\]**: The diagram above uses a
“<a href="https://arxiv.org/abs/2002.05202"
rel="external nofollow noopener" target="_blank">gating einsum</a>“
where we split the up-projection matrix into two matrices
(\$W\_\text{In1}\$ and \$W\_\text{In2}\$ above) whose outputs are
elementwise multiplied as a kind of “gating function”. Not all LLMs use
this, so you will sometimes see a single \$W\_\text{In}\$ matrix and a
total MLP parameter count of 2DF instead of 3DF. Typically in this case,
D and F will be scaled up to keep the parameter count the same as the 3
matrix case. With that said, some form of gating einsum is used by
LLaMA, DeepSeek, and many other models.

**Note 2 \[MHA attention\]**: With self-attention, T and S are the same
but for cross-attention they may be different. With vanilla Multi-Head
Attention (MHA), N and K are the same while for
<a href="https://arxiv.org/abs/1911.02150"
rel="external nofollow noopener" target="_blank">Multi-Query
Attention</a> (MQA) K=1 and for
<a href="https://arxiv.org/abs/2305.13245"
rel="external nofollow noopener" target="_blank">Grouped MQA</a> (GMQA),
K merely has to divide N.

**Note 3 \[pre-norm vs. post-norm\]:** The above diagram shows what is
known as a “pre-norm” architecture in which the norm occurs before the
residual connection, usually as `x + attn(norm(x))`. Models like LLaMA-3
use this today. The original Transformer paper used a “post-norm”
architecture in which the layernorm occurs after the residual
connection, i.e. `norm(x + attn(x))`.

## Global FLOPs and Params Calculation

Let’s calculate the per-layer FLOPs of a Transformer (so we can avoid
having to stick factors of **L** everywhere). Note that the training
FLOPs below are almost always 3x the inference FLOPs, so you can divide
any total by 3 to get the cost of just the forward pass.

### MLPs

The MLPs of a Transformer typically consist of 2 input matmuls that are
element-wise combined and a single output matmul:

\\\begin{array}{ccc} \textrm{operation} & \textrm{train FLOPs} &
\textrm{params} \\ \hline \\ A\[B,T,\red{D}\] \cdot W\_{in1}\[\red{D},
F\] & 6BTDF & DF \\\[10pt\] A\[B,T,\red{D}\] \cdot W\_{in2}\[\red{D},
F\] & 6BTDF & DF \\\[10pt\] \sigma\left(A\_{in1}\right)\[B,T, F\] \*
A\_{in2}\[B,T, F\] & \gray{O(BTF)} \\\[10pt\] A\[B,T,\red{F}\] \cdot
W\_{out}\[\red{F}, D\] & 6BTDF & DF \\\[10pt\] \hline \\ & \approx
18BTDF & 3DF \end{array}\\

### Attention

For the generic grouped-query attention case with different **Q** and
**KV** head numbers, let us assume equal head dimension H for
**Q**,**K**,**V** projections, and estimate the cost of the **QKVO**
matmuls:

\\\begin{array}{ccc} \textrm{operation} & \textrm{train FLOPs} &
\textrm{params} \\ \hline \\ A\[B,T,\red{D}\] \cdot W\_{Q}\[\red{D}, N,
H\] & 6BTDNH & DNH \\\[10pt\] A\[B,T,\red{D}\] \cdot W\_{K}\[\red{D}, K,
H\] & 6BTDKH & DKH \\\[10pt\] A\[B,T,\red{D}\] \cdot W\_{V}\[\red{D}, K,
H\] & 6BTDKH & DKH \\\[10pt\] A\[B,T,\red{N}, \red{H}\] \cdot
W\_{O}\[\red{N}, \red{H}, D\] & 6BTDNH & DNH \\\[10pt\] \hline \\ &
12BTD(N+K)H & 2D(N+K)H \end{array}\\

The dot-product attention operation is more subtle, effectively being a
\\TH \cdot HS\\ matmul batched over the \\B\\, \\K\\ dimensions, a
softmax, and a \\TS \cdot SH\\ matmul again batched over the \\B\\,
\\K\\ dimensions. We highlight the batched dims in blue:

\\\begin{array}{cc} \textrm{operation} & \textrm{train FLOPs} \\ \hline
\\\[3pt\] Q\[\blue{B}, T, \blue{K}, G, \red{H}\] \cdot K\[\blue{B}, S,
\blue{K}, \red{H}\] & 6BTSKGH = 6BTSNH \\\[3pt\] \textrm{softmax}\_S
\\\\ L\[B, T, S, K, G\] & \gray{O(BTSKG) = O(BTSN)} \\\[3pt\]
S\[\blue{B}, T, \red{S}, \blue{K}, G\] \cdot V\[\blue{B}, \red{S},
\blue{K}, H\] & 6BTSKGH = 6BTSNH \\\[3pt\] \hline \\ & \approx 12BTSNH =
12BT^2NH \\ \end{array}\\

**Note \[causal masking\]**: Most recent transformers use a causal mask
as opposed to full bidirectional attention. In this case the useful
FLOPs of the dot product operations are reduced by half. To achieve this
reduction in practice we need to make use of an attention kernel, rather
than a naive einsum.

### Other Operations

There are several other operations happening in a Transformer.
Layernorms are comparatively cheap and can be ignored for first-order
cost estimates. Note that each layer typically has two of them (one
before attention and one before the MLP). There is also the final
enormous (though not per-layer) unembedding matrix multiply.

\\\begin{array}{ccc} \textsf{operation} & \textsf{train FLOPs} &
\textsf{params} \\ \hline \\ 2 \times \textrm{layernorm}\_D \\\\
A\[B,T,\red{D}\] & \gray{O\left(BTD\right)} & \gray{2D} \\\[10pt\]
A\[B,T,\red{D}\] \cdot W\_{unembed}\[\red{D}, V\] & 6BTDV & DV \\
\end{array}\\

### General rule of thumb for Transformer FLOPs

If we neglect the cost of dot-product attention (which is reasonable for
shorter-context training), then the total FLOPs across all layers is

\\\begin{align\*} (18BTDF + 12BTD(N+K)H)L = 6 \*BT \* (3DF + 2D(N+K)H)L
\\ = 6 \* \textrm{num tokens} \* \textrm{parameter count}
\end{align\*}\\

This leads to a famous rule of thumb for estimating dense Transformer
FLOP count, ignoring the attention FLOPs. (Unembedding is another simple
matmul with \$6BTDV\$ FLOPs and \$DV\$ params, and follows the same rule
of thumb.)

### Fractional cost of attention with context length

If we do account for dot-product attention above and assume \\F=4D\\,
\\D=NH\\ (as is typical) and \\N=K\\, the ratio of dot-product attention
FLOPs to all matmul FLOPs (including the attention projections) is:

\\\small{\frac{\textrm{attention FLOPs}}{\textrm{matmul FLOPs}} =
\frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4\*18 BTD^2 + 24
BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}\\

The upshot is that **dot-product attention FLOPs only become dominant
during training once T\>8D**. For D ~ 8k, this would be ~64K tokens.
This makes some sense, since it means as the MLP size increases, the
attention FLOPs become less critical. For large models, the quadratic
cost of attention is not actually a huge obstacle to longer-context
training. However, for smaller models, e.g. Gemma-27B with D=4608,
attention becomes dominant around 37k sequence lengths.Note that some
modern OSS models introduce local attention or other optimizations that
reduce the cost of attention and change this roofline. Flash Attention
also helps alleviate the cost of long-context, which we discuss briefly
[in Appendix A](#appendix-a-how-does-flash-attention-work).

## Miscellaneous Math

### Sparsity and Mixture-of-Experts

We’d be remiss not to briefly discuss Mixture of Experts (MoE) models,
which replace the single dense MLP blocks in a standard Transformer with
a set of independent MLPs that can be dynamically routed between. To a
first approximation, **an MoE is just a normal dense model with E MLP
blocks per layer**, instead of just one. Each token activates \$k\$ of
these experts, typically \$k \ll E\$. The ratio \$E / k\$ is called the
sparsity and is usually between 8 and 64 (e.g.
<a href="https://arxiv.org/pdf/2412.19437"
rel="external nofollow noopener" target="_blank">DeepSeek v3</a> has
effectively \$k=8\$, \$E=256\$). This increases the parameter count by
\$O(E)\$, while multiplying the total number of activated parameters per
token by \$k\$, compared with the dense version.

<figure>
<img src="/scaling-book/assets/img/moe.png" class="img-fluid img-small"
style="width:100.0%" loading="lazy"
onerror="this.onerror=null; $(&#39;.responsive-img-srcset&#39;).remove();" />
<figcaption><strong>Figure:</strong> an example MoE layer with $n$
experts. The gating expert routes each token to $k$ of them, and the
output of those $k$ MLPs get summed. Our parameter count is $n$ times
the size of each expert, but only $k$ are used for each token. <a
href="https://deepgram.com/learn/mixture-of-experts-ml-model-guide"
rel="external nofollow noopener"
target="_blank">Source</a>.</figcaption>
</figure>

Compared to a dense model, an MoE introduces new comms, primarily two
AllToAlls (one before and one after the MoE block) that route tokens to
the correct expert and bring them back to their home device.Technically,
this only happens if we are data or sequence sharded along the same axis
as our experts. However, as we saw in the previous section, the cost of
each AllToAll is only 1/4 that of a comparable AllGather along a single
axis (for a bidirectional ring).

### Gradient checkpointing

Backpropagation as an algorithm trades compute for memory. Instead of a
backward pass requiring \\O(n\_\text{layers}^2)\\ FLOPs, **it requires
\\O(n\_\text{layers})\\ memory**, saving all intermediate activations
generated during the forward pass. While this is better than quadratic
compute, it’s incredibly expensive memory-wise: a model with \\B \*
T=4M\\ (4M total tokens per batch), L=64, and D=8192 that avoids all
unnecessary backward pass compute would have to save roughly \\2 \* 20
\* B \* T \* D \* L = 84TB\\ of activations in bfloat16. The 20 comes
from (roughly) counting every intermediate node in the Transformer
diagram above, since e.g.

\\f(x) = \exp(g(x))\\ \\\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}\\

so to avoid recomputing we need to save \\g(x)\\ and \\\exp(g(x))\\ from
the forward pass. To avoid saving this much memory, we can choose to
only save some fraction of the intermediate activations. Here are a few
strategies we use.

- **Block remat**: only save the input to each layer. This is the most
  aggressive method we use and only saves 1 checkpoint per layer,
  meaning we’d only save 4.2TB in the example above. This forces us to
  repeat essentially all forward pass FLOPs in the backward pass,
  meaning we increase our FLOPs from \\6 \cdot \text{num params} \cdot
  \text{num tokens}\\ to roughly \\8 \cdot \text{num params} \cdot
  \text{num tokens}\\.
- **Big matmuls only:** another simple policy is to only save the
  outputs of large matmuls. This lets us avoid recomputing any large
  matmuls during the backward pass, but still makes us recompute other
  activation functions and parts of attention. This reduces the 20 per
  layer above to closer to 7 per layer.

This is by no means comprehensive. When using JAX, these are typically
controlled by `jax.remat`/`jax.checkpoint` (you can read more <a
href="https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html"
rel="external nofollow noopener" target="_blank">here</a>).

### Key-Value (KV) caching

As we’ll see in [Section 7](../inference), LLM inference has two key
parts, prefill and generation.

- **Prefill** processes a long prompt and saves its attention
  activations in a Key-Value Cache (KV Cache) for use in generation,
  specifically the key-value projections in the attention block.
- **Generation** batches several of these KV caches together and samples
  tokens from each of them.

Each KV cache is then effectively an array of size \$\[2, S, L, K, H\]\$
where the 2 accounts for the keys and values. This is quite large! The
total size of the Key-Value cache in int8 is \$2SLKH\$. For a moderately
sized model with 8k context length, 64 layers, and \$KH = NH = D =
8192\$, this is \$2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}\$. You
can see why we would want to use GMQA with \$K \ll N\$.

## What Should You Take Away from this Section?

- The overall parameters and FLOPs of a Transformer are fairly easy to
  calculate, and are summarized here, assuming MHA (with batch size B,
  vocab size V, a sequence of length T, D=d<sub>model</sub>, and
  F=d<sub>ff</sub>):

| Component     | Params per layer          | Training FLOPs per layer     |
|:--------------|:--------------------------|:-----------------------------|
| **MLP**       | 3DF                       | 18BTDF                       |
| **Attention** | 4DNH                      | 24BTDNH + 12BT<sup>2</sup>NH |
| **Other**     | 2D                        | BTD                          |
| **Vocab**     | DV (total, not per-layer) | 12BTDV                       |

- The parameter count of the MLP block dominates the total parameter
  count and the MLP block also dominates the FLOPs budget as long as the
  sequence length \$T \< 8D\$.
- The total FLOPs budget during training is well approximated by \\6
  \cdot \text{num_params} \cdot \text{num_tokens}\\ for reasonable
  context lengths.
- During inference, our KV caches are roughly \\2 \cdot S \cdot L \cdot
  K \cdot H\\ per cache (where K is the number of KV heads), although
  architectural modifications can often reduce this.

## A Few Problems to Work

**Question 1:** How many parameters does a model with \$D=4096\$, \$F=4
\cdot D\$, \$V=32,000\$, and \$L=64\$ have? What fraction of these are
attention parameters? How large are our KV caches per token? *You can
assume \$N\cdot H=D\$ and multi-head attention with int8 KVs.*

Click here for the answer.

1.  The total parameters is roughly \\L \cdot (3DF + 4DNH + 2D) + 2DV\\
    (counting the two layernorms per layer). For the given numbers, this
    is \\64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 2
    \cdot 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9\\, or 16B parameters.
2.  The ratio of attention parameters to total parameters in general is
    \\4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4\\. This means
    roughly 1/4 of the parameters are used in attention.
3.  Per token, our KV caches are \\2 \cdot L \cdot N \cdot H = 2 \cdot
    64 \cdot 4096\\ in int8, which is `512 KiB / token`.

**Question 2:** How many total FLOPs are required to perform
A\[B<sub>X</sub>, D<sub>Y</sub>\] \*<sub>D</sub> W\[D<sub>Y</sub>, F\]
on `{'X': 4, 'Y': 8, 'Z': 4}`? How many FLOPs are performed by each TPU?

Click here for the answer.

The total “theoretical” FLOPs of the operation is \\2 \cdot B \cdot D
\cdot F\\. However, because the computation isn’t sharded across the Z
dimension, we’re actually doing Z extra FLOPs, meaning \\2 \cdot B \cdot
D \cdot F \cdot Z\\ total FLOPs. Since the computation is sharded across
the other dimensions, the total per-device is roughly \\2 \cdot B \cdot
D \cdot F / (X \cdot Y)\\.

**Question 3:** How many FLOPs are involved in performing \$A\[I,J,K,L\]
\* B\[I,J,M,N,O\] \rightarrow C\[K,L,M,N,O\]\$?

Click here for the answer.

Following the rule above, we have I and J as contracting dimensions and
K, L, M, N, and O as non-contracting dimensions. We have no “batching
dimensions”, so this is just \\2 \cdot I \cdot J \cdot K \cdot L \cdot M
\cdot N \cdot O\\, the product of all the axes. If we had a shared axis,
it would only be counted once.

**Question 4:** What is the arithmetic intensity of self-attention
(ignoring the Q/K/V/O projections)? *Give the answer as a function of
the Q and KV lengths T and S.* At what context length is attention
FLOPs-bound? Given the HBM bandwidth of our TPUs, plot the effective
relative cost of attention to the FFW block as the context length grows.

Click here for the answer.

Self-attention requires loading the \\Q\\, \\K\\, and \\V\\ activations,
then computing \\\text{softmax}(Q \cdot K) \cdot V\\, then writing the
result back to HBM. This will be done with Flash Attention so there are
some caveats to this math, but basically in bf16 self-attention performs

\\\text{Q\[B,T,N,H\]} \rightarrow\_\text{reshape} \text{Q\[B, T, K, G,
H\]} \cdot \text{K\[B, S, K, H\]} \rightarrow \text{O\[B, T, S, K,
G\]}\\ \\U=\text{softmax}\_S(\text{O\[B, T, S, K, G\]})\\ \\\text{U\[B,
T, S, K, G\]} \cdot \text{V\[B, S, K, H\]} \rightarrow \text{X\[B, T, K,
G, H\]}\\

So our total bytes is \\2 \* \text{sizeof}(Q) + 2 \* \text{sizeof(K or
V)} = 4BTNH + 4BSKH = 4BHK \* (TG + S)\\, total FLOPs is \\4BTSNH +
O(BTSN)\\ and the arithmetic intensity is \\4BTSKGH / (4BHK \* (TG +
S))\\.

So basically, during prefill we have \\S=T\\ so we have an arithmetic
intensity of \\4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)\\.
During generation, \\T=1\\ so we have \\4BSKGH / (4BHK \cdot (G + S)) =
SG / (G + S) \rightarrow G\\ assuming \\S\\ is very large. Depending on
how you interpret the question, during prefill or training
self-attention is compute-bound at S=240 assuming no sequence sharding.
During generation, we are never compute-bound because \\G\\ is small.
Nonetheless, you can see that increasing \\G\\ leads to us being closer
to compute-bound.

**Question 5:** At what sequence length are self-attention FLOPs equal
to the QKVO projection FLOPs?

Click here for the answer.

This is purely a question of when \\24BTDNH = 12BT^2NH\\. Simplifying we
get \\2D = T\\, so e.g. for \\D=4096\\, this is \\8192\\. This tells us
that for most reasonable context lengths, matmul FLOPs are greater.

**Question 6:** Say we only save the output of each of the 7 main
matmuls in a Transformer layer during our forward pass (Q, K, V, O + the
three FFW matrices). How many extra FLOPs do we need to “rematerialize”
during the backward pass?

Click here for the answer.

Saving only the seven matmul outputs (Q, K, V, O, W₁, W₂, W₃) means the
backward pass must recompute the two attention matmuls

\\QK^{\top} \quad\text{and}\quad \operatorname{softmax}(QK^{\top})V\\

in order to obtain \$\frac{\partial L}{\partial W\_\text{O}}\$.

Each is a \$T \times T\$ matmul batched over \$B\$ sequences and \$N\$
heads, so the additional FLOPs are

\\4 \\ B \\ T^{2} \\ N \\ H.\\

Other recomputed operations are:

1.  \$O(BTD)\$ for \$\frac{\partial L}{\partial W\_\text{In1}}\$ and
    \$\frac{\partial L}{\partial W\_\text{In2}}\$.
2.  And \$O(BTF)\$ for \$\frac{\partial L}{\partial W\_\text{Out}}\$.

**Question 7:** DeepSeek v3 says it was trained for 2.79M H800 hours on
14.8T tokens (<a href="https://arxiv.org/pdf/2412.19437v1"
rel="external nofollow noopener" target="_blank">source</a>). Given that
it has 37B activated parameters, roughly what hardware utilization did
they achieve? *Hint: note that they used FP8 FLOPs without structured
sparsity.*

Click here for the answer.

From the spec sheet <a href="https://lenovopress.lenovo.com/lp1814.pdf"
rel="external nofollow noopener" target="_blank">here</a>, we find 3,026
TFLOPs/s of FP8 performance with sparsity, or typically half this
(`1.513e15` FLOPs/s) without sparsity. 2.79M H800 hours means
`2.79e6 * 1.513e15 * 60 * 60 = 1.52e25` total FLOPs. Given the activated
parameter count of 37B, this training run should have used about
`6 * 37e9 * 14.8e12 = 3.3e24` FLOPs. That means the FLOPs utilization is
about `3.3e24 / 1.52e25 = 21.7%`.

**Question 8:** Mixture of Experts (MoE) models have \$E\$ copies of a
standard dense MLP block, and each token activates \$k\$ of these
experts. What batch size in tokens is required to be compute-bound for
an MoE with weights in int8 on TPU v5e? For DeepSeek, which has 256
(routed) experts and \$k=8\$, what is this number?

Click here for the answer.

Because we have \$E\$ copies of each expert, in int8, for each weight
matrix we need to load \$E \cdot D \cdot F\$ bytes. Because each token
activates \$k\$ experts, for each weight matrix we have \$2\cdot k \cdot
B \cdot D \cdot F\$ FLOPs. To be compute-bound with int8 weights and
bfloat16 FLOPs, we need the arithmetic intensity (FLOPs per byte loaded)
to exceed the TPU’s ~240 FLOPs/byte, which happens when \$(2\cdot k
\cdot BDF) / EDF \> 240\$ or \$k \cdot B / E \> 120\$.

Therefore, we need \$B \> 120 \cdot E / k\$ to be compute-bound. For
DeepSeek, this gives us \$B \> 120 \cdot 256 / 8 = 3840\$. This is a
remarkably large batch size at generation time.

### That’s it for Part 4! For Part 5 (about scaling Transformer training), [click here](../training)!

## Appendix

### Appendix A: How does Flash Attention work?

The traditional objection to scaling Transformers to very long context
is that the attention FLOPs and memory usage scale quadratically with
context length. While it’s true that the attention QK product has shape
\$\[B, T, S, N\]\$ where B is the batch size, T and S are the Q and K
sequence dims, and N is the number of heads, this claim comes with some
serious caveats:

1.  As we noted earlier, even though this is quadratic, the attention
    FLOPs only dominate when \\T \> 8 \cdot D\\, and during training the
    memory of a single attention matrix is small compared to all of the
    weights and activation checkpoints living in memory, especially when
    sharded.
2.  We don’t need to materialize the full attention matrix in order to
    compute attention! We can compute local sums and maxes and avoid
    ever materializing more than a small chunk of the array. While the
    total FLOPs is still quadratic, we drastically reduce memory
    pressure.

This second observation was first made by
<a href="https://arxiv.org/abs/2112.05682"
rel="external nofollow noopener" target="_blank">Rabe et al. 2021</a>
and later in the <a href="https://arxiv.org/abs/2205.14135"
rel="external nofollow noopener" target="_blank">Flash Attention
paper</a> (Dao et al. 2022). The basic idea is to compute the attention
in chunks of K/V, where we compute the local softmax and some auxiliary
statistics, then pass them on to the next chunk which combines them with
its local chunk. Specifically, we compute

1.  **M:** The running max of \\q \cdot k\\ over the sequence dimension
2.  **O:** The running full attention softmax over the sequence
    dimension
3.  **L:** The running denominator \\\sum_i \exp(q \cdot k_i -
    \text{running max})\\

With these, we can compute the new max, the new running sum, and the new
output with only a constant amount of memory. To give a sketchy
description of how this works, attention is roughly this operation:

\\\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot
K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}\\

The max is subtracted for numerical stability and can be subtracted
without affecting the outcome since \\\sum_i \exp(a_i + b) = \exp(b)
\sum \exp(a)\\. Looking just at the denominator above, if we imagine
having two contiguous chunks of key vectors, \\K^1\\ and \\K^2\\ and we
compute the local softmax sums \\L^1\\ and \\L^2\\ for each

\\L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)\\ \\L^2 =
\sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)\\

Then we can combine these into the full softmax sum for these two chunks
together by using

\\L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 -
\max(M^1, M^2)) \cdot L^2\\

where

\\M^1 = \max_j Q \cdot K_j^1 \text{ and } M^2 = \max_j Q \cdot K_j^2\\

This can be done for the full softmax as well, giving us a way of
accumulating arbitrarily large softmax sums. Here’s the full algorithm
from the Flash Attention paper.

<figure>
<img src="/scaling-book/assets/img/flash-algo.png" class="img-fluid"
style="width:100.0%" loading="lazy"
onerror="this.onerror=null; $(&#39;.responsive-img-srcset&#39;).remove();" />
</figure>

From a hardware standpoint, this lets us fit our chunk of Q into VMEM
(what the algorithm above calls on-chip SRAM) so we only have to load
the KV chunks on each iteration, increasing the arithmetic intensity. We
can also keep the running statistics in VMEM.

One last subtle point worth emphasizing is an attention softmax property
that’s used to make the Flash VJP (reverse mode derivative) calculation
practical for training. We define an intermediate softmax array:

\\S\_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_l e^{\tau q_i \cdot
k_l}}\\

In attention, we obtain *dS* from reverse-mode *dO* and *V* arrays:

\\dS\_{ij} = dO\_{id} \cdot_d V\_{jd} = \sum_d dO\_{id} V\_{jd}\\

During the backpropagation of this gradient to Q and K

\\d(q_i \cdot k_j) = (dS\_{ij} - S\_{ij} \cdot_j dS\_{ij}) S\_{ij}\\

We exploit an identity that allows us to exchange a contraction along
the large key **length** dimension with a local contraction along the
feature **depth** dimension.

\\\begin{align\*} S\_{ij} \cdot_j dS\_{ij} &= \sum_j \frac{e^{\tau q_i
\cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO\_{id} V\_{jd} \\ &=
\sum_d dO\_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i
\cdot k_k}} V\_{jd} \\ &= \sum_d dO\_{id} O\_{id} \\ &= dO\_{id} \cdot_d
O\_{id} \end{align\*}\\

This replacement is crucial for being able to implement a sequence-block
*local* calculation for the VJP, and enables further clever sharding
schemes like ring attention.

### Miscellaneous

<sup>\*</sup>Work done at Google DeepMind, now at MatX.

### Citation

For attribution in academic contexts, please cite this work as:

``` highlight
    Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
```

or as a BibTeX entry:

``` highlight
    @article{scaling-book,
      title = {How to Scale Your Model},
      author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad
      and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
      publisher = {Google DeepMind},
      howpublished = {Online},
      note = {Retrieved from https://jax-ml.github.io/scaling-book/},
      year = {2025}
    }
```
