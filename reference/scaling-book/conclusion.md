> Source: https://jax-ml.github.io/scaling-book/conclusion/ — Austin et al., "How To Scale Your Model", Google DeepMind, 2025

---

# Conclusions and Further Reading

Part 11 of [How To Scale Your Model](/scaling-book) ([Part 10:
JAX](../jax-stuff) \| [Part 12: GPUs](../gpus))

Thank you for reading! Here we'll include a few more references for
further study.

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

[Acknowledgments](#acknowledgments)

[Further Reading](#further-reading)

[Feedback](#feedback)

**Thank you for reading the whole thing and congratulations on making it
all the way to the end.** Before we conclude, a few acknowledgments:

## Acknowledgments

This document represents a significant collective investment from many
people at Google DeepMind, who we’d like to briefly acknowledge!

- James Bradbury, Reiner Pope, Noam Shazeer, and Blake Hechtman
  originally derived many of the ideas in this manuscript, and were
  early to understand the systems view of the Transformer.
- Sholto Douglas wrote the first version of this doc and is responsible
  for kicking off the project. He is more than anyone responsible for
  the overall narrative of this doc.
- Jacob Austin led the work of transforming this first version from
  rough notes into a more polished and comprehensive artifact. He did
  much of the work of editing, formatting, and releasing this document,
  and coordinated contributions from other authors.
- Most of the figures and animations were made by Anselm Levskaya and
  Charlie Chen.
- Charlie Chen wrote the inference section and drew many of the
  inference figures.
- Roy Frostig helped with publication, editing, and many other steps of
  the journey.

We’d also like to thank many others who gave critical feedback
throughout the process, in particular Zak Stone, Nikhil Sethi, Caitlin
Stanton, Alek Dimitriev, Sridhar Lakshmanamurthy, Albert Magyar, Diwakar
Gupta, Jeff Dean, Corry Wang, Matt Johnson, Peter Hawkins, and many
others. Thanks to Ruiqi Gao for help with the HTML formatting.

**Thank you all!**

Before you go, you might also enjoy reading the new [Part 12](../gpus)
on NVIDIA GPUs!

## Further Reading

There is a bunch of related writing, including the following:

- <a href="https://henryhmko.github.io/posts/tpu/tpu.html"
  rel="external nofollow noopener" target="_blank"><strong>TPU Deep
  Dive</strong></a>: a wonderful in-depth look at the TPU architecture
  in the spirit of this book.
- <a href="https://fleetwood.dev/posts/domain-specific-architectures"
  rel="external nofollow noopener" target="_blank"><strong>Domain specific
  architectures for AI inference</strong></a>: a hardware and model deep
  dive in the spirit of this book.
- <a href="https://dl.acm.org/doi/pdf/10.1145/3360307"
  rel="external nofollow noopener" target="_blank"><strong>A
  Domain-Specific Supercomputer for Training Deep Neural
  Networks</strong></a>: one of the OG TPU papers, this has a lot of
  great details about the Google TPU program not covered here.
- <a href="https://horace.io/brrr_intro.html"
  rel="external nofollow noopener" target="_blank"><strong>Making Deep
  Learning Go Brrrr From First Principles</strong></a>: a more GPU and
  PyTorch-focused tutorial on LLM rooflines and performance engineering.
- <a href="https://jax.readthedocs.io/en/latest/pallas/tpu/details.html"
  rel="external nofollow noopener" target="_blank"><strong>Writing TPU
  Kernels with Pallas</strong></a>: increasingly, TPU programming
  involves writing custom kernels in Pallas. This series discusses how
  to write kernels and many lower level TPU details that aren’t
  mentioned here.
- <a href="https://siboehm.com/articles/22/CUDA-MMM"
  rel="external nofollow noopener" target="_blank"><strong>How to Optimize
  a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</strong></a>:
  while GPU and CUDA specific, this is an excellent blog post showing
  how to optimize a matmul kernel in CUDA. This might be a good deep
  dive into how TPUs and GPUs are different.
- <a
  href="https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html"
  rel="external nofollow noopener" target="_blank"><strong>Distributed
  arrays and automatic parallelization</strong></a>: this is a really
  nice guide to parallelism APIs in JAX and is a good way to learn how
  to actually implement some of the ideas we’ve discussed here.
- <a href="https://github.com/rwitten/HighPerfLLMs2024"
  rel="external nofollow noopener" target="_blank"><strong>Rafi Witten’s
  High Performance LLMs 2024 Class</strong></a>: our former colleague
  Rafi gave a great course on TPU performance engineering and the slides
  are all on GitHub. This covers a bunch of things in more depth than we
  do here.
- <a href="https://arxiv.org/abs/2211.05102"
  rel="external nofollow noopener" target="_blank"><strong>[2211.05102]
  Efficiently Scaling Transformer Inference</strong></a>: a detailed
  paper on the mathematics of Transformer inference. This is the
  inspiration for a lot of this document.
- <a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook"
  rel="external nofollow noopener" target="_blank"><strong>Huggingface
  Ultra-Scale Playbook</strong></a>: something of a GPU analog to this
  book, this talks more at depth about how PyTorch implements
  parallelism techniques and memory-saving techniques during training.
- <a href="https://kipp.ly/transformer-inference-arithmetic/"
  rel="external nofollow noopener" target="_blank"><strong>Transformer
  Inference Arithmetic</strong></a>: a blog with many of the same ideas
  as this book and some excellent illustrations.
- <a
  href="https://stanford-cs336.github.io/spring2025/index.html#coursework"
  rel="external nofollow noopener" target="_blank"><strong>Stanford CS336
  Slides and Videos</strong></a>: a fantastic Stanford course covering
  many details of LLM training and serving, with some useful exercises.
  Assignments 1 and 2 are particularly relevant.
- <a href="https://github.com/stas00/ml-engineering"
  rel="external nofollow noopener" target="_blank"><strong>Stas Bekman’s
  ML Engineering Handbook</strong></a>: a highly practical guide to ML
  infrastructure, covering topics not addressed in this book like how to
  negotiate with cloud providers, cluster management, and empirical
  measurements of GPU throughput.
- <a
  href="https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/"
  rel="external nofollow noopener" target="_blank"><strong>ezyang’s
  blog</strong></a>: a PyTorch lead’s blog on all things sharding +
  PyTorch, including a
  <a href="https://blog.ezyang.com/2019/05/pytorch-internals/"
  rel="external nofollow noopener" target="_blank">guide to PyTorch
  internals</a> and a <a
  href="https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/"
  rel="external nofollow noopener" target="_blank">writeup of sharded
  matrix multiplication</a>. Lots of other good things here.
- <a href="https://www.aleksagordic.com/blog/collective-operations"
  rel="external nofollow noopener" target="_blank"><strong>The Anatomy of
  Collective Communication</strong></a>: a nice walkthrough of GPU and
  TPU collectives in the spirit of this book. Has a better writeup of
  N-D and GPU collectives than this book.

There remains a lot of room for comprehensive writing in this area, so
we hope this manuscript encourages more of it! We also believe that this
is a fruitful area to study and research. In many cases, it can be done
even without having many hardware accelerators on hand.

## Feedback

Please leave comments or questions so that we can improve this further.
You can reach our corresponding author, Jacob Austin, at jacobaustin123
\[at\] gmail \[dot\] com, or suggest edits by posting issues, pull
requests, or discussions
<a href="https://github.com/jax-ml/scaling-book"
rel="external nofollow noopener" target="_blank">on GitHub</a>.

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
