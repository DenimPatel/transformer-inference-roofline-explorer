> Source: https://jax-ml.github.io/scaling-book/jax-stuff/ — Austin et al., "How To Scale Your Model", Google DeepMind, 2025

---

# Programming TPUs in JAX

Part 10 of [How To Scale Your Model](/scaling-book) ([Part 9:
Profiling](../profiling) \| [Part 11: Conclusions](../conclusion))

How to use JAX to program TPUs efficiently! Much of this section is
taken from
<a href="https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html"
rel="external nofollow noopener" target="_blank">here</a>. You can run
the code examples in this section with free TPUs on
<a href="https://colab.sandbox.google.com/"
rel="external nofollow noopener" target="_blank">Google Colab</a>.

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

<a href="https://x.com/yashk2810" class="name"
rel="external nofollow noopener" target="_blank">Yash Katariya</a>

<span class="affiliation"></span>

<a href="https://x.com/reinerpope" class="name"
rel="external nofollow noopener" target="_blank">Reiner
Pope<sup>*</sup></a>

<span class="affiliation"></span>

### Published

Feb. 4, 2025

### Contents

[How Does Parallelism Work in JAX?](#how-does-parallelism-work-in-jax)

[](#)

- [Auto sharding mode](#auto-sharding-mode)
- [Explicit sharding mode](#explicit-sharding-mode)
- [Manual sharding mode via
  shard_map](#manual-sharding-mode-via-shard-map)

[Worked Problems](#worked-problems)

## How Does Parallelism Work in JAX?

JAX supports three schools of thought for multi-device programming:

1.  **Compiler, take the wheel!** Let the XLA compiler automatically
    partition arrays and decide what communication to add to facilitate
    a given program. This lets you take a program that runs on a single
    device and automatically run it on thousands without changing
    anything.
2.  **JAX, take the wheel!** Automatic parallelism is great, but
    sometimes the compiler does something crazy. Explicit sharding lets
    you write single-device code like usual, but have JAX handle
    sharding propagation (not the compiler). This means JAX can ask you
    for clarification when it’s unclear what you want.
3.  **Just let me write what I mean, dammit!** While compilers are nice,
    they sometimes do the wrong thing and add communication you don’t
    intend. Sometimes we want to be explicit about exactly what
    communication we intend to run.

|   Mode   |   View?    | Explicit sharding? | Explicit Collectives? |
|:--------:|:----------:|:------------------:|:---------------------:|
|   Auto   |   Global   |         ❌         |          ❌           |
| Explicit |   Global   |         ✅         |          ❌           |
|  Manual  | Per-device |         ✅         |          ✅           |

Correspondingly, JAX provides APIs for each of these modes:

1.  `jax.jit` (with `Auto` mesh axes) lets you take any existing JAX
    function and call it with sharded inputs. JAX then uses XLA’s
    <a href="https://openxla.org/shardy" rel="external nofollow noopener"
    target="_blank">Shardy</a> compiler which automatically parallelizes
    the program. XLA will add communication for you (AllGathers,
    ReduceScatters, AllReduces, etc.) when needed to facilitate existing
    operations. While it isn’t perfect, it usually does a decent job at
    automatically scaling your program to any number of chips without
    code changes.
2.  `jax.jit` with `Explicit` mesh axes looks similar to (1), but lets
    JAX handle the sharding propagation instead of XLA. That means the
    sharding of an array is actually part of the JAX type system, and
    JAX can error out when it detects ambiguous communication and lets
    the user resolve it.
3.  `jax.shard_map` is the more manual counterpart. You get a
    device-local view of the program and have to write any communication
    you want explicitly. Have a sharded array and want the whole thing
    on each device? Add a `jax.lax.all_gather`. Want to sum an array
    across your devices? Add a `jax.lax.psum` (an AllReduce).
    Programming is harder but far less likely to do something you don’t
    want.

### Auto sharding mode

`jax.jit` plays two roles inside JAX. As the name suggests, it
“just-in-time” compiles a function from Python into bytecode (via
XLA/HLO/LLO) so it runs faster. But if the input is sharded or the user
specifies an `in_sharding` or `out_sharding`, it also lets XLA
distribute the computation across multiple devices and add communication
as needed. For example, here’s how you could write a sharded matmul
using `jax.jit`:

``` highlight
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

# This creates a fake set of 8 CPU devices so you can run this on a CPU without TPUs.
jax.config.update("jax_num_cpu_devices", 8)

# This creates a 2D 4x2 mesh with axis names X and Y that JAX uses by default.
# We explicitly tell JAX to let the XLA compiler infer sharding along these axes.
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'), axis_types=(Auto, Auto))
jax.set_mesh(mesh)

# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# We can explicitly compile the sharded matmul function here. This adds all the
# necessary comms (e.g. an AllReduce after the matmul).
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

This will run automatically with any sharding and partition the
computation across our devices. **But what’s actually happening at the
hardware level?**

1.  First we create In and W sharded across our devicesNotice how we did
    this. This is one way to create an array with a particular sharding
    (i.e. by adding the device argument to the creation function).
    Another one is to create an array normally with \`jnp.array(....)\`
    and then do e.g. \`jax.device_put(..., jax.P('X', 'Y'))\`. Yet
    another is to write a function which creates the array you want, and
    jit-compile it with \`out_shardings\` being what you want.. W is
    sharded 2-way along the contracting dimension, while In is sharded
    8-way: 4-way along the input dimension and 2-way along the
    contracting dimension. This corresponds to a sharding
    W\[D<sub>Y</sub>, F\] and In\[B<sub>X</sub>, D<sub>Y</sub>\], aka a
    kind of model and data parallelism.
2.  If we were running this locally (i.e. on one device),
    `matmul_square` would simply square the input and perform a simple
    matmul. But because we specify the `out_shardings` as
    `P('X', None)`, the output will be sharded along the batch but
    replicated across the model dimension and will require an AllReduce
    to compute.

Using our notation from previous sections, this will likely do something
like

1.  Out\[B<sub>X</sub>, F\] { U<sub>Y</sub> } = In\[B<sub>X</sub>,
    D<sub>Y</sub>\] \*<sub>D</sub> W\[D<sub>Y</sub>, F\]
2.  Out\[B<sub>X</sub>, F\] = **AllReduce**(Out\[B<sub>X</sub>, F\] {
    U<sub>Y</sub> })

`jax.jit` will add this for us automatically! We can actually print the
HLO with `jit_matmul.as_text()` and see the following HLO (abbreviated
dramatically):

``` highlight
# This fusion is the actual matmul of the sharded inputs and matrix
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# We reduce the partially summed results across devices
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

We can see the matmul (the fusion) and the AllReduce above. Pay
particular attention to the shapes. `bf16[2, 1024]` is a local view of
the activations, since our `batch_size=8` is split across 4 devices and
our `d_model=2048` is likewise split 2 ways.

**This is pretty magical!** No matter how complicated our program is,
<a href="https://openxla.org/shardy" rel="external nofollow noopener"
target="_blank">Shardy</a> and jit will attempt to find shardings for
all the intermediate activations and add communication as needed. With
that said, Shardy has its flaws. It can make mistakes. Sometimes you’ll
look at a profile and notice something has gone wrong. A giant AllGather
takes up 80% of the profile, where it doesn’t need to. When this
happens, we can try to correct the compiler by explicitly annotating
intermediate tensors with `jax.lax.with_sharding_constraint`. For
instance, with two matmuls I can force the intermediate activations to
be sharded along the `y` dimension (not that this is a good idea) with
the following:

``` highlight
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X', 'Y'), (Auto, Auto))
jax.set_mesh(mesh)

def matmul(x, W_in, W_out):
  hidden = jnp.einsum('bd,df->bf', x, W_in)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('X', 'Y'))
  return jnp.einsum('bf,df->bd', hidden, W_out)
```

This makes up about 60% of JAX parallel programming in the automatic
partitioning world where you control the intermediate shardings via
`jax.lax.with_sharding_constraint`. But “compiler tickling” is famously
not a fun programming model. You could annotate every intermediate
variable and still not know if you’ll get the right outcome. Instead,
what if JAX itself could handle and control sharding propagation?

### Explicit sharding mode

Explicit sharding (or “sharding in types”) looks a lot like automatic
sharding, but sharding propagation happens at the JAX level! Each JAX
operation has a sharding rule that takes the shardings of the op’s
arguments and produces a sharding for the op’s result. You can see the
resulting sharding using `jax.typeof`:

``` highlight
import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# Running on a TPU v5e 2x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))

# This tells JAX to use this mesh for all operations, so you can just specify the PartitionSpec P.
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16, dtype=np.float32).reshape(8, 2), jax.P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # float32[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # float32[8@X,2@Y]
  return out

f(x)
```

As you can see, JAX propagated the sharding from input (`x`) to output
(`out`) which are inspectable at trace-time via `jax.typeof`. For most
operations these rules are simple and obvious because there’s only one
reasonable choice (e.g. elementwise ops retain the same sharding). But
for some operations it’s ambiguous how to shard the result, in which
case JAX throws a trace-time error and asks the programmer to provide an
`out_sharding` argument explicitly (e.g. jnp.einsum, jnp.reshape, etc).
Let’s see another example where you have conflicts:

``` highlight
# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # This will error
```

This code errors with:

``` highlight
Contracting dimensions are sharded and it is ambiguous how the output should be sharded.
Please specify the output sharding via the `out_sharding` parameter.
Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)
```

This is awesome because how the output of einsum should be sharded is
ambiguous. The output sharding can be:

- P(‘X’, ‘Y’) which will induce a ReduceScatter or
- P(‘X’, None) which will induce an AllReduce

Unlike Auto mode, explicit mode errors out when it detects ambiguous
communication and requires the user to resolve it. So here you can do:

``` highlight
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=jax.P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

Auto mode and Explicit mode can be composed via `jax.sharding.auto_axes`
and `jax.sharding.explicit_axes` APIs. This is a <a
href="https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html"
rel="external nofollow noopener" target="_blank">great doc to read</a>
for more information.

### Manual sharding mode via shard_map

While Shardy is the “compiler take the wheel” mode, jax
<a href="https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html"
rel="external nofollow noopener" target="_blank">shard_map</a> puts
everything in your hands. You specify the sharding of the inputs, like
in jax.jit, but then you write all communication explicitly. Whereas
`jax.jit` leaves you with a global cross-device view of the program,
`shard_map` gives you a local per-device view.

Here’s an example. Try to reason about what this function does:If you
want to play with this yourself in a colab by emulating a mesh, you can
do so using the following cell \`import jax;
jax.config.update('jax_num_cpu_devices', 8)\`

``` highlight
import jax
import jax.numpy as jnp

Explicit = jax.sharding.AxisType.Explicit

mesh = jax.make_mesh((2, 4), ('x', 'y'), (Explicit, Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=jax.P(('x', 'y')))

# This function will operate on 1/8th of the array.
@jax.shard_map(in_specs=jax.P(('x', 'y')), out_specs=jax.P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**What does this do?** `slice_and_average` is run on each TPU with 1/8th
of the array, from which we slice the first 4 elements and average them
across the full mesh. This means we’re effectively doing
`mean(x[:4], x[64:68], x[128:132], …)`. This is pretty cool, because
that’s not an easy operation to express in JAX otherwise.

**Why do this instead of jax.jit?** If we’d used `jax.jit`,
`slice_and_average` would have seen a global view of the array (the full
`[512,]` array). We’d have had to slice out this non-uniform slice and
then perform an average which XLA would have had to interpret correctly.
XLA might have added the wrong communication or gotten confused. Here we
see the local view and write only the communication we need.

**Example \[Collective Matmul\]:** To take a more realistic example, say
we want to implement model parallelism where the activations are
initially model sharded, i.e. A\[B<sub>X</sub>, D<sub>Y</sub>\]
\*<sub>D</sub> W\[D, F<sub>Y</sub>\] -\> Out\[B<sub>X</sub>,
F<sub>Y</sub>\]. Naively, we would do this by AllGathering A first
followed by a local matrix multiplication:

1.  A\[B<sub>X</sub>, D\] = **AllGather**<sub>Y</sub>(A\[B<sub>X</sub>,
    D<sub>Y</sub>\])
2.  Out\[B<sub>X</sub>, F<sub>Y</sub>\] = A\[B<sub>X</sub>, D\]
    \*<sub>D</sub> W\[D, F<sub>Y</sub>\]

Sadly, this is bad because it doesn’t allow us to overlap the
communication with the computation. Overlapping them can be done with a
“collective matmul”, as described in
<a href="https://dl.acm.org/doi/pdf/10.1145/3567955.3567959"
rel="external nofollow noopener" target="_blank">Wang et al. 2023</a>.
The algorithm is basically as follows:

- For each Y shard, perform a matmul of the local chunk of A with the
  local chunk of W, producing a result of shape `[B / X, F / Y]`.
  Simultaneously, permute A so you get the next chunk locally, perform
  the matmul, and sum the result.

We can implement that quite easily with `jax.shard_map`:

``` highlight
import functools

import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# This is intended to run on a TPU v5e-8 runtime. If you can't get this,
# try setting jax.config.update('jax_num_cpu_devices', 8).
#
mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs is the looped operand; rhs is the local operand
  axis_size = jax.lax.axis_size('Y')  # axis_size = 4 for this example
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # Matmul for a chunk
    update = lhs @ rhs_chunk
    # Circular shift to the left
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pcast(accum, ('X', 'Y'), to='varying')
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # Compute the last chunk after the final permute to leave lhs in the state we found it
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

This is pretty neat! We can benchmark this and see that it’s also a lot
faster!
<a href="https://imgur.com/a/e9I6SrM" rel="external nofollow noopener"
target="_blank">Here’s</a> the profile with the default jit matmul which
takes 311us with a big blocking AllGather at the beginning:

<figure>
<img src="/scaling-book/assets/img/not-overlapped.png" class="img-fluid"
style="width:100.0%" loading="lazy"
onerror="this.onerror=null; $(&#39;.responsive-img-srcset&#39;).remove();" />
</figure>

And
<a href="https://imgur.com/a/21iy0Sv" rel="external nofollow noopener"
target="_blank">here’s</a> the version above that takes 244us. You can
see the profile doesn’t have the AllGather. It’s all useful work! Our
FLOPs utilization is also a lot higher.

<figure>
<img src="/scaling-book/assets/img/overlapped.png" class="img-fluid"
style="width:100.0%" loading="lazy"
onerror="this.onerror=null; $(&#39;.responsive-img-srcset&#39;).remove();" />
</figure>

It’s also worth noting that the matmul time with no sharding on the
contracting dimension is
<a href="https://imgur.com/a/i3gNKfq" rel="external nofollow noopener"
target="_blank">224us</a>, so we’re remarkably close to the unsharded
baseline here. This is a good example of the kind of performance
engineering you might end up doing to improve TPU utilization. For more
`shard_map` examples, <a
href="https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side"
rel="external nofollow noopener" target="_blank">this note is great</a>.

Now here are a couple of useful worked problems to try and implement
using `jax.jit` or `shard_map`!

## Worked Problems

Here are some random JAX-related problems. I’ll add some more later. For
all of these, you’ll need some number of TPUs. Colab no longer hands out
TPU v2-8 slices, so use
<a href="https://www.kaggle.com/" rel="external nofollow noopener"
target="_blank">Kaggle</a> (which still provides them for free) or an
8-core GCP slice.If you just want to emulate a mesh on fake problems,
you can also fake 8 devices on CPU with \`import jax;
jax.config.update('jax_num_cpu_devices', 8)\` (requires jax \>=
0.4.27ish), though this won't reflect real performance. From now on,
we’ll assume you have N devices available.

**Question 1:** Let **A** be an array of activations of shape
float32\[S<sub>X</sub>, D<sub>Y</sub>\] with `X * Y = N`. Do the
following:

1.  Write a function in JAX that computes the average within each
    `(X, Y)` shard, i.e. it returns an array of size \[X, Y\] where
    `arr[i, j]` is the average over shard `(i, j)`. Do this with both
    `jax.jit` and `shard_map`. Profile each and see how long they took.
    Was there any communication added? *Hint: there shouldn’t be, but
    sometimes XLA adds it anyway.*

2.  Write a function in JAX that returns `roll(x, shift, axis=0) - x`
    for some shift **within each shard along X**. I’m not enough of a
    masochist to make you do this in jax.jit, so just do this with
    `shard_map`.

Click here for the answer.

Part 1: Here is a solution to part 1. Note the fairly complex reshapes
we have to do for the `jax.jit` solution.

``` highlight
import numpy as np

import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

Part 2: Here is a similar solution to Part 2.

``` highlight
import numpy as np

import jax
import jax.numpy as jnp

import functools

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0),
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

**Question 2:** Here we’ll make a basic “mixture of experts” model
together. Let **W**: float32\[E<sub>X</sub>, D, F\] be a set of E
“expert” matrices. Let **A**: float32\[S<sub>X</sub>, D\] (our
activations) and let **B**: int32\[S<sub>X</sub>\] be a set of “routing
assignments” where B\[i\] is an integer in the range `[0, E)` telling us
which matrix we want to process that activation. We want to write a
function in JAX that returns `Out[i] = A[i] @ W[B[i]]`.

1.  Let’s start by ignoring sharding altogether. Make all of these
    tensors small enough so they fit in one device. Write a local
    implementation of this function. *Make sure you don’t materialize an
    array of shape `[S, D, F]`! Hint: try sorting the tokens into a new
    buffer of shape `[E, S, D]` with some attention to masking (why do
    we need the second dimension to have size S?).*

2.  If you just `jax.jit` the above method, something will happen.
    Profile this and see what communication it decided to do. How long
    does it take?

3.  One problem you’ll notice with the above is that it likely gathers
    the full set of activations **A** locally, i.e.
    AllGather<sub>X</sub>(\[S<sub>X</sub>, D\]). Not only is this
    expensive communication-wise, it’s also incredibly expensive
    memory-wise if we can’t fit the full set of activations locally.
    Implement the above using `shard_map` and explicit communication.

    1.  For a first pass, it might be easiest to use a
        `jax.lax.all_gather` and reorder as in step 1.

    2.  For a second pass, try to avoid materializing any array of size
        `[E, S, D]`, i.e. try to perform the computation in a ragged
        fashion using a `jax.lax.all_to_all` inside a
        `jax.lax.while_loop`. This way, you can avoid materializing the
        full activations and wasting compute on padding. How much faster
        is this than your original implementation?

4.  Most MoEs route to multiple (k) experts and then average the result.
    Refactor the above to implement this. Let **B**:
    int32\[S<sub>X</sub>, k\] in this case for the k experts to route
    to.

Click here for the (partial) answer.

1/2. For part (1), you have a lot of choices. Here’s one option that
just iterates over the experts with masking.

``` highlight
def moe_local(W: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry  # [S, F]
        mask = (B == e)[:, None]  # [S, 1]
        expert_result = A @ W[e]  # [S, F] - this expert's transform of ALL tokens
        output = output + expert_result * mask  # Only keep results for assigned tokens
        return output, None

    output = jnp.zeros((S, F))
    output, _ = jax.lax.scan(expert_forward, output, jnp.arange(E))

    return output
```

You can also use `jax.lax.ragged_dot` which will do something similar
but more efficiently.

1.  I’m only going to sketch the pseudocode here (if you have a clean
    solution feel free to add it):

``` highlight
chunk_size = 128
def matmul(W, x, B):
  i = 0
  x = # sort x according to assignments
  while (chunk := x[i:i+chunk_size]).any():
     chunk = all_to_all(chunk)
     out = matmul_local(W, chunk)
     i += chunk_size
  return concat(out)
```

The basic idea is to iterate over chunks of the array, sort them and do
an all_to_all, then do the local FLOPs.

**Question 3:** The collective matmul example above is actually super
relevant for real LLMs. Let’s tweak the example to do the full
Transformer stack.

1.  As an exercise, let’s start by implementing an AllReduce collective
    matmul, i.e. A\[B<sub>X</sub>, D<sub>Y</sub>\] \*<sub>D</sub>
    W\[D<sub>Y</sub>, F\] -\> Out\[B<sub>X</sub>, F\]. Note that the
    output isn’t replicated. The naive algorithm is discussed above,
    basically just a local matmul followed by an AllReduce. Try to make
    a comms overlapped “collective” version of this operation. *Hint:
    tile over the output dimension and feel free to use `jax.lax.psum`
    (aka AllReduce).* *Note: due to the way XLA handles this, it may not
    actually be faster than the baseline.*

2.  The complement to the AllReduce collective matmul above is a
    ReduceScatter collective matmul, as in Tmp\[B<sub>X</sub>,
    F<sub>Y</sub>\] \*<sub>F</sub> W2\[F<sub>Y</sub>, D\] -\>
    Out\[B<sub>X</sub>, D<sub>Y</sub>\]. This occurs in the
    down-projection matrix in a Transformer. Implement a collective,
    overlapped version of this in JAX. Be careful about passing only the
    minimal amount of data you need. *Hint: try permuting the result as
    you accumulate it.*

3.  Put these two together into an end-to-end Transformer block that
    performs In\[B<sub>X</sub>, D<sub>Y</sub>\] \*<sub>D</sub>
    W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub>
    W<sub>out</sub>\[F<sub>Y</sub>, D\] -\> Out\[B<sub>X</sub>,
    D<sub>Y</sub>\] with overlapped communication.As before, we can't do
    \$W\_{in} \cdot W\_{out}\$ first because of a non-linearity we've
    omitted here. How much faster is this than a `jax.jit`
    implementation?

**Question 4:** All of the collective matmuls implemented above are
unidirectional: they only permute in one direction. Rewrite the
collective AllReduce matmul and the collective ReduceScatter matmuls to
use bidirectional communication. How much faster are these?

### That’s all for Part 10. That’s basically it! For final conclusions and further reading, click [here](../conclusion).

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
