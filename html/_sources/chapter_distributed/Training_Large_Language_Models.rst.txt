
Training Large Language Models
==============================

Fully Sharded Data Parallelism
------------------------------

Data parallelism is the most versatile strategy for parallelizing deep
learning models. However, vanilla data parallelism requires each device
to have sufficient memory for accommodating the whole model replica,
limiting its applicability in training models with billions of
parameters. For example, a 7.5B model already requires 120 GB of memory
for its model training states (parameters, gradients, and optimizer
states), which is much more than the 80GB of memory in the latest
generation of GPUs.

Fully sharded data parallelism (FSDP) is a memory-optimized variation of
data parallelism. FSDP shards the model training states across multiple
accelerators to reduce the memory consumption of each accelerator. As a
result, each accelerator holds a shard of training states of multiple
layers. In the forward pass, each accelerator needs to introduce an
all-gather operation to collect the parameters for the other shards. The
parameters from the other shards are discarded immediately once the
corresponding computation is finished. In the backward pass, it
re-collects parameters with all-gather first, then each accelerator can
compute local gradients. The gradients perform reduce-scatter operation
to aggregate and redistribute gradients across the accelerators.
Finally, each accelerator uses the sharded optimizer states and
gradients to update its parameter shard.

FSDP also comes with two reduced variants: the first variant shards only
the optimizer states while keeping the parameters and gradients as a
whole, and the second variant shards also the gradients in addition to
the optimizer states. These variants have less communication volume than
the full version but have more memory consumption.

One of the primary advantages of FDSP is the significant reduction in
memory requirements per device, enabling the training of much larger
models and batch sizes than vanilla data parallelism. Compared to tensor
model parallelism, it requires no code refactoring and has fewer
constraints to the layer type and model architecture so that it can be
simply applied to a wide range of models. In contrast to pipeline model
parallelism, it has no load imbalance issue since all devices are
executing the same set of layers.

On the other hand, FSDP has communication overheads due to the higher
communication volume from gathering parameters in forward and backward
phases and reducing gradients, especially in networks with limited
bandwidth.

Mixture of Experts
------------------

Mixture-of-Experts (MoE) is a machine learning technique that has been
employed in transformer models to scale model capacity while maintaining
constant computation per token during both training and inference.
Typically, an MoE layer replaces of the dense feed-forward layer of a
transformer model. This layer is composed of a group of trainable expert
networks, each of which is feed-forward network (FFN) in practice, along
with a gating network that selects a subset of experts for each input
token. MoE-based models have deomonstrated comparable accuracy as their
dense counterparts, with the same model size, while requiring less
computation. Notable examples of large-scale MoE-based transformer
models with more than 1 trillion parameters are Switch Transformer and
Gshard.

Despite MoE capability to scale model size significantly, the memory
requirement of MoE-based grows linearly with the model size, making it
challenging for a single device to handle. Expert Parallelism is a form
of parallelism to address this challenge by distributing subsets of
experts across multiple devices. During execution, the gating network
assigns tokens to experts and these tokens are dispatched to the devices
hosting their assigned experts uasing all-to-all communication. Once the
expert computations are complete, the tokens are sent back to their
source device via another round of all-to-all communication.

From a system perspective, the primary advantage of MoE is its ability
to improve computational efficiency and reduce memory consumption. By
increasing the number of experts in proportion to the number of devices,
the growth of computation and memory requirement per device can be kept
nearly constant. However, the major drawback of expert parallelism is
the communication overhead due to the two additional rounds of
all-to-all communications per MoE layer. This overhead can become a
significant bottleneck, especially when communication spans multiple
nodes, leading to decreased efficiency. The other drawback comes from
the imbalance of expert workloads. The tokens are assigned to experts
based on the indeterministic result of the router network, although this
imbalance issue can be mitigated with the introduction of auxiliary load
balancing loss to encourage a balanced load across experts. However, the
load imbalance can still be a significant problem, especially during the
early stage of training or working with a small batch size.

Activation Recomputation
------------------------

During the backward pass, the layers need to access the activations
generated during the forward pass to compute the gradients. These
activations are stored in memory, and for very deep models, the
accumulating memory used for the saved activations can consume a
considerable amount of the accelerator’s memory. For example, each layer
of GPT-3 175B model requires approximately 34 GB of activation memory
for a batch size of 1.

Activation recomputation (or activation checkpointing) is a technique
that trades extra computation for memory. Instead of storing all
activations during the forward pass, it only stores (or checkpoints) the
input activations of a group of layers and recomputes other required
activations using an extra forward pass during the backward pass. The
activations to be saved can be determined heuristically or optimally
using certain computationally intensive optimization algorithms such as
mixed integer linear programming.

The primary advantage is the substantial reduction in memory usage,
allowing for the training of larger models or the use of larger batch
sizes. In addition, this technique is not restricted to any type of
model architecture and is generally available in most deep learning
frameworks.

On the contrary, the recomputation introduces computational overheads,
which can be very considerable when recomputing the
computational-intensive layers. As a result, the saved activations have
to be chosen carefully. For transformer architecture, if only the
activations at transformer layer boundaries are saved, 30%-40% execution
time overhead can be observed. Although some algorithms, such as
Checkmate, are proposed to find the optimal recomputation plan, they
require a long search time (> hours) for large models.
