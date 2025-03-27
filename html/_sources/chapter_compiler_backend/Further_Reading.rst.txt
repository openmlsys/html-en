
Further Reading
===============

1. Memory allocation is an important concept of a machine learning
   backend. For further reading, see *Training Deep Nets with Sublinear
   Memory Cost*\  [1]_ and *Dynamic Tensor Rematerialization*\  [2]_.

2. For more about runtime scheduling and execution, see A *Lightweight
   Parallel and Heterogeneous Task Graph Computing System*\  [3]_,
   *Dynamic Control Flow in Large-Scale Machine Learning*\  [4]_, and
   *Deep Learning with Dynamic Computation Graphs*\  [5]_.

3. For further reading about operator compilers, see *Halide: A Language
   and Compiler for Optimizing Parallelism, Locality, and Recomputation
   in Image Processing Pipelines*\  [6]_, *Ansor: Generating
   High-Performance Tensor Programs for Deep Learning*\  [7]_, and
   *Polly - Polyhedral optimization in LLVM*\  [8]_.

4. One of challenges faced by modern deep learning compiler frameworks
   is to achieve performance levels comparable to manually optimized
   libraries that are specific to the target platform. To address this
   challenge, auto-tuning frameworks utilize statistical cost models to
   dynamically and efficiently optimize code. However, these frameworks
   also have certain drawbacks, such as the need for extensive
   exploration and training overheads in order to establish the cost
   model. Recent work, like *MetaTune: Meta-Learning Based Cost Model
   for Fast and Efficient Auto-tuning Frameworks*\  [9]_ predicts the
   performance of optimized codes with pre-trained model parameters.

.. [1]
   https://arxiv.org/abs/1604.06174

.. [2]
   https://arxiv.org/abs/2006.09616

.. [3]
   https://arxiv.org/abs/2004.10908

.. [4]
   https://arxiv.org/abs/1805.01772

.. [5]
   https://arxiv.org/abs/1702.02181

.. [6]
   https://dl.acm.org/doi/abs/10.1145/2499370.2462176

.. [7]
   https://arxiv.org/abs/2006.06762

.. [8]
   https://arxiv.org/abs/2105.04555

.. [9]
   https://arxiv.org/abs/2102.04199
