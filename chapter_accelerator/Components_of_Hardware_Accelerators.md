# Components of Hardware Accelerators

A hardware accelerator typically comprises multiple on-chip caches and
various types of arithmetic units. In this section, we'll examine the
fundamental components of hardware accelerators, using the Nvidia Volta
GPU architecture as a representative example.

## Architecture of Accelerators

Contemporary graphics processing units (GPUs) offer remarkable computing
speed, ample memory storage, and impressive I/O bandwidth. A top-tier
GPU frequently surpasses a conventional CPU by housing double the number
of transistors, boasting a memory capacity of 16 GB or greater, and
operating at frequencies reaching up to 1 GHz. The architecture of a GPU
comprises streaming processors and a memory system, interconnected
through an on-chip network. These components can be expanded
independently, allowing for customized configurations tailored to the
target market of the GPU.

Figure :numref:`ch06/ch06-gv100` illustrates the architecture of the
Volta GV100 . This architecture has:

![Volta GV100](../img/ch06/V100.png)
:label:`ch06/ch06-gv100`
