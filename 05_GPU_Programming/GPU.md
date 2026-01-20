## Learning GPU
- Good diagram: https://medium.com/codex/understanding-the-architecture-of-a-gpu-d5d2d2e8978b
- Basically, GPU cores execute the exact same instruction over lots of different data
- You can do N (number of cores) same ops per clock cycle

- GPUs are basically lots of threads (cuda cores) that are grouped into blocks (cores)
- You have SIMD (Single Instruction Multiple Data) on CPUs and SIMT (Single Instruction Multiple
    Thread) on GPUs

### CPU cores
- Larger for more complex logical tasks
- Include out of order execution, branch prediction, speculation, and register renaming
- Big caches because the time it takes to go from ram to core is much larger (compared to the
    time to go from VRAM to a GPU core)
- GPUs: 1024 bit wide SIMD machines

### GPU cores
- No heavy branch prediction or out of order execution
- Small or no private caches (meaning a set of many cores share the same cache)
- Basically only integer and floating point ops
- Thousands of cores executing the same instructions across many data elements at once
    SIMD (Singe Instruction, Multiple Data) / SIMT (Single Instruction, Multiple Threads)
- They are basically just vector ALUs with minimal scheduling logic

- GTX 1660 Ti FP32: ~5.44 TFLOPS
- RTX 3090 FP32: ~35.58 TFLOPS

- kernel: name of a function run by CUDA on the GPU
- thread: CUDA will run many threads in parallel on the GPU, each thread executes the kernel
    can think of it like a series of cores
- blocks: threads are grouped into blocks, a programming abstraction, thread block can contain up to 1024 threads
- grid: contains thread blocks
- voxel: like a pixel is the smallest data point in a 2d image, a voxel is the smallest data point in a 3d image

