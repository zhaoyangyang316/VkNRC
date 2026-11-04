# Checking the KHR cooperative matrix backend

Run this from the repository root on a Vulkan 1.3 host with CMake, a C++17 compiler, the Vulkan SDK, Python 3 and NumPy. The test compiles a headless Vulkan runner, runs `glslc` and `spirv-val`, and dispatches both the inference and gradient kernels for each supported KHR accumulator precision. It also runs the x86_64 emulator's scalar compute kernels on a logical device created with cooperative matrices disabled.

```sh
python test/khr_matrix/run.py
```

All generated files are under `build/khr-matrix-validation`. The runner prints the GPU and tile used. A missing precision is reported as a skip. The program exits with status 77 if it cannot run any of the variants. For an additional Vulkan validation-layer check, enable `VK_LAYER_KHRONOS_validation` through `VK_INSTANCE_LAYERS` before running.

The test shader includes the renderer's `shader/src/NN_khr.glsl`. The fixture supplies 256 encoded input records from a seeded generator across two workgroups. Five 64×64 shifted, signed permutation matrices exercise weight orientation and ReLU masks, followed by a 3×64 RGB layer. The inputs, weights and targets use binary fractions. The CPU reference evaluates these same matrices using NumPy and compares all 768 output channels for finite, exact agreement. It computes the luminance-normalized loss derivative, propagates it through all five hidden layers, and checks all 20,672 accumulated FP32 weight gradients. The allowed relative L2 error is below 0.003, with a per-weight absolute tolerance of 0.001 plus 0.5% of the reference magnitude.

An RTX 5090 run using the 16×16×16 KHR tile passed all four KHR variants and both scalar variants. Gradient relative L2 errors were approximately 5.05×10⁻⁵ for KHR FP16 accumulation and 2.1×10⁻⁸ for both KHR FP32 accumulation and scalar compute. All variants store weights, activations and activation derivatives in FP16. The relative error can vary slightly with the order of concurrent gradient additions.

The test exercises the neural kernels directly. The production tracer, image descriptors, WSI and Android driver also need the app-level smoke test on the target device. The x86_64 emulator run exercises scalar compute. An arm64 hardware run exercises the KHR matrix driver.
