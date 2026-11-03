# Vulkan Neural Radiance Caching

[![Windows MinGW](https://github.com/AdamYuan/VkNRC/actions/workflows/windows-mingw.yml/badge.svg)](https://github.com/AdamYuan/VkNRC/actions/workflows/windows-mingw.yml)
[![Windows MSVC](https://github.com/AdamYuan/VkNRC/actions/workflows/windows-msvc.yml/badge.svg)](https://github.com/AdamYuan/VkNRC/actions/workflows/windows-msvc.yml)

![](https://raw.githubusercontent.com/AdamYuan/VkNRC/master/screenshot/0.png)

Vulkan Implementation of NVIDIA's
paper [Real-time Neural Radiance Caching for Path Tracing](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing).

The Fully-Fused MLP is implemented with VK_NV_cooperative_matrix (the KHR one is too limited), and it has better
backpropagation performance than the author's [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

## Build on Windows

Requirements: Visual Studio 2022 C++ Build Tools (with a Windows SDK), CMake 3.22+
and a Vulkan SDK with `glslc` on `PATH`. The C++ dependencies are vendored in `dep`;
vcpkg, CUDA and Python packages are not required for this application.

From the repository root in PowerShell:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target VkNRC --parallel 8
.\run-demo.cmd
```

The executable is `build/Release/VkNRC.exe`. Shaders are compiled and embedded in
the executable. The renderer needs Vulkan 1.3 functionality, NVIDIA cooperative
matrices (`VK_NV_cooperative_matrix`), ray queries/acceleration structures, FP16
arithmetic/storage and float32 atomic addition. A CUDA installation alone does not
satisfy these Vulkan requirements. Compiled neural shaders support subgroups of
16 or 32 threads.

The bundled MyVK dependency traversal has a compatibility workaround for an MSVC
19.44 internal compiler error: nested generic visitors are separated into local
helpers without changing traversal or dependency edges.

Verified locally on 2026-11-03 with MSVC 19.44, CMake 3.31.1, Vulkan SDK 1.4.321.1,
and an RTX 5090 (driver 617.42): x64 Release compilation succeeded and the bundled
scene rendered interactively. This is a build/startup/visual smoke test, not a
numerical validation of the neural cache or a performance benchmark.

## Try the demo

Double-click `run-demo.cmd` to open the included Cornell box, using
[Cornell's measured geometry and camera](https://bowers.cornell.edu/computer-graphics/data).
The room is nearly cubic, with rotated short/tall blocks and an emissive ceiling
panel. RGB materials approximate the reference's spectral reflectances. No external
textures or downloads are needed. Landscape windows show margins beside the square
box; the geometry is not stretched to fill the window.

The launcher places the camera outside the opening with the reference vertical
field of view. To launch that view directly:

```powershell
.\build\Release\VkNRC.exe .\assets\cornell-box\cornell-box.obj --camera 0 -0.005007153 3.86123033 --fov 39.30765
```

To use another scene, pass an OBJ path:

```powershell
.\run-demo.cmd "C:\path\to\scene.obj"
```

The OBJ must reference a valid MTL and assign a material to every face. Either
provide valid UVs for every face or omit UVs entirely. The loader centers and
rescales geometry to a maximum extent of 2 units. Without camera options the camera
starts at the origin looking toward negative Z; large imported scenes may start
with the camera inside geometry. `--camera x y z` sets its position in normalized
scene coordinates, and `--fov degrees` sets the vertical field of view.

Environment lighting is automatically disabled when a material assigned to scene
triangles emits light. Otherwise the renderer keeps its white environment at
intensity 10. Emission textures replace `Ke`, just as in the shading code: a loaded
texture with any nonzero RGB texel counts as emissive, a black texture does not,
and a failed texture load falls back to `Ke`. Unused materials do not affect this
decision. The chosen environment intensity is printed at startup. With the Cornell
box, illumination comes from the ceiling panel and the background is black.
Plain path tracing will be noisy without accumulation because the renderer samples
the small ceiling light through BSDF rays; it has no explicit light-sampling pass.

Controls:

- **W/A/S/D:** move; **Space / Left Shift:** up / down.
- **Hold left mouse and drag:** look around. Click outside the panel if camera
  movement is blocked by UI focus.
- **View → Left / Right:** select `None` (plain path tracing), `NRC`, or `Cache`
  (direct cache visualization). Both halves initially use NRC.
- Compare **Left = None**, **Right = NRC**, with **Accumulate** off to see noise
  differences. Accumulate averages successive frames while the camera is still.
- **NRC → Use EMA:** smooth network weights (initially off). **Lock** pauses
  training; **Train 1-Frame** advances a locked network; **Re-Train** resets it.

For a stationary comparison, first let the network train, then enable **Lock** and
**Accumulate**. Lock resets the accumulation so earlier learning transients do not
remain in the comparison. NRC is an approximation and will retain some bias even
with many accumulated samples; SPP reduces sampling noise, not cache error.

## Repeatable radiance comparison

The diagnostic mode trains with the same NRC/plain-PT split as the interactive
comparison, freezes the network, then captures both methods over the full image.
It uses a fixed camera/viewport and configurable RNG seed, counts submitted frames,
and writes linear RGB PFM files without tone mapping or UI. GPU atomic scheduling
can still cause small training differences between runs with the same seed.

```powershell
.\build\Release\VkNRC.exe .\assets\cornell-box\cornell-box.obj --camera 0 -0.005007153 3.86123033 --fov 39.30765 --compare build/comparison --warmup 8192 --samples 4096 --seed 1337
```

Outputs are `build/comparison/nrc.pfm` and `build/comparison/path-tracing.pfm`.
The program exits after capture. Use an empty output directory to retain old runs.

Local checks with seeds 1337, 42 and 2026, 8192 training frames and 4096 samples per
capture produced finite images. Across shaded pixels (reference luminance >0.005,
maximum RGB <5, excluding the background and direct emitter), mean NRC luminance
ranged from 3.6% below to 1.3% above plain path tracing. Relative luminance MAE was
7.2–7.6%; this includes residual cache bias and reference sampling noise. These
checks did not reproduce the reported severe dark regions and do not establish
their exact cause.

## Source guide

The [paper](https://arxiv.org/html/2106.12372v2) trains a radiance cache during
rendering, using short paths for display and longer paths for self-training.
Cached predictions replace the remaining path contribution. A small fused neural
network and averaged weights make the cache practical for interactive rendering.

This implementation's main components are:

| Source | Responsibility |
| --- | --- |
| `src/main.cpp` | Device setup, scene loading, render loop and controls |
| `src/rg/NRCRenderGraph.cpp` | Visibility → tracing → inference → training → display |
| `shader/src/path_tracer.comp` | Short rendering paths, extended training paths and cache queries |
| `shader/src/NRCRecord.glsl` | Encoded position, direction, normal, roughness and reflectance inputs |
| `shader/src/NN_nv.glsl` | Fused FP16 cooperative-matrix neural network operations |
| `shader/src/nrc_gradient.comp` | Backpropagation through five 64-wide hidden layers |
| `shader/src/nrc_optimize.comp` | Adam updates and optional EMA weights |
| `src/VkNRCState.hpp` | Network state and four training batches of up to 16,384 records |
| `src/RenderCapture.cpp` | Synchronized linear-radiance readback for numerical comparisons |

Cache queries represent scattered radiance toward the preceding path vertex,
excluding local emission. Training uses the same quantity and throughput factors.
Training coverage is independent of the displayed left/right methods. One in
sixteen training paths uses traced transport with Russian roulette rather than a
cache endpoint; bootstrap endpoints are not used as their own labels. The
path-spread heuristic uses the preceding sample's PDF and the incoming cosine.

The neural shaders synchronize cooperative output stores before shared-memory
reads, zero padded inputs, and discard nonfinite prediction channels before
compositing. FP16 backpropagation uses a loss scale of 1/128, reversed in the FP32
optimizer, to reduce overflow risk with bright area lights.

The renderer is simplified: plain tracing and reference training paths use Russian
roulette with a 64-vertex safety cap, which leaves a small truncation bias. There
is a fallback white environment for scenes without emitters, with no ReSTIR,
next-event light sampling or adaptive training tiles. Its images and timings
should not be treated as a full reproduction of the paper's renderer.
