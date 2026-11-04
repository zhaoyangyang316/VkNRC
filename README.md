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
arithmetic/storage, float32 atomic addition and nonuniform sampled-image indexing.
A CUDA installation alone does not
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

Double-click `build/Release/VkNRC.exe` or `run-demo.cmd` to open the included Cornell box, using
[Cornell's measured geometry and camera](https://bowers.cornell.edu/computer-graphics/data).
The room is nearly cubic, with rotated short/tall blocks and an emissive ceiling
panel. RGB materials approximate the reference's spectral reflectances. No external
textures or downloads are needed. Landscape windows show margins beside the square
box; the geometry is not stretched to fill the window.

No command-line arguments are needed. The application places the camera outside
the opening with the reference vertical field of view:

```powershell
.\build\Release\VkNRC.exe
```

To use another scene, pass an OBJ path:

```powershell
.\run-demo.cmd "C:\path\to\scene.obj"
```

The **Scene** dropdown lists OBJ files recursively under `assets`, plus a custom
scene supplied on the command line. The bundled scenes are **Cornell Box**,
**Living Room**, and **Veach Door**, each with a camera preset. Place additional
OBJ/MTL files and their textures under `assets` before starting to add them to
the list. Selecting a scene resets its camera, network and
accumulation, unlocks training, and resumes rendering. A failed load keeps the
previous scene and shows an error. Assets are located relative to the executable
(including the repository above `build/Release`), with the configured source path
as a fallback, so launching from another working directory works. CMake installation
also includes the assets beside the executable.

The OBJ must reference a valid MTL and assign a material to every face. Either
provide valid UVs for every face or omit UVs entirely. The loader centers and
rescales geometry to a maximum extent of 2 units. For custom scenes, without camera
options the camera starts at the origin looking toward negative Z; large imported scenes may start
with the camera inside geometry. `--camera x y z` sets its position in normalized
scene coordinates, and `--fov degrees` sets the vertical field of view.
A sibling `scene.camera` file can supply a friendly dropdown label and default
camera for `scene.obj`; command-line camera options override it at startup:

```text
name My Scene
position 0 0 1
target 0 0 0
fov 60
```

Camera coordinates use the same normalization as the OBJ: subtract the mesh AABB
center, then multiply by `2 / max(extent)`. Cameras use Y-up without roll; `fov`
is vertical in degrees. Lines beginning with `#` are comments. Invalid sidecars
produce a catalog diagnostic and use the normal camera fallback.

The two imported scenes preserve their source area lights; no extra illumination
was added. Their assigned, nondegenerate emissive geometry was checked in the
running renderer:

| Scene | Source lights | Triangles sampled by NEE | Environment intensity |
| --- | --- | ---: | ---: |
| Living Room | BackLight (`Ke 2 2 2`), LeftLight, MiddleLight, RightLight (`Ke 5 5 5`) | 8 | 0 |
| Veach Door | Original rectangle (`Ke 1000 1000 1000`) | 2 | 0 |

**Living Room** is Jay-Artist's *The White Room Cycles* from the requested
[McGuire archive entry](https://casual-effects.com/g3d/data10/index.html#mesh21),
licensed CC BY 3.0. The OBJ conversion is by Nicholas Hull, with material
corrections by Morgan McGuire; the original camera comes from Benedikt Bitterli's
matching `living-room-2` conversion. Its geometry, textures and emitter radiance
are retained. The MTL adds explicit roughness using `Pr = sqrt(2 / (Ns + 2))`.
This is the archive model; the NRC paper's exact camera, lighting and Falcor
material configuration are not reproduced or independently verified here.
See [Living Room attribution](assets/living-room/LICENSE.txt).

**Veach Door** is Benedikt Bitterli's CC0 *Veach, Ajar* from his
[rendering resources](https://benedikt-bitterli.me/resources/). Its transformed
meshes, original area emitter, textures and camera were converted to OBJ/MTL;
62 zero-area source triangles were omitted. The conversion is reproducible with
`scripts/import-veach-door.py --source <extracted-veach-ajar-directory>` using
Python 3's standard library. See [conversion details](assets/veach-door/ATTRIBUTION.txt)
and [input hashes and light audit](assets/veach-door/import-report.json).

Both are adaptations to VkNRC's current surface model: it uses flat triangle
normals and opaque materials, with Beckmann specular reflection and two-sided
mesh emission. Living Room's lampshade is opaque; Veach Door's glass teapot has
reflection but no transmission or refraction, and conductor/GGX materials are
approximated. These limitations affect agreement with the source renders.
The imported camera/catalog checks and Release build passed locally. Both scenes
also completed 1,024 training frames followed by 512 samples per method at
1280x720, with finite NRC/path-tracing captures and no reported Vulkan validation
errors. This checks successful loading and rendering, not convergence or agreement
with the paper.

Environment lighting is automatically disabled when a material assigned to scene
triangles emits light. Otherwise the renderer keeps its white environment at
intensity 10. Emission textures replace `Ke`, just as in the shading code: a loaded
texture with any nonzero RGB texel counts as emissive, a black texture does not,
and a failed texture load falls back to `Ke`. Unused materials do not affect this
decision. The chosen environment intensity is printed at startup. With the Cornell
box, illumination comes from the ceiling panel and the background is black.
Next event estimation (NEE) is enabled by default. At each continued surface vertex,
the renderer samples an emissive triangle and traces a shadow ray to it. Power-heuristic
multiple importance sampling (MIS) combines this estimate with BRDF-sampled emitter
hits. Scenes without emitters instead sample the constant environment. This applies
to both plain path tracing and NRC training; indirect illumination still benefits
from accumulation or the cache.

Controls:

- **Capture frame (EXR)** saves the latest rendered image, including the current
  split view and accumulation, as a linear 32-bit RGB OpenEXR. It works while
  rendering or paused and preserves the rendered resolution. Captures are taken
  before tone mapping and the UI overlay. The panel shows the full file path
  under `captures` in the working directory. **Include render metadata** is on
  by default. The header records the resolution, NEE, max bounces, SPP, left
  and right methods (**NRC**, **PT**, or **Cache**), and scene name. Labels
  are filled automatically. SPP counts the samples in the captured image:
  one with accumulation off, the number collected so far during warm-up, and
  at most the selected window size for rolling accumulation.
  Enable **Print metadata in lower right** to draw those same fields
  on a dark panel in the bottom-right corner of the saved image. This option
  is off by default and can be used independently of the EXR header metadata.
- **Indirect illumination only:** show illumination that has scattered at least
  twice before reaching the camera. This removes visible emitter emission and
  background, direct-light samples at the primary surface, and emitter/environment
  hits reached by the first sampled bounce. Later-bounce lighting remains, with
  NEE either on or off. It applies to both plain path tracing and NRC; **Cache**
  view also traces the first bounce before querying the cache in this mode.
  Changing it clears accumulation while preserving the trained cache and lock
  state. The option is off by default; enable it at startup with `--indirect-only`.
  With this renderer's depth convention, **Max bounces** 1 or 2 produces black;
  use at least 3 to retain indirect light.
- **Next event estimation:** toggle explicit direct-light sampling with MIS. Changing
  it resets accumulation and retrains the cache, while preserving the accumulation window. Use
  `--no-nee` to disable it from the command line, including in `--compare` mode.
- **Pause rendering / Resume rendering:** freeze the image, path tracing, cache
  training and sample count while keeping the UI responsive. Camera and rendering
  settings are inactive until resumed. Resizing displays the frozen image with its
  original aspect ratio and resets accumulation for the next rendered frame.
- **Max bounces:** set the trace limit from 1 to 64 (default 64). Changing it resets
  accumulation and retrains the cache. It counts explicitly traced surface vertices,
  including the primary hit; 1 shows only directly visible emission/background.
  NRC may still represent additional transport through an earlier cache query,
  because the network is not conditioned on remaining depth. The same limit is
  available as `--max-bounces 8` on the command line.
- **Accumulate last:** a positive value averages the latest N samples per pixel
  while rendering and training keep running. For 4, frame 5 averages samples 2–5,
  frame 6 averages samples 3–6, and so on. During startup it averages the samples
  collected so far. This is a finite sliding window, not an exponential average.
  **0** averages all samples since the last reset (when **Accumulate** is enabled).
  Changing the window or clicking **Restart accumulation** clears sample history
  while preserving the trained cache. Camera, scene, viewport and rendering-setting
  changes also clear history. **Accumulate** off shows only the current sample;
  turning it back on starts a fresh history with the selected window.
  Use `--accumulate-last 4` (or the compatibility alias `--max-spp 4`) to start
  with a four-sample window. In diagnostic `--compare` mode, `--samples` sets how
  many frames each method renders; the final capture averages the selected window
  (or all those frames if the window is 0).
  Stored history is limited to 64 samples and 1 GiB, so the maximum decreases
  with resolution (64 at 1280×720, 32 at 1920×1080, 8 at 3840×2160).
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
set **Accumulate last** to a small value such as 1, 4 or 8. Set the two view methods to
`None` and `NRC` to compare them at the same sample count. The window keeps advancing;
use **Pause rendering** to freeze an image for inspection. **Restart accumulation**
clears the window while keeping the trained weights. Lock resets the
accumulation so earlier learning transients do not remain in the comparison.
NRC is an approximation and will retain some bias even
with many accumulated samples; SPP reduces sampling noise, not cache error.

The indirect-light regression test uses Python 3's standard library and the built
renderer. It checks full = direct + indirect in plain path tracing, depth limits,
and isolated emitter/environment cases in both NRC and plain PT, with NEE on/off:

```powershell
python test/indirect_illumination.py --validation
```

All 19 render cases passed locally; the largest per-channel decomposition error
was below `1e-5`, and no Vulkan validation errors were reported. Captures and logs
are saved under `build/indirect-illumination`.

The rolling-window regression test runs the actual display shader on synthetic
radiance frames and checks GPU readback against a CPU average of the last N frames:

```powershell
cmake --build build --config Release --target VkNRCRollingTest --parallel 8
.\build\Release\VkNRCRollingTest.exe .\build\rolling-validation --validation
```

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

To compare NEE on/off, repeat the command with `--no-nee` and a different output
directory. A local Cornell check with seed 1337 measured 8-SPP plain-tracer
luminance RMSE of 0.2315 without NEE and 0.0213 with NEE, against a 4096-SPP NEE
reference (about 10.9 times lower error). The two 4096-SPP plain-tracer captures
differed in mean luminance by 0.015%. After 8192 training frames, NRC's mean
luminance was 0.16% below the NEE path tracer. These measurements use shaded pixels
(reference luminance >0.005, maximum RGB <5), excluding background and visible
emitters; they are one scene/seed check, not a general error bound or timing claim.

Additional checks covered constant/textured emitters, black and missing emission
maps, directly visible emission at depth one, and diffuse/glossy environment-only
scenes. An unoccluded diffuse plane with reflectance 0.7 under environment 10
matched the analytic RGB value (7,7,7) to within 1.2e-5. Shader validation and the
Vulkan validation-layer smoke run passed. Random samples use [0,1) to avoid
zero-PDF endpoints, and acceleration-structure buffers include device-address usage.

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
| `shader/src/path_tracer.comp` | NEE/MIS, short rendering paths, extended training paths and cache queries |
| `src/VkScene.cpp` | Scene upload and emissive-triangle table, including emission texture handling |
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

The cache represents scattered radiance including direct lighting. Cache endpoints
therefore skip local NEE; continuing paths include it in both the displayed result
and training targets. MIS weights apply to emitted radiance reached by a BRDF ray,
not to the scattered-radiance cache prediction. Triangle selection is uniform,
followed by uniform area sampling; PDFs use transformed world-space areas and the
existing two-sided emission convention. Emission textures are evaluated at sampled
UVs. The direct-light estimator follows the light/BSDF sampling approach described
in [PBRT's path tracer](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer).

The neural shaders synchronize cooperative output stores before shared-memory
reads, zero padded inputs, and discard nonfinite prediction channels before
compositing. FP16 backpropagation uses a loss scale of 1/128, reversed in the FP32
optimizer, to reduce overflow risk with bright area lights.

The renderer is simplified: plain tracing and reference training paths use Russian
roulette with a configurable vertex cap (64 by default), which leaves a truncation bias. There
is a fallback white environment for scenes without emitters, with no ReSTIR,
power-weighted light selection or adaptive training tiles. Its images and timings
should not be treated as a full reproduction of the paper's renderer.
