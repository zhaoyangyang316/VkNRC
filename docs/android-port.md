# VkNRC on Android

For prerequisites, downloads, and PowerShell commands from a fresh Windows checkout, use the [Android setup and build guide](android-build.md).

## Bring-up order

| Stage | Target | Pass condition |
| --- | --- | --- |
| 1. Build | Windows host, x86_64 and arm64-v8a ABIs | Compile the emulator's scalar shaders and the phone's KHR variants, validate SPIR-V, and package a signed APK with the Cornell OBJ and MTL for each ABI. |
| 2. Run in the simulator | Android Emulator, API 35, x86_64 | Install the APK, read the guest Vulkan report, render Cornell and train the scalar GPU reference network. Exercise Home, resume and activity recreation. Save a screenshot and the device log. |
| 3. Validate NRC | A Vulkan implementation exposing the required matrix tuple and ray-query features | Run inference and training; check finite output and compare frozen NRC and path tracing from the same camera. A capability report alone does not pass this stage. |
| 4. Test the phone | iQOO 15, arm64-v8a | Record the shipping driver and negotiated matrix tuple, run the same scene and lifecycle checks, then measure sustained performance on the actual device. |

The iQOO 15 uses Snapdragon 8 Elite Gen 5. Its product specification identifies the SoC; the installed Vulkan driver provides the supported matrix dimensions, component types, and scopes. Android Emulator exposes its own guest driver. A successful emulator launch tests Android integration; it cannot establish the phone's matrix support or frame rate. [iQOO specifications](https://www.iqoo.com/en/products/param/iqoo-15) · [Android Vulkan engine guidance](https://developer.android.com/games/develop/vulkan/native-engine-support)

## Design

- Use the existing Cornell geometry, camera convention, renderer and neural weights layout. Keep window handling and packaged asset access in the Android frontend.
- Select the phone matrix implementation using `VK_KHR_cooperative_matrix`. Enumerate supported multiply-add tuples, including A/B/C/result types, saturation and scope. The desktop NV shader uses different operations and layouts. Cooperative vectors are outside this implementation. [Khronos extension](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html) · [Matrix properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkCooperativeMatrixPropertiesKHR.html)
- Package a separate x86_64 emulator backend. Its Vulkan compute shaders intersect the small single-instance Cornell mesh directly and use scalar products for the neural network. This keeps training, inference, optimizer and presentation testable without guest access to hardware ray queries or cooperative matrices.
- Check the ray tracer and render graph independently of matrix support. A guest can support basic Vulkan presentation without supporting ray queries, the necessary storage-image formats, or matrix instructions. Record the actual rejection reason before making a pipeline.
- Start with the bundled Cornell Box and a small fixed render extent. Use the same camera and extent when comparing the two ABIs. Measure performance separately on the handset after correctness and lifecycle tests pass.

## Interpreting the screen

- **Emulator GPU reference** — Vulkan renders the untextured Cornell Box and runs online NRC training and inference. Its scalar MLP and triangle traversal provide a functional integration test. Its timing is specific to the emulator backend.
- **Phone KHR renderer** — stationary Cornell Box at 320×180, NRC on the left and path tracing on the right. The log prints the physical device, matrix tuple, precision and first presented NRC frame. The app builds the scene's acceleration structures and trains the MLP on the GPU.
- **CPU preview** — the capability dialog lists unavailable GPU features. Its 160×90 image uses the packaged OBJ/MTL, emitter triangles, diffuse shading and shadow rays. It is a static test of asset loading, camera and Android display. It does not exercise GPU tracing or neural training.
- **Native loading error** — the launcher shows the loading failure directly. Check the APK ABI against `adb shell getprop ro.product.cpu.abi`.

## Android controls

Tap **Settings** in the upper-right corner of the render. Scroll the drawer to reach the training, capture, and timing controls. **Done** returns to the full image. Tapping the image with the drawer closed still cycles split NRC/PT, full PT, and full NRC.

| Section | What you can change |
| --- | --- |
| View | Set the left and right halves independently to PT, NRC, or Cache. Choose a fixed resolution from 160×90 through 1280×720. Cornell Box is the bundled mobile scene. |
| Lighting | Toggle next event estimation, indirect illumination only, or NRC fast mode. Adjust max bounces and, with fast mode enabled, minimum roughness. |
| Accumulation | Toggle accumulation, average the latest N samples, or restart at zero SPP. A rolling window of zero retains all samples. The maximum reflects the selected resolution and the 64 MiB mobile history budget. |
| Cache training | Toggle EMA weights, lock training, train one frame while locked, or retrain from fresh weights. |
| Capture | **Capture frame (EXR)** stores a linear HDR image at the native render resolution with optional settings in its header. Capture also works on a paused frame. **Save EXR as…** opens Android's file picker so you can copy the latest capture to Downloads or another destination. |
| Benchmark | Choose 1–120 seconds. The result reports frame count, elapsed time, average ms/frame and FPS after 32 warmup frames. |

**Pause rendering** freezes the image, SPP and training. Resume to edit the workload. Changing NEE, max bounces, or the active fast-mode rule restarts training. Changing the split or resolution resets accumulation and keeps the trained network. Options and cache state survive window recreation within the running activity. A new process starts with default settings.

## Handset handoff

Build with `python .\scripts\build-android.py --abi arm64-v8a`. When the iQOO is available, enable USB debugging, authorize this computer on the phone, select its serial from `adb devices -l`, and repeat the install and launch commands with that serial and `VkNRC-arm64-v8a-debug.apk`. Save the `VkNRC` log before tuning the shaders. For a successful GPU run, compare a screenshot against the desktop Cornell camera, verify that training stays finite, and check Home/resume and repeated launches before timing the device. Phone timing should include resolution, driver version, training state and run length.

## Verification record — 2026-11-05

| Target | Observed result |
| --- | --- |
| Android build | Built and signed x86_64 and arm64-v8a APKs using NDK r27c and Build Tools 35.0.0. Signature verification and 16 KiB alignment checks passed. |
| Android Emulator | API 35 Google APIs x86_64 image, emulator 37.2.13, WHPX, host graphics on RTX 5090. The guest reported Vulkan 1.3, subgroup size 32, and no KHR cooperative-matrix or ray-query extension. |
| End-to-end GPU reference | Cornell rendered at 160×90 and presented to a 2340×1080 landscape surface. GPU readback returned Adam step 16 at frame 4 and step 512 at frame 128. Tapping cycled split NRC/PT → full PT → full NRC → split. Switching views restarts accumulation while preserving the trained weights. A cold relaunch presented the scene and trained again. |
| Image smoke check | Independent full NRC and PT captures both showed the ceiling emitter, rotated blocks and colored walls. In a central crop their mean absolute RGB difference was 3.01 on an 8-bit display scale. Sample counts were not matched; use a controlled linear-radiance comparison for quality evaluation. |
| Lifecycle and launcher | Sent the app Home, returned through its launcher icon, and verified it presented and trained again. Started a fresh visible emulator using the PowerShell launcher. The same activity instance resumed after simulated screen sleep/wake and presented Cornell again. |
| Numerical test | On the native RTX 5090 driver, both KHR accumulator variants and the scalar variant passed inference and gradient checks against the CPU reference. [Commands, workload and tolerances](android-khr-validation.md) |
| Windows regression | The desktop Release target built and completed its timed Cornell benchmark. |
| Android Settings | Drove the API 35 emulator's Settings drawer by touch. Verified frozen and advancing SPP across Pause/Resume, changed resolution and view, set lighting, rolling SPP, training lock and EMA, and requested one training frame. The five-second benchmark returned 301 frames in 5.016 s. Exported an EXR through the Android Downloads picker and read back its 320×180 resolution, 2 SPP, Cache/PT split, NEE off, 8 bounces, fast mode on and minimum roughness 0.311. |
| iQOO 15 | ARM64 APK available. Physical-device extension report, pipeline creation, image validation, memory use and sustained frame timing await an actual phone. |

Local evidence from this checkout is in `build/android/`: `emulator-reference.log`, `emulator-split.png`, `emulator-pt.png`, `emulator-nrc.png` and `emulator-relaunch.log`. SDK packages and generated artifacts stay under `build/`.
