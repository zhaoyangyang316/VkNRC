# Build VkNRC for Android on Windows

Use PowerShell at the root of the `androidnrc` checkout. Build the x86_64 APK first to test the renderer and Settings drawer in the Android Emulator. Build the arm64-v8a APK for the iQOO 15. Both contain the Cornell Box.

## Quick start on the prepared machine

On the prepared `E:\VkNRC` checkout, the Android toolchain and virtual device are already installed. Run these commands from the repository root.

```powershell
python .\scripts\build-android.py --abi x86_64 --sdk .\build\android-tools\sdk --jdk .\build\android-tools\jdk
.\scripts\run-android-emulator.ps1
```

Build the phone APK with the same tools by selecting the arm64 ABI.

```powershell
python .\scripts\build-android.py --abi arm64-v8a --sdk .\build\android-tools\sdk --jdk .\build\android-tools\jdk
```

| APK under `build\android\` | Runtime backend |
| --- | --- |
| `VkNRC-x86_64-debug.apk` | Emulator. GPU compute shaders intersect the Cornell triangles and run scalar neural inference and training. |
| `VkNRC-arm64-v8a-debug.apk` | Phone. The renderer selects a supported `VK_KHR_cooperative_matrix` tile and uses hardware ray queries. The iQOO 15 driver still needs physical-device verification. |

## 1. Prepare a new Windows checkout

Open PowerShell in the cloned repository, run `git switch androidnrc`, and confirm that `scripts\build-android.py` and `assets\cornell-box\cornell-box.obj` exist. The C++ dependencies are vendored. Python runs the packager, CMake and Ninja drive the NDK compiler, and the Windows Vulkan SDK compiles GLSL into embedded SPIR-V. The APK build uses the Python standard library.

Install the following Windows host tools and make their executables available on `PATH`. The versions below identify the tested setup.

| Tool | Tested version | Publisher |
| --- | --- | --- |
| Python 3 | 3.13.9 | [Python for Windows](https://www.python.org/downloads/windows/) |
| CMake | 3.31.1 (project minimum 3.22) | [CMake downloads](https://cmake.org/download/) |
| Ninja | 1.13 | [Ninja releases](https://github.com/ninja-build/ninja/releases) |
| Vulkan SDK for Windows, including `glslc.exe` | 1.4.321.1 | [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) |

```powershell
Get-Command python, cmake, ninja, glslc
python --version
cmake --version
ninja --version
glslc --version
```

## 2. Set up Java and the Android SDK

**Existing SDK and JDK.** Set `JAVA_HOME` to a JDK 17 directory containing `bin\javac.exe` and `ANDROID_HOME` to the Android SDK directory. An Android Studio SDK on Windows is commonly under `$env:LOCALAPPDATA\Android\Sdk`. Use its installed `cmdline-tools\<version>\bin\sdkmanager.bat` in step 3. If you use Android Studio's Device Manager, use the same AVD home when launching from PowerShell.

**Fresh setup.** The following layout matches the prepared checkout. Keep this PowerShell session open through the build. For a later session, repeat the variable assignments.

```powershell
$toolDir = Join-Path $PWD 'build\android-tools'
$env:JAVA_HOME = Join-Path $toolDir 'jdk'
$env:ANDROID_HOME = Join-Path $toolDir 'sdk'
$env:ANDROID_USER_HOME = Join-Path $toolDir 'user'
$env:ANDROID_EMULATOR_HOME = $env:ANDROID_USER_HOME
$env:ANDROID_AVD_HOME = Join-Path $toolDir 'avd'
$cliDir = Join-Path $env:ANDROID_HOME 'cmdline-tools\12.0'
New-Item -ItemType Directory -Force -Path $env:JAVA_HOME,$cliDir,$env:ANDROID_USER_HOME,$env:ANDROID_AVD_HOME | Out-Null
```

Review the publisher links, then download the two Windows x64 archives. Save them under `$toolDir` with the filenames below. The Microsoft link follows the current JDK 17 patch release, so use the checksum for the archive you downloaded.

| Save as | Download and verify |
| --- | --- |
| `jdk.zip` | [Microsoft OpenJDK 17 ZIP](https://aka.ms/download-jdk/microsoft-jdk-17-windows-x64.zip) · [Matching SHA-256](https://aka.ms/download-jdk/microsoft-jdk-17-windows-x64.zip.sha256sum.txt) |
| `sdktools.zip` | [Google Android command-line tools 12.0 ZIP](https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip) · [Google package index](https://dl.google.com/android/repository/repository2-1.xml). SHA-256 for this archive is `4d6931209eebb1bfb7c7e8b240a6a3cb3ab24479ea294f3539429574b1eec862`. |

Compare the hashes before extracting. Stripping the outer directory places `javac.exe` and `sdkmanager.bat` where the build commands expect them.

```powershell
Get-FileHash "$toolDir\jdk.zip","$toolDir\sdktools.zip" -Algorithm SHA256
tar.exe -xf "$toolDir\jdk.zip" -C "$env:JAVA_HOME" --strip-components 1
tar.exe -xf "$toolDir\sdktools.zip" -C "$cliDir" --strip-components 1
& "$env:JAVA_HOME\bin\javac.exe" -version
& "$cliDir\bin\sdkmanager.bat" --version
```

## 3. Install the Android packages

For an existing SDK, set `$cliDir` to its command-line tools directory first. The build targets Android API 35 and compiles native code for API 28 and later. Use SDK Manager to install the pinned Build Tools and NDK versions and review its license prompts. [Google SDK Manager guide](https://developer.android.com/tools/sdkmanager) · [NDK r27 release notes](https://github.com/android/ndk/wiki/Changelog-r27)

```powershell
$sdkmanager = Join-Path $cliDir 'bin\sdkmanager.bat'
& $sdkmanager "--sdk_root=$env:ANDROID_HOME" 'platform-tools' 'platforms;android-35' 'build-tools;35.0.0' 'ndk;27.2.12479018'
& $sdkmanager "--sdk_root=$env:ANDROID_HOME" 'emulator' 'system-images;android-35;google_apis;x86_64'
$env:ANDROID_NDK_HOME = Join-Path $env:ANDROID_HOME 'ndk\27.2.12479018'
$adb = Join-Path $env:ANDROID_HOME 'platform-tools\adb.exe'
Test-Path "$env:ANDROID_NDK_HOME\build\cmake\android.toolchain.cmake"
Test-Path "$env:ANDROID_HOME\platforms\android-35\android.jar"
& $adb version
```

Both path checks should print `True`. The emulator and its system image are needed only for the virtual phone.

## 4. Compile and package

```powershell
python .\scripts\build-android.py --abi x86_64
python .\scripts\build-android.py --abi arm64-v8a
Get-ChildItem .\build\android\VkNRC-*-debug.apk
```

Run the ABIs sequentially on the first build so they share the generated `build\android\debug.keystore`. Each build configures a separate Ninja tree, compiles the shaders and native `libvknrc.so` in Release mode, compiles the Java UI, packages the scene, signs the debug APK, and verifies its signature and 16 KiB alignment. Successful output ends with `Built ...apk`. The debug designation refers to the APK's signing and debuggable manifest.

Rerun the same command after source changes. Use `--jobs 4` to reduce parallelism, `--glslc` to select a host compiler explicitly, or `--ndk` to override a stale `ANDROID_NDK_HOME`. For an isolated clean build, use a new output directory, for example `--out .\build\android-clean`. That directory gets its own signing key. Keep the original key for updates to an installed demo.

## 5. Create and launch the emulator

Create the AVD once. Keep its name and port consistent with the launcher. On Windows, check that the Android Emulator reports a usable hypervisor. This checkout was tested with Windows Hypervisor Platform (WHPX). [Windows acceleration setup](https://developer.android.com/studio/run/emulator-acceleration) · [AVD Manager reference](https://developer.android.com/tools/avdmanager)

```powershell
& "$env:ANDROID_HOME\emulator\emulator.exe" -accel-check
'no' | & "$cliDir\bin\avdmanager.bat" create avd -n VkNRC_API35 -k 'system-images;android-35;google_apis;x86_64' -d pixel_5
& "$env:ANDROID_HOME\emulator\emulator.exe" -list-avds
.\scripts\run-android-emulator.ps1
```

The list should include `VkNRC_API35`. The launcher starts or reuses that AVD on `emulator-5572`, waits for Android, installs the x86_64 APK from `build\android`, and opens the app. It uses the checkout-local SDK and AVD when present. Cornell should appear with **Settings** in the upper right. To check training from another PowerShell window, set `$adb` to the same executable as in step 3.

```powershell
& $adb -s emulator-5572 logcat -d -s VkNRC AndroidRuntime
```

Look for `scalar NRC + compute traversal`, `First Cornell NRC frame presented` and an increasing `optimizer_step`. The emulator verifies Android rendering and training. Its frame time measures the scalar backend.

## 6. Install and verify on the iQOO 15

Enable developer options and USB debugging on the phone, connect a data-capable USB cable, unlock the screen, and accept the debugging authorization prompt. Select the phone's serial from `adb devices -l`. The explicit serial lets the emulator remain connected. [Android device setup](https://developer.android.com/studio/run/device)

```powershell
& $adb devices -l
$serial = Read-Host 'Enter the iQOO serial from the device list'
& $adb -s $serial shell getprop ro.product.cpu.abi
& $adb -s $serial install -r .\build\android\VkNRC-arm64-v8a-debug.apk
& $adb -s $serial shell am start -W -n org.vknrc.demo/.LauncherActivity
& $adb -s $serial logcat -d -s VkNRC AndroidRuntime
```

Expect `arm64-v8a` for the ABI. A successful hardware run reports the selected KHR matrix dimensions and precision, `KHR matrix + ray query`, a presented Cornell frame, and advancing optimizer steps. If a capability dialog appears, save its text and the `VkNRC` log. The phone's installed driver determines which tuples and other renderer features are available.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| SDK Manager or `javac` missing | Check the two extraction paths in step 2. `JAVA_HOME` points to the JDK root. Ensure the archive did not introduce an extra top-level directory. |
| Build cannot find `glslc`, or rejects `GL_KHR_cooperative_matrix` | Use the Windows Vulkan SDK's `Bin\glslc.exe`. Pass `--glslc "$env:VULKAN_SDK\Bin\glslc.exe"` to select it. |
| Missing `android.jar`, `llvm-strip` or `zipalign` | Check the SDK root, NDK revision, and Build Tools package in step 3. An explicit `--sdk` still allows an existing `ANDROID_NDK_HOME` to override the NDK path. |
| Launcher cannot find the AVD | Run `-list-avds` with the same SDK and `ANDROID_AVD_HOME` used to create it. The launcher expects the default build output and `VkNRC_API35` on port 5572. |
| `unauthorized`, wrong ABI, or signature mismatch on install | Authorize the computer on the phone, verify its serial and ABI, and use the corresponding APK. For `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, rebuild with the signing key used for the installed version. |

For render controls, observed emulator results, and remaining hardware checks, see the [Android port notes](android-port.md). To validate the KHR inference and gradient math separately on a capable host GPU, follow the [numerical test instructions](android-khr-validation.md).
