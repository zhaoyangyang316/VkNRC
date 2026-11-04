"""Build a locally signed VkNRC NativeActivity APK using an existing Android SDK and NDK."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
API = "35"
BUILD_TOOLS = "35.0.0"
NDK = "27.2.12479018"
MIN_API = "28"
CORNELL_FILES = ("cornell-box.obj", "cornell-box.mtl")


def require_file(path):
    if not path.is_file():
        raise ValueError(f"Missing required file: {path}")
    return path.resolve()


def executable(name, directory=None):
    path = shutil.which(name, path=str(directory) if directory else None)
    if not path:
        raise ValueError(f"Cannot find {name} in {directory or 'PATH'}")
    return Path(path).resolve()


def run(*argv):
    subprocess.run([str(arg) for arg in argv], check=True)


def build(args):
    if not args.sdk:
        raise ValueError("Specify --sdk or set ANDROID_HOME to the installed SDK.")
    sdk = Path(args.sdk).resolve()
    ndk = Path(args.ndk).resolve() if args.ndk else sdk / "ndk" / NDK
    toolchain = require_file(ndk / "build/cmake/android.toolchain.cmake")
    strip_name = "llvm-strip.exe" if os.name == "nt" else "llvm-strip"
    strip = next(ndk.glob(f"toolchains/llvm/prebuilt/*/bin/{strip_name}"), None)
    if not strip:
        raise ValueError(f"Cannot find {strip_name} in {ndk}")
    platform = require_file(sdk / "platforms" / ("android-" + API) / "android.jar")
    tools = sdk / "build-tools" / BUILD_TOOLS
    aapt = executable("aapt2", tools)
    align = executable("zipalign", tools)
    signer = require_file(tools / "lib/apksigner.jar")
    d8 = require_file(tools / "lib/d8.jar")
    jdk_bin = Path(args.jdk) / "bin" if args.jdk else executable("javac").parent
    java, javac, keytool = (executable(name, jdk_bin) for name in ("java", "javac", "keytool"))
    cmake = executable("cmake")
    ninja = executable("ninja")
    glslc = require_file(Path(args.glslc)) if args.glslc else executable("glslc")

    out = args.out.resolve()
    native = out / "native" / args.abi
    out.mkdir(parents=True, exist_ok=True)
    run(cmake, "-S", ROOT, "-B", native, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}", f"-DANDROID_ABI={args.abi}", f"-DANDROID_PLATFORM=android-{MIN_API}",
        "-DANDROID_STL=c++_static", "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON", "-DCMAKE_BUILD_TYPE=Release",
        f"-DGLSLC={glslc}", f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={native / 'lib'}")
    run(cmake, "--build", native, "--target", "vknrc", "--parallel", args.jobs)
    library = require_file(native / "lib/libvknrc.so")
    apk = out / f"VkNRC-{args.abi}-debug.apk"
    keystore = out / "debug.keystore"
    if not keystore.is_file():
        run(keytool, "-genkeypair", "-keystore", keystore, "-storepass", "android", "-keypass", "android",
            "-alias", "androiddebugkey", "-keyalg", "RSA", "-keysize", "2048", "-validity", "10000",
            "-dname", "CN=Android Debug,O=Android,C=US", "-noprompt")

    with tempfile.TemporaryDirectory(dir=out) as stage_dir:
        stage = Path(stage_dir)
        classes = stage / "classes"
        classes.mkdir()
        sources = sorted((ROOT / "android/java").rglob("*.java"))
        run(javac, "-encoding", "UTF-8", "--release", "8", "-classpath", platform,
            "-d", classes, *sources)
        jar = stage / "classes.jar"
        with zipfile.ZipFile(jar, "w") as archive:
            for cls in classes.rglob("*.class"):
                archive.write(cls, cls.relative_to(classes).as_posix())
        run(java, "-cp", d8, "com.android.tools.r8.D8", "--lib", platform, "--min-api", MIN_API,
            "--output", stage, jar)
        unsigned = stage / "unsigned.apk"
        run(aapt, "link", "-I", platform, "--manifest", ROOT / "android/AndroidManifest.xml", "-o", unsigned)
        # Retain symbols in the build tree for debugging, and keep the install small.
        stripped = stage / "libvknrc.so"
        run(strip, "--strip-unneeded", "-o", stripped, library)
        with zipfile.ZipFile(unsigned, "a") as archive:
            archive.write(stage / "classes.dex", "classes.dex")
            archive.write(stripped, f"lib/{args.abi}/libvknrc.so")
            for filename in CORNELL_FILES:
                asset = Path("assets") / "cornell-box" / filename
                archive.write(ROOT / asset, asset.as_posix())
        aligned = stage / "aligned.apk"
        run(align, "-P", "16", "4", unsigned, aligned)
        signed = stage / "signed.apk"
        run(java, "-jar", signer, "sign", "--ks", keystore, "--ks-key-alias", "androiddebugkey",
            "--ks-pass", "pass:android", "--key-pass", "pass:android", "--out", signed, aligned)
        run(java, "-jar", signer, "verify", signed)
        run(align, "-c", "-P", "16", "4", signed)
        os.replace(signed, apk)
    print(f"Built {apk}\nInstall: adb install -r \"{apk}\"\nLaunch: adb shell am start -n org.vknrc.demo/.LauncherActivity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abi", choices=["x86_64", "arm64-v8a"], default="x86_64",
                        help="x86_64 for the emulator (default), arm64-v8a for a phone")
    parser.add_argument("--sdk", default=os.getenv("ANDROID_HOME") or os.getenv("ANDROID_SDK_ROOT"),
                        help="SDK path (default ANDROID_HOME); requires platform 35 and build-tools 35.0.0")
    parser.add_argument("--ndk", default=os.getenv("ANDROID_NDK_HOME"), help=f"NDK path (default SDK/ndk/{NDK})")
    parser.add_argument("--jdk", default=os.getenv("JAVA_HOME"), help="JDK 17 path (default JAVA_HOME or PATH)")
    parser.add_argument("--glslc", help="host glslc executable supporting GL_KHR_cooperative_matrix (default PATH)")
    parser.add_argument("--out", type=Path, default=ROOT / "build/android", help="output directory")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="parallel native build jobs")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        build(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Android build failed: {exc}")


if __name__ == "__main__":
    main()
