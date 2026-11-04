#!/usr/bin/env python3
"""Sequential GPU regression for --indirect-only; Python 3 standard library only.

Run after building Release: python test/indirect_illumination.py --validation
Each renderer launch is bounded; captures and combined stdout/stderr remain in --out.
"""
import argparse
from array import array
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_pfm(path):
    with path.open("rb") as stream:
        require(stream.readline().strip() == b"PF", f"{path}: expected RGB PFM")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        payload = stream.read()
    require(width > 0 and height > 0 and math.isfinite(scale) and scale != 0,
            f"{path}: invalid PFM header")
    require(len(payload) == width * height * 3 * 4, f"{path}: wrong payload size")
    values = array("f")
    require(values.itemsize == 4, "This test requires 32-bit Python array floats")
    values.frombytes(payload)
    if (scale < 0) != (sys.byteorder == "little"):
        values.byteswap()
    if abs(scale) != 1:
        values = array("f", (v * abs(scale) for v in values))
    require(all(map(math.isfinite, values)), f"{path}: nonfinite radiance")
    return (width, height), values


def fixtures(directory):
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, diffuse, emission in (("environment", .7, 0), ("emitter", 0, 3)):
        path = directory / f"{name}.obj"
        path.write_text(f"mtllib {name}.mtl\nv -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\n"
                        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl surface\n"
                        "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n", encoding="ascii")
        path.with_suffix(".mtl").write_text(
            f"newmtl surface\nKd {diffuse} {diffuse} {diffuse}\nKs 0 0 0\n"
            f"Ke {emission} {emission} {emission}\nNi 1.5\nPr 1\n", encoding="ascii")
        result[name] = path
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=root / "build/Release/VkNRC.exe")
    parser.add_argument("--out", type=Path, default=root / "build/indirect-illumination")
    parser.add_argument("--samples", type=int, choices=(64, 128), default=64)
    parser.add_argument("--timeout", type=float, default=180, help="Seconds allowed per renderer launch")
    parser.add_argument("--validation", action="store_true", help="Enable VK_LAYER_KHRONOS_validation")
    args = parser.parse_args()
    args.exe, args.out = args.exe.resolve(), args.out.resolve()
    require(args.exe.is_file(), f"Renderer missing: {args.exe}")
    require(args.timeout > 0 and math.isfinite(args.timeout), "--timeout must be positive and finite")
    args.out.mkdir(parents=True, exist_ok=True)
    scenes = fixtures(args.out / "fixtures")
    scenes["cornell"] = root / "assets/cornell-box/cornell-box.obj"
    require(scenes["cornell"].is_file(), "Cornell fixture is missing")
    environment = os.environ.copy()
    if args.validation:
        environment["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    results = {}
    dimensions = None

    def capture(scene, depth, indirect, nee):
        nonlocal dimensions
        label = f"{scene}-d{depth}-{'indirect' if indirect else 'full'}-{'nee' if nee else 'bsdf'}"
        output = args.out / label
        output.mkdir(exist_ok=True)
        for filename in ("nrc.pfm", "path-tracing.pfm"):
            (output / filename).unlink(missing_ok=True)
        command = [str(args.exe), str(scenes[scene]), "--compare", str(output), "--warmup", "64",
                   "--samples", str(args.samples), "--seed", "1337", "--max-bounces", str(depth)]
        if scene != "cornell":
            # Include both the quad and background to exercise primary misses.
            command += ["--camera", "0", "0", "3", "--fov", "60"]
        if indirect:
            command += ["--indirect-only"]
        if not nee:
            command += ["--no-nee"]
        (output / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        print(f"Running {label}", flush=True)
        log_path = output / "renderer.log"
        with log_path.open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(command, cwd=root, env=environment, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"{label}: timed out; see {log_path}") from error
        require(process.returncode == 0, f"{label}: exit {process.returncode}; see {log_path}")
        log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log_path.read_text(encoding="utf-8", errors="replace"))
        require(not re.search(r"VUID-|Validation Error|VK_ERROR_|\[(?:error|critical)\]", log, re.I),
                f"{label}: renderer/validation error; see {log_path}")
        images, stats = {}, {}
        for method, filename in (("nrc", "nrc.pfm"), ("pt", "path-tracing.pfm")):
            require(filename in log and "Captured" in log, f"{label}: capture not confirmed in log")
            size, values = read_pfm(output / filename)
            dimensions = size if dimensions is None else dimensions
            require(size == dimensions, f"{label}: capture dimensions changed ({size} vs {dimensions})")
            images[method] = values
            stats[method] = {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
            require(stats[method]["min"] >= 0, f"{label}/{method}: negative radiance")
        results[label] = stats
        (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return images

    def black(images, label):
        for method, values in images.items():
            require(not any(values), f"{label}/{method}: expected exactly black; max={max(values):.9g}")

    def positive(images, label):
        for method, values in images.items():
            require(sum(values) / len(values) > 1e-5, f"{label}/{method}: expected positive radiance")

    # Frozen PT uses identical random sequences across modes. NRC warmup/cache
    # paths can differ, so exact decomposition is intentionally tested only on PT.
    for nee in (True, False):
        full = capture("cornell", 64, False, nee)["pt"]
        indirect_images = capture("cornell", 64, True, nee)
        positive(indirect_images, "Cornell depth 64 indirect")
        indirect = indirect_images["pt"]
        direct = capture("cornell", 2, False, nee)["pt"]
        errors = [max(abs(full[i] - indirect[i] - direct[i]) for i in range(c, len(full), 3))
                  for c in range(3)]
        require(max(errors) <= 5e-5, f"Cornell decomposition NEE={nee}: RGB max error {errors}")
        print(f"  PT full - indirect = direct, RGB max error: {errors}", flush=True)
        del full, indirect_images, indirect, direct
        black(capture("cornell", 2, True, nee), "Cornell depth 2 indirect")
        positive(capture("cornell", 3, True, nee), "Cornell depth 3 indirect")
        for scene in ("environment", "emitter"):
            positive(capture(scene, 64, False, nee), f"{scene} full")
            black(capture(scene, 64, True, nee), f"{scene} indirect")
    black(capture("cornell", 1, True, True), "Cornell depth 1 indirect")
    print(f"PASS: {len(results)} sequential captures; NRC/PT finite, decomposition and isolation checks passed.")
    print(f"Logs, fixtures, PFM images, and statistics: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
