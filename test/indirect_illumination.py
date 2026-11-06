#!/usr/bin/env python3
"""Sequential GPU regression for --indirect-only; Python 3 standard library only.

Run after building Release: python test/indirect_illumination.py --validation
For the fast-mode primary/secondary regression add --fast-gi (640x360, one warmup frame).
Each renderer launch is bounded; captures and combined stdout/stderr remain in --out.
"""
import argparse
from array import array
import json
import math
import os
from pathlib import Path
import re
import struct
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


def write_solid_tga(path, rgb):
    # One true-color, uncompressed pixel. TGA stores BGR; no image library needed.
    header = struct.pack("<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 24, 0x20)
    path.write_bytes(header + bytes(reversed(rgb)))


def secondary_fixtures(directory):
    # The camera looks along -Z from between two facing quads. At a depth of 3,
    # only the back quad can supply cached indirect light: the primary must be
    # traced and the third hit has already reached the depth limit.
    meshes = {}
    write_solid_tga(directory / "specular-black.tga", (0, 0, 0))
    # A single channel at or above .01 must retain the material roughness.
    write_solid_tga(directory / "specular-red.tga", (128, 0, 0))
    # 24/255 in sRGB becomes approximately .00913 in linear space. Testing the
    # encoded texel rather than the sampled linear value would reject this map.
    write_solid_tga(directory / "specular-weak.tga", (24, 24, 24))
    cases = (
        ("smooth", .25, ".2 .2 .2", None),
        ("equal", .5, ".2 .2 .2", None),
        ("rough", .75, ".2 .2 .2", None),
        ("diffuse", .05, "0 0 0", None),
        ("weak-specular", .05, ".009 .009 .009", None),
        ("specular-equal", .05, ".01 0 0", None),
        ("specular-above", .05, ".0101 0 0", None),
        ("mapped-diffuse", .05, ".2 .2 .2", "specular-black.tga"),
        ("mapped-weak-specular", .05, ".2 .2 .2", "specular-weak.tga"),
        ("mapped-specular", .05, "0 0 0", "specular-red.tga"),
    )
    for name, roughness, specular, texture in cases:
        path = directory / f"secondary-{name}.obj"
        path.write_text(f"mtllib {path.stem}.mtl\n"
                        "v -1 -1 -.25\nv 1 -1 -.25\nv 1 1 -.25\nv -1 1 -.25\n"
                        "v -1 -1 .25\nv 1 -1 .25\nv 1 1 .25\nv -1 1 .25\nvt 0 0\n"
                        "usemtl primary\nf 1/1 2/1 3/1\nf 1/1 3/1 4/1\n"
                        "usemtl secondary\nf 5/1 8/1 7/1\nf 5/1 7/1 6/1\n", encoding="ascii")
        material_text = (
            "newmtl primary\nKd .7 .7 .7\nKs 0 0 0\nKe 0 0 0\nNi 1.5\nPr 1\n"
            f"newmtl secondary\nKd .7 .7 .7\nKs {specular}\nKe 0 0 0\nNi 1.5\nPr {roughness}\n")
        if texture:
            # The loaded map replaces Ks, including an all-black map.
            material_text += f"map_Ks {texture}\n"
        path.with_suffix(".mtl").write_text(material_text, encoding="ascii")
        meshes[path.stem] = path
    return meshes


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=root / "build/Release/VkNRC.exe")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--samples", type=int, choices=(4, 64, 128))
    parser.add_argument("--fast-gi", action="store_true", help="Check explicit primary transport and secondary roughness gating")
    parser.add_argument("--timeout", type=float, default=180, help="Seconds allowed per renderer launch")
    parser.add_argument("--validation", action="store_true", help="Enable VK_LAYER_KHRONOS_validation")
    args = parser.parse_args()
    args.out = args.out or root / "build" / ("fast-gi" if args.fast_gi else "indirect-illumination")
    args.samples = args.samples or (4 if args.fast_gi else 64)
    args.exe, args.out = args.exe.resolve(), args.out.resolve()
    require(args.exe.is_file(), f"Renderer missing: {args.exe}")
    require(args.timeout > 0 and math.isfinite(args.timeout), "--timeout must be positive and finite")
    args.out.mkdir(parents=True, exist_ok=True)
    scenes = fixtures(args.out / "fixtures")
    if args.fast_gi:
        scenes.update(secondary_fixtures(args.out / "fixtures"))
    scenes["cornell"] = root / "assets/cornell-box/cornell-box.obj"
    require(scenes["cornell"].is_file(), "Cornell fixture is missing")
    environment = os.environ.copy()
    if args.validation:
        environment["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    results = {}
    dimensions = None

    def capture(scene, depth, indirect, nee, threshold=.5):
        nonlocal dimensions
        label = f"{scene}-d{depth}-{'indirect' if indirect else 'full'}-{'nee' if nee else 'bsdf'}"
        if args.fast_gi:
            label += f"-fast-{threshold:g}"
        output = args.out / label
        output.mkdir(exist_ok=True)
        for filename in ("nrc.pfm", "path-tracing.pfm"):
            (output / filename).unlink(missing_ok=True)
        command = [str(args.exe), str(scenes[scene]), "--compare", str(output), "--warmup", "1" if args.fast_gi else "64",
                   "--samples", str(args.samples), "--seed", "1337", "--max-bounces", str(depth)]
        if args.fast_gi:
            command += ["--fast-mode", "--min-roughness", str(threshold), "--resolution", "640x360"]
        if scene.startswith("secondary-"):
            command += ["--camera", "0", "0", "0", "--fov", "60"]
        elif scene != "cornell":
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
        if args.fast_gi:
            require(f"NRC fast mode: enabled, min roughness: {threshold:g}" in log, f"{label}: unexpected mode")
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

    if args.fast_gi:
        def explicit(images, label):
            require(images["nrc"] == images["pt"], f"{label}: NRC diverged from explicit PT")

        def cached(images, label):
            # Cold weights make a lookup observable; require differences across a substantial region.
            changed = sum(abs(a - b) > .001 for a, b in zip(images["nrc"], images["pt"]))
            require(changed > .005 * len(images["pt"]), f"{label}: only {changed} RGB elements differ from PT")

        for nee in (True, False):
            plane = capture("environment", 3, False, nee)
            explicit(plane, "eligible primary plane")
            w, h = dimensions
            center = 3 * ((h // 2) * w + w // 2)
            require(abs(plane["pt"][center] - 7) < .003, "Expected direct radiance 10 * .7 on the plane")
            black(capture("environment", 3, True, nee), "plane has no indirect illumination")
            direct = capture("cornell", 2, False, nee)
            explicit(direct, "Cornell direct lighting")
            if nee:
                full = capture("cornell", 64, False, nee)
                gi = capture("cornell", 64, True, nee)
                cached(full, "full Cornell")
                cached(gi, "indirect Cornell")
                error = max(abs(f - i - d) for f, i, d in zip(full["pt"], gi["pt"], direct["pt"]))
                require(error <= 5e-5, f"Cornell PT full - indirect - direct max error {error}")
                for indirect in (False, True):
                    explicit(capture("cornell", 64, indirect, nee, 1), "Cornell upper roughness cutoff")

        diffuse_like_scenes = ("secondary-diffuse", "secondary-weak-specular",
                               "secondary-mapped-diffuse", "secondary-mapped-weak-specular")
        for scene in ("secondary-smooth", "secondary-equal", "secondary-rough",
                      *diffuse_like_scenes, "secondary-specular-equal", "secondary-specular-above",
                      "secondary-mapped-specular"):
            for indirect in (False, True):
                image = capture(scene, 3, indirect, True)
                require(sum(image["pt"]) / len(image["pt"]) > .01, f"{scene}: insufficient illumination")
                if scene == "secondary-rough" or scene in diffuse_like_scenes:
                    cached(image, scene)
                else:
                    explicit(image, scene)
                if scene in diffuse_like_scenes:
                    # Effective roughness 1 still fails a strictly-greater-than
                    # cutoff of 1. This must not change baseline path tracing.
                    cutoff = capture(scene, 3, indirect, True, 1)
                    explicit(cutoff, f"{scene}: effective upper roughness cutoff")
                    require(image["pt"] == cutoff["pt"], f"{scene}: threshold changed baseline PT")
        print(f"PASS: {len(results)} fast GI captures; primary light, strict secondary cutoff, "
              "strict specular cutoff, linear texture handling and depth constraints passed.")
        print(f"Logs, fixtures, PFM images, and statistics: {args.out}")
        return

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
