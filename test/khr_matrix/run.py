"""Build a small Vulkan runner, execute production KHR kernels and compare with NumPy."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parent
SHADERS = SOURCE.parents[1] / "shader" / "src"
BUILD = SOURCE.parents[1] / "build" / "khr-matrix-validation"
BUILD.mkdir(parents=True, exist_ok=True)


def reference():
    # Sparse shifted weights exercise row/column orientation, negative ReLU gates
    # and exact binary fractions. All reference math below is ordinary FP32.
    width, samples = 64, 256
    rng = np.random.default_rng(7331)
    x = (rng.integers(-8, 16, (samples, width)) / 16).astype(np.float16)
    target = (rng.integers(0, 9, (samples, 4)) / 16).astype(np.float32)
    weights = []
    rows = np.arange(width)
    for layer in range(5):
        matrix = np.zeros((width, width), np.float16)
        columns = (rows + 2 * layer + 1) % width
        matrix[rows, columns] = np.where(rows % 19 == 0, -1.0, 1.0)
        weights.append(matrix)
    output = np.zeros((3, width), np.float16)
    for c in range(3):
        output[c, c::7] = .125 * (1 + c)
    weights.append(output)
    np.concatenate([matrix.ravel() for matrix in weights]).tofile(BUILD / "weights.bin")
    x.tofile(BUILD / "input.bin")
    target.tofile(BUILD / "target.bin")

    activations = [x.astype(np.float32)]
    for matrix in weights[:-1]:
        activations.append(np.maximum(activations[-1] @ matrix.T, 0))
    rgb = activations[-1] @ output.T
    luminance = np.maximum(rgb, 0) @ np.array([.299, .587, .114], np.float32)
    grad = (2 / 128) * (rgb - target[:, :3]) / (luminance[:, None] ** 2 + .01)
    grad = grad.astype(np.float16).astype(np.float32)
    gradients = []
    for layer in reversed(range(6)):
        gradients.append(grad.T @ activations[layer])
        if layer:
            grad = (grad @ weights[layer]) * (activations[layer] > 0)
            grad = grad.astype(np.float16).astype(np.float32)
    return rgb, np.concatenate([matrix.ravel() for matrix in reversed(gradients)])


def main():
    expected_rgb, expected_grad = reference()
    runner_build = BUILD / "host"
    subprocess.run(["cmake", "-S", SOURCE, "-B", runner_build], check=True)
    subprocess.run(["cmake", "--build", runner_build, "--config", "Release"], check=True)
    filename = "khr_matrix_check" + (".exe" if os.name == "nt" else "")
    executable = next(runner_build.rglob(filename))
    passed = 0
    for stage in ("inference", "gradient"):
        for precision in ("f16", "f32", "scalar"):
            name = f"{stage}_{precision}"
            flags = ["-DNN_BACKPROPAGATION=1"] if stage == "gradient" else []
            if precision == "f32":
                flags.append("-DNRC_KHR_FP32=1")
            elif precision == "scalar":
                flags.extend(("-DNRC_SCALAR=1", "-O"))
            spirv = BUILD / f"{name}.spv"
            subprocess.run(["glslc", "--target-env=vulkan1.3", f"-I{SHADERS}", *flags,
                            SOURCE / "check.comp", "-o", spirv], check=True)
            subprocess.run(["spirv-val", "--target-env", "vulkan1.3", spirv], check=True)
            result = subprocess.run([executable, spirv, precision], cwd=BUILD)
            if result.returncode == 77:
                print(f"SKIP {name}: no compatible GPU matrix tile")
                continue
            result.check_returncode()
            rgb = np.fromfile(BUILD / "prediction.bin", np.float32).reshape(-1, 4)[:, :3]
            if not np.isfinite(rgb).all() or not np.array_equal(rgb, expected_rgb):
                raise RuntimeError(f"{name}: RGB predictions differ from CPU reference")
            if stage == "gradient":
                actual = np.fromfile(BUILD / "gradient.bin", np.float32)
                if actual.shape != expected_grad.shape or not np.isfinite(actual).all():
                    raise RuntimeError(f"{name}: invalid gradient buffer")
                relative_l2 = np.linalg.norm(actual - expected_grad) / np.linalg.norm(expected_grad)
                if relative_l2 >= .003:
                    raise RuntimeError(f"{name}: gradient relative L2 {relative_l2} exceeds tolerance")
                np.testing.assert_allclose(actual, expected_grad, rtol=.005, atol=.001, err_msg=name)
                print(f"{name} gradient relative L2 {relative_l2:.8g}")
            print(f"PASS {name}: all {len(rgb)} RGB records")
            passed += 1
    return 0 if passed else 77


if __name__ == "__main__":
    sys.exit(main())
