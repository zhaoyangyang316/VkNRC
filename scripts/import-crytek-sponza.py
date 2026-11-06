#!/usr/bin/env python3
"""Import the pinned McGuire Crytek Sponza archive for VkNRC (stdlib only).

Download the source archive separately, then run from the repository root:
python scripts/import-crytek-sponza.py --archive build/scene-import/crytek-sponza.zip

The OBJ, textures and original copyright notice are copied without changes.
Only the MTL paths/roughness and VkNRC camera/attribution/report are adapted.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import zipfile


ARCHIVE_URL = "https://casual-effects.com/g3d/data10/common/model/crytek_sponza/sponza.zip"
ARCHIVE_SHA256 = "da005cbee0be2df2abc8513f3ceb61bcb6f69aac112babcd9c00169a27c2770c"
RAW_CAMERA_POSITION = (1000.0, 180.0, 0.0)
RAW_CAMERA_TARGET = (-1000.0, 220.0, 0.0)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(p in ("..", ".") or ":" in p for p in path.parts):
        raise ValueError(f"Unsafe archive or texture path: {name}")
    return path.as_posix()


def load_archive(path):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"Archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual}")
    files = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            name = safe_relative(item.filename)
            if item.is_dir():
                continue
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"Archive symlink is not supported: {name}")
            if name in files:
                raise ValueError(f"Duplicate archive file: {name}")
            if name not in ("sponza.obj", "sponza.mtl", "copyright.txt") and not (
                name.startswith("textures/") and name.endswith(".png")
            ):
                raise ValueError(f"Unexpected source file: {name}")
            files[name] = archive.read(item)  # ZipFile also verifies each CRC.
    if not {"sponza.obj", "sponza.mtl", "copyright.txt"} <= files.keys():
        raise ValueError("Archive is missing its OBJ, MTL or original copyright notice")
    return files


def read_materials(data, files):
    materials = {}
    current = None
    output = ["# VkNRC adaptation: portable texture paths and explicit Pr = sqrt(2 / (Ns + 2))."]
    for line in data.decode("utf-8-sig").splitlines():
        fields = line.partition("#")[0].split()
        if fields:
            key = fields[0]
            if key == "newmtl":
                name = " ".join(fields[1:])
                if not name or name in materials:
                    raise ValueError(f"Invalid or duplicate material: {name}")
                current = {"Ns": None, "Pr": None, "Ke": [0.0, 0.0, 0.0], "textures": {}}
                materials[name] = current
            elif current is not None:
                if key == "Ns":
                    ns = float(fields[1])
                    if not math.isfinite(ns) or ns < 0 or current["Ns"] is not None:
                        raise ValueError("Material has invalid or duplicate Ns")
                    current["Ns"] = ns
                    current["Pr"] = math.sqrt(2.0 / (ns + 2.0))
                elif key == "Pr":
                    raise ValueError("Source already has roughness; review the conversion")
                elif key == "Ke":
                    current["Ke"] = list(map(float, fields[1:]))
                    if len(current["Ke"]) != 3 or not all(math.isfinite(v) and v >= 0 for v in current["Ke"]):
                        raise ValueError("Invalid emission value")
                elif key.startswith("map_") or key in ("bump", "disp", "decal"):
                    if len(fields) != 2:
                        raise ValueError(f"Review unsupported texture options: {line}")
                    texture = safe_relative(fields[1])
                    if texture not in files:
                        raise ValueError(f"Missing texture: {texture}")
                    current["textures"][key] = texture
                    line = line.replace("\\", "/")
        output.append(line)
        if fields and fields[0] == "Ns":
            output.append(f"\tPr {current['Pr']:.12g}")
    if not materials or any(mat["Pr"] is None for mat in materials.values()):
        raise ValueError("All source materials must have Ns for roughness conversion")
    return materials, ("\n".join(output) + "\n").encode("utf-8")


def read_geometry(data, materials):
    counts = Counter()
    assigned = Counter()
    bounds_min = [math.inf] * 3
    bounds_max = [-math.inf] * 3
    material = None
    libraries = []
    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        key = fields[0]
        if key in ("v", "vt", "vn"):
            counts[key] += 1
            xyz = list(map(float, fields[1:]))
            if not all(math.isfinite(v) for v in xyz):
                raise ValueError(f"Nonfinite vertex attribute at line {line_number}")
            if key == "v":
                if len(xyz) != 3:
                    raise ValueError(f"Unexpected position at line {line_number}")
                bounds_min = [min(a, b) for a, b in zip(bounds_min, xyz)]
                bounds_max = [max(a, b) for a, b in zip(bounds_max, xyz)]
        elif key == "mtllib":
            libraries.extend(fields[1:])
        elif key == "usemtl":
            material = " ".join(fields[1:])
            if material not in materials:
                raise ValueError(f"Unknown material {material} at line {line_number}")
        elif key == "f":
            if material is None or len(fields) not in (4, 5):
                raise ValueError(f"Unassigned or unsupported face at line {line_number}")
            counts["polygons"] += 1
            counts[f"{len(fields) - 1}_vertex_faces"] += 1
            triangles = len(fields) - 3
            counts["triangles"] += triangles
            assigned[material] += triangles
            for token in fields[1:]:
                indices = token.split("/")
                if len(indices) != 3 or any(not i for i in indices):
                    raise ValueError(f"Missing face attribute index at line {line_number}")
                for index, kind in zip(indices, ("v", "vt", "vn")):
                    value = int(index)
                    resolved = value - 1 if value > 0 else counts[kind] + value
                    if value == 0 or not 0 <= resolved < counts[kind]:
                        raise ValueError(f"Invalid {kind} index at line {line_number}")
    if libraries != ["sponza.mtl"] or not counts["triangles"]:
        raise ValueError("Unexpected material library or empty geometry")
    center = [(a + b) * 0.5 for a, b in zip(bounds_min, bounds_max)]
    scale = 2.0 / max(b - a for a, b in zip(bounds_min, bounds_max))
    return counts, assigned, bounds_min, bounds_max, center, scale


def camera_data(center, scale):
    normalize = lambda position: [(v - c) * scale for v, c in zip(position, center)]
    position, target = normalize(RAW_CAMERA_POSITION), normalize(RAW_CAMERA_TARGET)
    text = (
        "# VkNRC camera preset; authored for the central atrium, Y up.\n"
        "# Raw OBJ position 1000 180 0, target -1000 220 0.\n"
        "# Normalized as (raw position - AABB center) * (2 / max(AABB extent)).\n"
        "name Crytek Sponza\n"
        f"position {' '.join(f'{v:.12g}' for v in position)}\n"
        f"target {' '.join(f'{v:.12g}' for v in target)}\n"
        "fov 60\n"
    )
    return text.encode("utf-8"), {
        "raw_position": RAW_CAMERA_POSITION, "raw_target": RAW_CAMERA_TARGET,
        "normalized_position": position, "normalized_target": target, "vertical_fov_degrees": 60,
    }


ATTRIBUTION = f"""Crytek Sponza
Copyright (c) 2010 Frank Meinl, Crytek.

Licensed under Creative Commons Attribution 3.0 Unported (CC BY 3.0),
as listed in Morgan McGuire's Computer Graphics Archive:
https://creativecommons.org/licenses/by/3.0/
https://creativecommons.org/licenses/by/3.0/legalcode

Archive: https://casual-effects.com/data/
Model metadata: https://casual-effects.com/g3d/data10/common/model/crytek_sponza/info.js
Download: {ARCHIVE_URL}
Source ZIP SHA-256: {ARCHIVE_SHA256}

Frank Meinl at Crytek remodeled the Sponza Palace atrium, inspired by Marko
Dabrovic's original model. Morgan McGuire prepared this OBJ/PNG archive,
including material corrections, bump maps derived from normal maps, and
cleaned opacity masks stored in diffuse alpha. See copyright.txt for the
unmodified original notice and archive history.

Archive citation: Morgan McGuire, Computer Graphics Archive, July 2017,
https://casual-effects.com/data

VkNRC adaptations:
- The OBJ, all 54 PNG textures, and copyright.txt are unchanged from the ZIP.
- MTL texture path separators are changed from backslashes to forward slashes.
- Each material gains explicit Beckmann roughness Pr = sqrt(2 / (Ns + 2)).
  This approximates its Phong exponent for VkNRC; it is not an exact BSDF match.
- A normalized camera preset is added for the central atrium.
- No light geometry or emission is added. All 25 source materials have Ke 0
  and no map_Ke. VkNRC's existing white environment at intensity 10 illuminates
  the scene, and next event estimation samples that environment.

Current renderer limitations:
VkNRC treats all triangles as opaque. The source diffuse alpha and map_d
opacity masks are preserved but not evaluated, so leaves and chains retain
their rectangular polygon silhouettes. Bump maps and OBJ vertex normals are
also preserved but ignored; shading uses geometric triangle normals. The
renderer uses its existing Beckmann reflection model without transmission.
Texture maps replace the corresponding constant material colors. These
limitations mean this is not an exact reproduction of Crytek's source renders.

Reproduce the import from the repository root (Python 3.11+, standard library):
python scripts/import-crytek-sponza.py --archive build/scene-import/crytek-sponza.zip

You may share and adapt the licensed material, including commercially, provided
you give appropriate credit, link the license and indicate modifications.
No endorsement by the original authors is implied. The license governs.
"""


def import_scene(archive_path, output):
    files = load_archive(archive_path)
    materials, adapted_mtl = read_materials(files["sponza.mtl"], files)
    counts, assigned, lower, upper, center, scale = read_geometry(files["sponza.obj"], materials)
    camera, camera_report = camera_data(center, scale)
    emissive = [name for name, mat in materials.items()
                if name in assigned and (any(v > 0 for v in mat["Ke"]) or "map_Ke" in mat["textures"])]
    if emissive:
        raise ValueError("Source lighting changed; review lighting and attribution before import")
    textures = {}
    for name, data in files.items():
        if not name.endswith(".png"):
            continue
        if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
            raise ValueError(f"Invalid PNG header: {name}")
        width, height = struct.unpack(">II", data[16:24])
        if width < 1 or height < 1:
            raise ValueError(f"Invalid PNG dimensions: {name}")
        textures[name] = {"width": width, "height": height, "bytes": len(data), "sha256": digest(data)}
    if len(materials) != 25 or len(textures) != 54 or counts["triangles"] != 262267:
        raise ValueError("Source counts differ from the documented archive")
    references = {path for mat in materials.values() for path in mat["textures"].values()}
    report = {
        "source_url": ARCHIVE_URL, "source_archive_sha256": ARCHIVE_SHA256,
        "source_files_sha256": {name: digest(files[name]) for name in ("sponza.obj", "sponza.mtl", "copyright.txt")},
        "adapted_mtl_sha256": digest(adapted_mtl),
        "geometry": {
            "positions": counts["v"], "texcoords": counts["vt"], "normals": counts["vn"],
            "polygons": counts["polygons"], "triangle_faces": counts["3_vertex_faces"],
            "quad_faces": counts["4_vertex_faces"], "triangles_after_triangulation": counts["triangles"],
            "all_face_indices_valid": True, "all_faces_have_uvs_normals_and_materials": True,
            "bounds_min": lower, "bounds_max": upper, "normalization_center": center,
            "normalization_scale": scale, "triangles_by_material": dict(sorted(assigned.items())),
        },
        "camera": camera_report,
        "lighting": {
            "assigned_emissive_materials": emissive, "emissive_triangles": 0,
            "added_lights": 0, "environment_intensity": 10,
            "note": "Original materials have no Ke or map_Ke emission. VkNRC uses its existing constant environment.",
        },
        "materials": materials, "textures": textures,
        "texture_checks": {"all_referenced_textures_present": True, "unique_referenced_textures": len(references),
                           "all_54_source_pngs_preserved": True},
        "limitations": ["Opaque alpha and map_d masks", "Bump maps and OBJ vertex normals ignored",
                        "Phong-to-Beckmann roughness approximation"],
    }
    # Validate everything before writing, and never follow a destination symlink
    # outside the selected asset directory. Reruns only replace generated members.
    output = output.resolve()
    generated = dict(files)
    generated.update({"sponza.mtl": adapted_mtl, "sponza.camera": camera,
                      "ATTRIBUTION.txt": ATTRIBUTION.encode("utf-8"),
                      "import-report.json": (json.dumps(report, indent=2) + "\n").encode("utf-8")})
    for name in generated:
        target = (output / name).resolve()
        if not target.is_relative_to(output):
            raise ValueError(f"Destination escapes selected asset directory: {name}")
    for name, data in generated.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f"Imported Crytek Sponza to {output}")
    print(f"{counts['v']} positions, {counts['triangles']} triangles, {len(materials)} materials, {len(textures)} textures")
    print("All face indices and texture references verified; no emitters; environment intensity 10.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True, help="Downloaded, unmodified Crytek Sponza ZIP")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "crytek-sponza")
    args = parser.parse_args()
    try:
        import_scene(args.archive, args.output)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Import failed: {error}\n")


if __name__ == "__main__":
    main()
