#!/usr/bin/env python3
"""Import Bitterli's CC0 Mitsuba Veach Ajar scene for VkNRC (stdlib only).

python scripts/import-veach-door.py --source build/scene-import/veach-source/veach-ajar
The XML, source meshes, and LICENSE.txt must come from the unmodified archive:
https://benedikt-bitterli.me/resources/mitsuba/veach-ajar.zip
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import xml.etree.ElementTree as ET
import zlib


IDENTITY = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)


def numbers(text):
    return tuple(float(x) for x in text.replace(",", " ").split())


def value(element, tag, name, default=None):
    child = element.find(f"{tag}[@name='{name}']")
    return default if child is None else child.attrib["value"]


def rgb(element, name, default=(0, 0, 0)):
    result = value(element, "rgb", name)
    return default if result is None else numbers(result)


def matrix(element):
    node = element.find("transform[@name='to_world']/matrix")
    return IDENTITY if node is None else numbers(node.attrib["value"])


def point(transform, p):
    return tuple(sum(transform[row * 4 + col] * p[col] for col in range(3)) +
                 transform[row * 4 + 3] for row in range(3))


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def length(p):
    return math.sqrt(sum(x * x for x in p))


def fmt(values):
    return " ".join(f"{x:.12g}" for x in values)


def read_obj(path):
    vertices, texcoords, faces = [], [], []
    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append(tuple(map(float, fields[1:4])))
            elif fields[0] == "vt":
                texcoords.append(tuple(map(float, fields[1:3])))
            elif fields[0] == "f":
                face = []
                for token in fields[1:]:
                    indices = token.split("/")
                    vi = int(indices[0])
                    vi = vi - 1 if vi > 0 else len(vertices) + vi
                    ti = None
                    if len(indices) > 1 and indices[1]:
                        ti = int(indices[1])
                        ti = ti - 1 if ti > 0 else len(texcoords) + ti
                    if not 0 <= vi < len(vertices) or (ti is not None and not 0 <= ti < len(texcoords)):
                        raise ValueError(f"Invalid index at {path}:{line_number}")
                    face.append((vi, ti))
                if len(face) < 3:
                    raise ValueError(f"Invalid face at {path}:{line_number}")
                # Source meshes contain triangles; permit ordinary convex OBJ polygons.
                faces.extend((face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1))
    return vertices, texcoords, faces


def srgb_byte(linear):
    encoded = 12.92 * linear if linear <= .0031308 else 1.055 * linear ** (1 / 2.4) - .055
    return max(0, min(255, round(encoded * 255)))


def write_checker(path, color0, color1, tint):
    # A PNG stores one 2x2-check period. 256 pixels keep bilinear edge blur narrow.
    # VkNRC negates OBJ v before sampling a top-down image, so the bottom-left
    # check is color0. Matches Mitsuba checkerboard eval(), away from boundaries:
    # https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/textures/checkerboard.cpp
    size = 256
    colors = [bytes(srgb_byte(c * t) for c, t in zip(color, tint)) for color in (color0, color1)]
    rows = bytearray()
    for y in range(size):
        rows.append(0)  # PNG row filter: none.
        for x in range(size):
            rows.extend(colors[(x >= size // 2) == (y >= size // 2)])

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                     chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b""))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_scene(source, output):
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "textures").mkdir(exist_ok=True)
    xml_path = source / "scene_v3.xml"
    scene = ET.parse(xml_path).getroot()
    inputs = {"scene_v3.xml": sha256(xml_path), "LICENSE.txt": sha256(source / "LICENSE.txt")}
    materials, uv_scales, material_report = {}, {}, {}
    for wrapper in scene.findall("bsdf"):
        name = wrapper.attrib["id"]
        bsdf = wrapper.find("bsdf") if wrapper.attrib["type"] == "twosided" else wrapper
        kind = bsdf.attrib["type"]
        kd, ks, ior, roughness = (0, 0, 0), (0, 0, 0), 1.5, 1.0
        maps = []
        report = {"source_type": kind}
        if kind == "diffuse":
            kd = rgb(bsdf, "reflectance")
            texture = bsdf.find("texture[@name='reflectance']")
            if texture is not None:
                if texture.attrib["type"] != "bitmap":
                    raise ValueError(f"Unsupported diffuse texture: {name}")
                original = value(texture, "string", "filename")
                destination = f"textures/{name.lower()}" + Path(original).suffix.lower()
                shutil.copyfile(source / original, output / destination)
                inputs[original] = sha256(source / original)
                maps.append(("map_Kd", destination))
                kd = (1, 1, 1)  # VkNRC uses the bitmap instead of the constant.
        elif kind == "roughconductor":
            eta, k = rgb(bsdf, "eta"), rgb(bsdf, "k")
            f0 = tuple(((e - 1) ** 2 + kk ** 2) / ((e + 1) ** 2 + kk ** 2)
                       for e, kk in zip(eta, k))
            # Renderer uses scalar Schlick Fresnel times Ks. Fit RGB F0 exactly
            # using its largest channel as scalar F0 and Ks <= 1 as the tint.
            scalar_f0 = max(f0)
            root_f0 = math.sqrt(scalar_f0)
            ior = (1 + root_f0) / (1 - root_f0)
            tint = tuple(channel / scalar_f0 for channel in f0)
            ks = tuple(c * t for c, t in zip(rgb(bsdf, "specular_reflectance", (1, 1, 1)), tint))
            roughness = float(value(bsdf, "float", "alpha"))
            report.update(source_f0=f0, fitted_ior=ior, source_distribution=value(bsdf, "string", "distribution"))
            texture = bsdf.find("texture[@name='specular_reflectance']")
            if texture is not None:
                if texture.attrib["type"] != "checkerboard":
                    raise ValueError(f"Unsupported conductor texture: {name}")
                destination = f"textures/{name.lower()}-checker.png"
                write_checker(output / destination, rgb(texture, "color0"), rgb(texture, "color1"), tint)
                maps.append(("map_Ks", destination))
                scale = texture.find("transform[@name='to_uv']/scale")
                uv_scales[name] = (float(scale.attrib["x"]), float(scale.attrib["y"]))
        elif kind == "dielectric":
            ior = float(value(bsdf, "float", "int_ior")) / float(value(bsdf, "float", "ext_ior"))
            ks, roughness = (1, 1, 1), .001
            report["limitation"] = "Opaque reflective dielectric approximation; no transmission or caustics."
        else:
            raise ValueError(f"Unsupported BSDF: {name}: {kind}")
        materials[name] = {"kd": kd, "ks": ks, "ior": ior, "roughness": roughness, "maps": maps, "ke": (0, 0, 0)}
        material_report[name] = report

    vertices, texcoords, groups, lights = [], [(0, 0)], [], []
    mesh_cache = {}
    fallback_uv_corners = 0
    removed_degenerate_triangles = 0
    for shape in scene.findall("shape"):
        name, kind = shape.attrib["id"], shape.attrib["type"]
        transform = matrix(shape)
        material = shape.find("ref").attrib["id"]
        if material not in materials:
            raise ValueError(f"Unknown material in {name}: {material}")
        if kind == "rectangle":
            local_vertices = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
            local_uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
            local_faces = [((0, 0), (1, 1), (2, 2)), ((0, 0), (2, 2), (3, 3))]
        elif kind == "obj":
            mesh_name = value(shape, "string", "filename")
            if mesh_name not in mesh_cache:
                mesh_cache[mesh_name] = read_obj(source / mesh_name)
                inputs[mesh_name] = sha256(source / mesh_name)
            local_vertices, local_uvs, local_faces = mesh_cache[mesh_name]
        else:
            raise ValueError(f"Unsupported shape: {name}: {kind}")
        world_vertices = [point(transform, v) for v in local_vertices]
        vo, to = len(vertices), len(texcoords)
        vertices.extend(world_vertices)
        scale = uv_scales.get(material, (1, 1))
        texcoords.extend((u * scale[0], v * scale[1]) for u, v in local_uvs)
        faces = []
        for face in local_faces:
            a, b, c = (world_vertices[v] for v, _ in face)
            if length(cross(subtract(b, a), subtract(c, a))) == 0:
                # Zero-area source triangles cannot change the visible geometry.
                # Avoid undefined normalized geometric normals in the renderer.
                removed_degenerate_triangles += 1
                continue
            faces.append(tuple((v + vo + 1, 1 if t is None else t + to + 1) for v, t in face))
            fallback_uv_corners += sum(t is None for _, t in face)
        emitter = shape.find("emitter")
        if emitter is not None:
            if emitter.attrib["type"] != "area":
                raise ValueError(f"Unsupported emitter in {name}")
            radiance = rgb(emitter, "radiance")
            material += "_Emission"
            materials[material] = dict(materials[shape.find("ref").attrib["id"]], ke=radiance)
            area = sum(length(cross(subtract(world_vertices[f[1][0]], world_vertices[f[0][0]]),
                                    subtract(world_vertices[f[2][0]], world_vertices[f[0][0]]))) * .5
                       for f in local_faces)
            face_normal = cross(subtract(world_vertices[1], world_vertices[0]),
                                subtract(world_vertices[2], world_vertices[0]))
            face_normal = tuple(x / length(face_normal) for x in face_normal)
            lights.append({"shape": name, "triangle_count": len(faces), "radiance": radiance,
                           "world_area": area, "world_vertices": world_vertices,
                           "world_normal": face_normal})
        groups.append((name, material, faces))

    minimum = tuple(min(p[i] for p in vertices) for i in range(3))
    maximum = tuple(max(p[i] for p in vertices) for i in range(3))
    center = tuple((a + b) / 2 for a, b in zip(minimum, maximum))
    normalization = 2 / max(b - a for a, b in zip(minimum, maximum))
    sensor = scene.find("sensor")
    camera_transform = matrix(sensor)
    position = point(camera_transform, (0, 0, 0))
    target = point(camera_transform, (0, 0, 1))  # Mitsuba perspective looks along local +Z.
    normalized_position = tuple((a - b) * normalization for a, b in zip(position, center))
    normalized_target = tuple((a - b) * normalization for a, b in zip(target, center))
    defaults = {node.attrib["name"]: node.attrib["value"] for node in scene.findall("default")}

    def resolved(number):
        return float(defaults[number[1:]] if number.startswith("$") else number)

    film = sensor.find("film")
    width, height = (resolved(value(film, "integer", name)) for name in ("width", "height"))
    horizontal_fov = float(value(sensor, "float", "fov"))
    if value(sensor, "string", "fov_axis", "x") != "x":
        raise ValueError("Expected a horizontal source camera field of view")
    vertical_fov = math.degrees(2 * math.atan(math.tan(math.radians(horizontal_fov) / 2) * height / width))

    with (output / "veach-door.obj").open("w", encoding="ascii", newline="\n") as file:
        file.write("# Veach, Ajar by Benedikt Bitterli (CC0). See ATTRIBUTION.txt and LICENSE.txt.\n")
        file.write("# Generated by scripts/import-veach-door.py from scene_v3.xml; world transforms baked.\n")
        file.write("mtllib veach-door.mtl\n")
        for p in vertices:
            file.write("v " + fmt(p) + "\n")
        for uv in texcoords:
            file.write("vt " + fmt(uv) + "\n")
        for name, material, faces in groups:
            file.write(f"g {name}\nusemtl {material}\n")
            for face in faces:
                file.write("f " + " ".join(f"{v}/{t}" for v, t in face) + "\n")
    with (output / "veach-door.mtl").open("w", encoding="ascii", newline="\n") as file:
        file.write("# VkNRC material approximations; see ATTRIBUTION.txt.\n")
        for name, material in materials.items():
            file.write(f"\nnewmtl {name}\nKd {fmt(material['kd'])}\nKs {fmt(material['ks'])}\n")
            file.write(f"Ke {fmt(material['ke'])}\nNi {material['ior']:.12g}\nPr {material['roughness']:.12g}\n")
            file.write("Pm 0\nd 1\nillum 2\n")
            for keyword, path in material["maps"]:
                file.write(f"{keyword} {path}\n")
    (output / "veach-door.camera").write_text(
        "# Original Mitsuba camera, mapped by VkNRC's centered 2/max_extent normalization.\n"
        "name Veach Door\n" + f"position {fmt(normalized_position)}\n" +
        f"target {fmt(normalized_target)}\nfov {vertical_fov:.12g}\n", encoding="ascii")
    shutil.copyfile(source / "LICENSE.txt", output / "LICENSE.txt")

    degenerate = sum(length(cross(subtract(vertices[f[1][0] - 1], vertices[f[0][0] - 1]),
                                  subtract(vertices[f[2][0] - 1], vertices[f[0][0] - 1]))) == 0
                     for _, _, faces in groups for f in faces)
    report = {"source": "https://benedikt-bitterli.me/resources/mitsuba/veach-ajar.zip",
              "input_sha256": dict(sorted(inputs.items())), "vertex_count": len(vertices),
              "texcoord_count": len(texcoords), "triangle_count": sum(len(f) for _, _, f in groups),
              "shape_count": len(groups), "material_count": len(materials),
              "fallback_uv_corners": fallback_uv_corners, "degenerate_triangles": degenerate,
              "removed_zero_area_source_triangles": removed_degenerate_triangles,
              "world_bounds": {"min": minimum, "max": maximum}, "normalization_center": center,
              "normalization_scale": normalization, "camera": {"world_position": position,
              "world_target": target, "position": normalized_position, "target": normalized_target,
              "horizontal_fov": horizontal_fov, "vertical_fov": vertical_fov,
              "source_resolution": [width, height]}, "lights": lights, "materials": material_report}
    for light in lights:
        light["normalized_area"] = light["world_area"] * normalization ** 2
    (output / "import-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "ATTRIBUTION.txt").write_text(
        "Veach Door (source title: Veach, Ajar)\n"
        "Scene by Benedikt Bitterli, inspired by Eric Veach's ajar-door transport test.\n"
        "Source collection: https://benedikt-bitterli.me/resources/\n"
        "Source archive: https://benedikt-bitterli.me/resources/mitsuba/veach-ajar.zip\n"
        "License: CC0 / public domain; original license is preserved in LICENSE.txt.\n"
        "Catalog reference: https://casual-effects.com/g3d/data10/index.html\n\n"
        "Reproduce from the extracted Mitsuba archive, using Python 3 (standard library only):\n"
        "python scripts/import-veach-door.py --source build/scene-import/veach-source/veach-ajar\n\n"
        "Conversion and renderer limitations:\n"
        "- All source shapes and their world transforms are preserved. The Mitsuba rectangle\n"
        "  [-1,1]^2 is triangulated with the original +Z winding before transformation.\n"
        "  Its original area-emitter radiance is Ke 1000 1000 1000. No extra lights or\n"
        "  emissive surfaces were added. VkNRC evaluates mesh emission on both sides;\n"
        "  Mitsuba's area emitter is one-sided, so exact lighting is not guaranteed.\n"
        f"- {removed_degenerate_triangles} zero-area source triangles are omitted to avoid undefined normals;\n"
        "  this does not change visible geometry. All original vertices are retained.\n"
        "- The original camera position and +Z viewing direction are normalized using\n"
        "  the merged OBJ AABB, exactly as Scene.cpp does. Horizontal 60-degree FOV at\n"
        "  1280x720 becomes the vertical FOV stored in veach-door.camera. VkNRC uses\n"
        "  world-up; the source camera's negligible roll is not represented.\n"
        "- Diffuse colors and three original JPEG bitmap files are preserved.\n"
        "- Complex conductor Fresnel is approximated with a scalar Schlick F0 and RGB\n"
        "  Ks tint, fitted to the exact source RGB normal-incidence reflectance.\n"
        "  The large effective Ni is a numerical Fresnel fit, not a physical glass IOR.\n"
        "  Source alpha is preserved; GGX distributions become renderer-supported\n"
        "  Beckmann distributions, so glossy tails and grazing reflections differ.\n"
        "- The checkerboard has the source 20x80 UV scale and two checks per unit UV.\n"
        "  It is baked into a 256x256 lossless PNG, including the conductor tint,\n"
        "  encoded to sRGB because VkNRC linearizes all color textures. Eight-bit\n"
        "  quantization and bilinear filtering slightly soften the procedural edges.\n"
        "- The glass teapot is an opaque reflective dielectric (Ni 1.5, alpha .001).\n"
        "  VkNRC has no transmission, refraction, or glass caustics, so it cannot\n"
        "  reproduce the transparent original teapot. Geometry remains intact.\n"
        "- VkNRC uses triangle geometric normals, including where the source uses\n"
        "  smooth vertex normals. Twosided source surfaces are compatible with its\n"
        "  face-forward shading. Every face has a material and valid UV indices.\n\n"
        "import-report.json records input SHA-256 hashes, geometry counts, bounds,\n"
        "camera conversion, light geometry, and material fitting parameters.\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("vertex_count", "texcoord_count", "triangle_count",
                      "shape_count", "fallback_uv_corners", "degenerate_triangles", "world_bounds",
                      "normalization_scale", "camera", "lights")}, indent=2))


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "build/scene-import/veach-source/veach-ajar")
    parser.add_argument("--output", type=Path, default=root / "assets/veach-door")
    args = parser.parse_args()
    import_scene(args.source, args.output)


if __name__ == "__main__":
    main()
