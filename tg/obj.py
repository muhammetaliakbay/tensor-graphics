from typing import Iterable

def load(lines: Iterable[str]):
    vertices = []
    normals = []
    triangles = []
    triangle_normals = []
    for line in lines:
        parts = line.split(" ")
        if len(parts) == 0:
            continue
        elif parts[0] == "v":
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            vertices.append((x, y, z))
        elif parts[0] == "vn":
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            normals.append((x, y, z))
        elif parts[0] == "f":
            face = []
            face_normals = []
            for tri in parts[1:]:
                vi = int(tri.split("/")[0]) - 1
                vertex = vertices[vi]
                face.append(vertex)
                ni = int(tri.split("/")[2]) - 1
                normal = normals[ni]
                face_normals.append(normal)
            for i in range(0, len(face) - 2):
                a = face[(i * 2) % len(face)]
                b = face[(i * 2 + 1) % len(face)]
                c = face[(i * 2 + 2) % len(face)]
                aN = face_normals[(i * 2) % len(face_normals)]
                bN = face_normals[(i * 2 + 1) % len(face_normals)]
                cN = face_normals[(i * 2 + 2) % len(face_normals)]
                triangles.append((a, b, c))
                triangle_normals.append((aN, bN, cN))
    return triangles, triangle_normals
    