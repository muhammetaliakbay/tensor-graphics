from typing import Iterable

def load(lines: Iterable[str]):
    vertices = []
    normals = []
    uvs = []
    triangles = []
    triangle_normals = []
    triangle_uvs = []
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
        elif parts[0] == "vt":
            x, y = float(parts[1]), float(parts[2])
            uvs.append((x, y))
        elif parts[0] == "f":
            face = []
            face_normals = []
            face_uvs = []
            for tri in parts[1:]:
                vi = int(tri.split("/")[0]) - 1
                vertex = vertices[vi]
                face.append(vertex)
                uvi = int(tri.split("/")[1]) - 1
                uv = uvs[uvi]
                face_uvs.append(uv)
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
                aUV = face_uvs[(i * 2) % len(face_uvs)]
                bUV = face_uvs[(i * 2 + 1) % len(face_uvs)]
                cUV = face_uvs[(i * 2 + 2) % len(face_uvs)]
                triangles.append((a, b, c))
                triangle_normals.append((aN, bN, cN))
                triangle_uvs.append((aUV, bUV, cUV))
    return triangles, triangle_normals, triangle_uvs
    