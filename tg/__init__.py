import tensorflow as tf
import numpy as np

def in_triangle(triangle: tf.Tensor, points: tf.Tensor):
    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = points - triangle[0]

    dot00 = tf.tensordot(v0, v0, [0, 0])
    dot01 = tf.tensordot(v0, v1, [0, 0])
    dot02 = tf.tensordot(v0, v2, [0, 2])
    dot11 = tf.tensordot(v1, v1, [0, 0])
    dot12 = tf.tensordot(v1, v2, [0, 2])

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    return (u >= 0) & (v >= 0) & (u + v < 1)

def distances(triangle: tf.Tensor, points: tf.Tensor):
    diffs = tf.expand_dims(tf.expand_dims(triangle, 0), 1) - tf.expand_dims(points, 2)
    squared_distances = tf.reduce_sum(tf.square(diffs), -1)
    dists = tf.sqrt(squared_distances)
    return dists

def distanced_weights(triangle: tf.Tensor, points: tf.Tensor):
    raw_weights = 1 / distances(triangle, points)
    raw_weight_totals = tf.reduce_sum(raw_weights, -1)
    weights = raw_weights / tf.expand_dims(raw_weight_totals, -1)
    return weights

def depths(triangle: tf.Tensor, points: tf.Tensor):
    v1 = triangle[0] - triangle[1]
    v2 = triangle[0] - triangle[2]
    n = tf.linalg.cross(v1, v2)
    k = tf.tensordot(triangle[0], n, 1)
    z = ((k - (tf.reduce_sum(n[0:2] * points, -1))) / n[2])
    return z

def align(val, steps, ceil = False):
    div = val // steps
    modulo = val - div * steps
    rounded = div + tf.sign(modulo) if ceil else div
    aligned = rounded * steps
    return aligned

def align_boundary(val, ceil, width, height):
    rounded = tf.math.ceil(val) if ceil else tf.math.floor(val)
    quantize = tf.cast(tf.minimum(tf.maximum(rounded, (0, 0)), (width, height)), tf.int32)
    aligned = align(quantize, 1, ceil)
    return aligned

def render(triangles: tf.Tensor, colors: tf.Tensor, width: int, height: int, near_limit: float, far_limit: float, shader = None):
    dtype=tf.float32

    triangles = tf.ensure_shape(triangles, (None, 3, 3))
    boundaries_min = tf.reduce_min(triangles, -2)
    boundaries_max = tf.reduce_max(triangles, -2)

    colors = tf.ensure_shape(colors, (None, 3, 3))

    aligned_width, aligned_height = align(tf.constant((width, height), dtype=tf.int32), 1, True)

    color_image = tf.Variable(tf.zeros((aligned_width, aligned_height, 3)))
    boundaries_image = tf.Variable(tf.zeros((aligned_width, aligned_height)))
    depth_image = tf.Variable(tf.tile(
        tf.reshape(np.inf, (1, 1)),
        (aligned_width, aligned_height),
    ))

    for boundary_min, boundary_max, triangle, color in zip(boundaries_min, boundaries_max, triangles, colors):
        boundary_min_int = align_boundary(boundary_min[0:2], False, width, height)
        boundary_max_int = align_boundary(boundary_max[0:2], True, width, height)
        boundary_size_int = boundary_max_int - boundary_min_int
        
        coordinates_x = tf.broadcast_to(
            tf.expand_dims(
                tf.range(
                    boundary_min_int[0], boundary_max_int[0],
                    dtype=dtype,
                ),
                1,
            ),
            boundary_size_int,
        )

        coordinates_y = tf.broadcast_to(
            tf.expand_dims(
                tf.range(
                    boundary_min_int[1], boundary_max_int[1],
                    dtype=dtype,
                ),
                0,
            ),
            boundary_size_int,
        )
        
        coordinates = tf.stack(
            (coordinates_x, coordinates_y),
            -1,
        )

        triangle_mask = in_triangle(triangle[:, 0:2], coordinates)

        coordinates_z = depths(triangle, coordinates)
        coordinates_3d = tf.concat((coordinates, tf.expand_dims(coordinates_z, -1)), -1)

        depth_image_slice = depth_image[boundary_min_int[0]:boundary_max_int[0], boundary_min_int[1]:boundary_max_int[1]]

        triangle_mask = tf.logical_and(triangle_mask, coordinates_z <= depth_image_slice)
        triangle_mask = tf.logical_and(triangle_mask, coordinates_z >= near_limit)
        triangle_mask = tf.logical_and(triangle_mask, coordinates_z <= far_limit)

        boundary_depth_image = tf.where(
            triangle_mask,
            coordinates_z,
            depth_image_slice,
        )
        depth_image_slice.assign(boundary_depth_image)

        vertex_weights = distanced_weights(triangle, coordinates_3d)

        if shader is None:
            boundary_color_image = tf.expand_dims(vertex_weights, -1) * tf.expand_dims(tf.expand_dims(color, 0), 0)
            boundary_color_image = tf.reduce_sum(boundary_color_image, -2)
        else:

            boundary_color_image = shader(coordinates_3d)

        color_image_slice = color_image[boundary_min_int[0]:boundary_max_int[0], boundary_min_int[1]:boundary_max_int[1], :]
        boundary_color_image = tf.where(
            tf.expand_dims(triangle_mask, -1),
            boundary_color_image,
            color_image_slice,
        )
        color_image_slice.assign(boundary_color_image)

        boundary_image = tf.ones(boundary_size_int, dtype)

        boundaries_image_slice = boundaries_image[boundary_min_int[0]:boundary_max_int[0], boundary_min_int[1]:boundary_max_int[1]]
        boundaries_image_slice.assign(boundary_image)

    return color_image[:width, :height], boundaries_image[:width, :height], depth_image[:width, :height]
