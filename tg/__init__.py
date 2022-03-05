import tensorflow as tf
import numpy as np
import tg.util as util
import tg.triangle as tri

@tf.function
def render(triangles: tf.Tensor, colors: tf.Tensor, width: int, height: int, near_limit: float, far_limit: float, shader = None, dtype = tf.float32):
    triangles = tf.cast(tf.ensure_shape(triangles, (None, 3, 3)), dtype)
    planes = util.planes(triangles)

    colors = tf.cast(tf.ensure_shape(colors, (None, 3, 3)), dtype)

    boundary_size = 32

    aligned_size = util.align(tf.constant((width, height), dtype=tf.int32), boundary_size, True)
    aligned_width = aligned_size[0]
    aligned_height = aligned_size[1]

    background = tf.zeros((aligned_width, aligned_height, 3), dtype=dtype)
    background_depth = tf.tile(tf.expand_dims(tf.expand_dims(tf.constant(np.inf, dtype=dtype), -1), -1), (aligned_width, aligned_height))

    boundary_offsets = util.range_2d(aligned_width // boundary_size, aligned_height // boundary_size, boundary_size)

    color_image = tf.map_fn(
        lambda boundary_cols: tf.map_fn(
            lambda boundary: tri.render_triangles(
                triangles, planes,
                boundary, boundary_size,
                dtype,
                near_limit, far_limit,
                shader, colors,
                background[boundary[0]:boundary[0] + boundary_size, boundary[1]:boundary[1] + boundary_size],
                background_depth[boundary[0]:boundary[0] + boundary_size, boundary[1]:boundary[1] + boundary_size],
            ),
            boundary_cols,
            dtype=dtype,
        ),
        boundary_offsets,
        dtype=dtype,
    )
    color_image = util.collate_images(color_image)

    return color_image
