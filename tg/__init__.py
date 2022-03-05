import tensorflow as tf
import numpy as np
import tg.util as util
import tg.triangle as tri

@tf.function
def render(triangles: tf.Tensor, colors: tf.Tensor, width: int, height: int, near_limit: float, far_limit: float, shader = None, dtype = tf.float32, background = None, background_depth = None):
    triangles = tf.cast(tf.ensure_shape(triangles, (None, 3, 3)), dtype)
    planes = util.planes(triangles)

    colors = tf.cast(tf.ensure_shape(colors, (None, 3, 3)), dtype)

    tile_size = 32

    aligned_size = util.align(tf.constant((width, height), dtype=tf.int32), tile_size, True)
    aligned_width = aligned_size[0]
    aligned_height = aligned_size[1]

    if background is None:
        background = tf.zeros((aligned_width, aligned_height, 3), dtype=dtype)
    else:
        background = tf.image.resize(background, (width, height))
        background = tf.pad(background, ((0, aligned_width - width), (0, aligned_height - height), (0,0)), constant_values=np.Inf)
        background = tf.cast(background, dtype)

    if background_depth is None:
        background_depth = tf.tile(tf.expand_dims(tf.expand_dims(tf.constant(np.inf, dtype=dtype), -1), -1), background.shape[0:2])
    else:
        background_depth = tf.image.resize(tf.expand_dims(background_depth, -1), (width, height))
        background_depth = tf.pad(tf.squeeze(background_depth, -1), ((0, aligned_width - width), (0, aligned_height - height)), constant_values=np.Inf)
        background_depth = tf.cast(background_depth, dtype)

    tile_offsets = util.range_2d(aligned_width // tile_size, aligned_height // tile_size, tile_size)

    color_image = tf.map_fn(
        lambda tile_cols: tf.map_fn(
            lambda tile_offset: tri.render_triangles(
                triangles, planes,
                tile_offset, tile_size,
                dtype,
                near_limit, far_limit,
                shader, colors,
                background[tile_offset[0]:tile_offset[0] + tile_size, tile_offset[1]:tile_offset[1] + tile_size],
                background_depth[tile_offset[0]:tile_offset[0] + tile_size, tile_offset[1]:tile_offset[1] + tile_size],
            ),
            tile_cols,
            dtype=dtype,
        ),
        tile_offsets,
        dtype=dtype,
    )
    color_image = util.collate_images(color_image)

    return color_image
