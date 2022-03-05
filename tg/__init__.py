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

    tile_indices = util.range_2d(aligned_width // tile_size, aligned_height // tile_size)
    tile_offsets = util.range_2d(aligned_width // tile_size, aligned_height // tile_size, tile_size)
    tile_ends = tile_offsets + tile_size

    boundaries_min, boundaries_max = util.boundaries(triangles)

    tile_overlaps = util.intersects_tile(tile_offsets, tile_ends, boundaries_min, boundaries_max)
    tile_overlaps = tf.map_fn(
        lambda tile_overlaps_cols: tf.map_fn(
            lambda tile_overlaps_cell: tf.squeeze(tf.where(tile_overlaps_cell), -1),
            tile_overlaps_cols,
            fn_output_signature=tf.RaggedTensorSpec(
                shape=(None,),
                dtype=tf.int64,
            ),
        ),
        tile_overlaps,
        fn_output_signature=tf.RaggedTensorSpec(
            shape=(tile_overlaps.shape[1], None),
            dtype=tf.int64,
        ),
    )

    def render_tile(col, row):
        tile_offset = tile_offsets[col][row]
        overlaps = tile_overlaps[col][row]

        overlap_triangles = tf.gather(triangles, overlaps)
        overlap_planes = tf.gather(planes, overlaps)
        overlap_colors = tf.gather(colors, overlaps)

        return tri.render_triangles(
            overlap_triangles, overlap_planes,
            tile_offset, tile_size,
            dtype,
            near_limit, far_limit,
            shader, overlap_colors,
            background[tile_offset[0]:tile_offset[0] + tile_size, tile_offset[1]:tile_offset[1] + tile_size],
            background_depth[tile_offset[0]:tile_offset[0] + tile_size, tile_offset[1]:tile_offset[1] + tile_size],
        )

    color_image = tf.map_fn(
        lambda tile_col_indices: tf.map_fn(
            lambda indices: render_tile(indices[0], indices[1]),
            tile_col_indices,
            dtype=dtype,
        ),
        tile_indices,
        dtype=dtype,
    )
    color_image = util.collate_images(color_image)

    return color_image
