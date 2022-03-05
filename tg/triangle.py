import tensorflow as tf
import tg.util as util
import numpy as np

@tf.function
def render_triangles(
    triangles, planes,
    tile_offset, tile_size,
    dtype,
    near_limit, far_limit,
    shader, colors,
    background, background_depth_slice,
):
    coordinates_xy = tf.cast(
        util.range_2d(tile_size, tile_size) + tile_offset,
        dtype,
    )

    triangle_mask = util.in_triangle(triangles[:, :, 0:2], coordinates_xy)

    coordinates_z = util.depths(planes, coordinates_xy)
    coordinates_xy_ = tf.tile(tf.expand_dims(coordinates_xy, 0), (coordinates_z.shape[0], 1, 1, 1))
    coordinates = tf.concat((coordinates_xy_, tf.expand_dims(coordinates_z, -1)), -1)

    # triangle_mask = tf.logical_and(triangle_mask, coordinates_z >= near_limit)
    # triangle_mask = tf.logical_and(triangle_mask, coordinates_z <= far_limit)

    if shader is None:
        tile_color_image = tf.expand_dims(vertex_weights, -1) * tf.expand_dims(tf.expand_dims(colors, 0), 0)
        tile_color_image = tf.reduce_sum(tile_color_image, -2)
    else:
        tile_color_image = shader(coordinates)

    vertex_weights = util.distanced_weights(triangles, coordinates)

    if shader is None:
        tile_color_image = tf.expand_dims(vertex_weights, -1) * tf.expand_dims(tf.expand_dims(colors, 0), 0)
        tile_color_image = tf.reduce_sum(tile_color_image, -2)
    else:
        tile_color_image = shader(coordinates)
    
    coordinates_depth = tf.where(triangle_mask, coordinates_z, np.inf)
    indices = tf.argmin(
        tf.concat(
            (
                tf.expand_dims(background_depth_slice, 0),
                coordinates_depth,
            ),
            0,
        ),
        0,
    )

    tile_color_image = util.gather_images(
        tf.concat(
            (
                tf.expand_dims(background, 0),
               tile_color_image,
            ),
            0,
        ),
        indices,
    )

    return tile_color_image
