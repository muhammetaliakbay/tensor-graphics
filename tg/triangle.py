import tensorflow as tf
import tg.util as util
import numpy as np

@tf.function(experimental_relax_shapes=True)
def render_triangles(
    triangles, planes,
    tile_offset, tile_size,
    dtype,
    near_limit, far_limit,
    pixel_shader,
    data,
    background, background_depth_slice,
):
    coordinates_xy = tf.cast(
        util.range_2d(tile_size, tile_size) + tile_offset,
        dtype,
    )
    
    barycentric_weights = util.barycentric_distanced_weights(triangles, coordinates_xy)

    coordinates_z = tf.squeeze(util.interpolate_triangle(barycentric_weights, tf.expand_dims(triangles[...,2], -1)), -1)
    coordinates_xy_ = tf.tile(tf.expand_dims(coordinates_xy, 0), (tf.shape(coordinates_z)[0], 1, 1, 1))
    coordinates = tf.concat((coordinates_xy_, tf.expand_dims(coordinates_z, -1)), -1)

    triangle_mask = tf.reduce_all(barycentric_weights >= 0, -1)
    triangle_mask = tf.logical_and(triangle_mask, coordinates_z >= near_limit)
    triangle_mask = tf.logical_and(triangle_mask, coordinates_z <= far_limit)

    interpolated_data = [util.interpolate_triangle(barycentric_weights, data_item) for data_item in data]

    if pixel_shader is None:
        tile_color_image = interpolated_data[0]
    else:
        tile_color_image = pixel_shader(coordinates, *interpolated_data)
    
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
