import tensorflow as tf
import tg.util as util
import numpy as np

@tf.function(experimental_relax_shapes=True)
def render_triangles(
    triangles, normals, planes,
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
    coordinates_xy_ = tf.tile(tf.expand_dims(coordinates_xy, 0), (tf.shape(coordinates_z)[0], 1, 1, 1))
    coordinates = tf.concat((coordinates_xy_, tf.expand_dims(coordinates_z, -1)), -1)

    # triangle_mask = tf.logical_and(triangle_mask, coordinates_z >= near_limit)
    # triangle_mask = tf.logical_and(triangle_mask, coordinates_z <= far_limit)

    vertex_weights = util.distanced_weights(triangles, coordinates)

    if shader is None:
        tile_color_image = util.interpolate_triangle(vertex_weights, colors)
    else:
        tile_normal_image = util.interpolate_triangle(vertex_weights, normals)
        tile_color_image = shader(coordinates, tile_normal_image)
    
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
