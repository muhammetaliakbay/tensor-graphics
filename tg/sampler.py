import tensorflow as tf

def uv_to_texture(texture: tf.Tensor, uvs: tf.Tensor):
    texture = tf.ensure_shape(texture, (None, None, None))
    uvs = tf.ensure_shape(uvs, (None, None, None, 2))

    sizes = tf.shape(texture)
    width, height = sizes[0], sizes[1]

    return uvs * (width, height)

def gather_pixels(texture: tf.Tensor, xy: tf.Tensor):
    texture = tf.ensure_shape(texture, (None, None, None))
    xy = tf.ensure_shape(xy, (None, None, None, 2))

    sizes = tf.shape(texture)
    xy = tf.math.floormod(xy, sizes[0:2])

    return tf.gather_nd(texture, xy)

def sample_floor(texture: tf.Tensor, uvs: tf.Tensor):
    xy = uv_to_texture(texture, uvs)
    return gather_pixels(texture, tf.cast(tf.math.floor(xy), tf.int32))

def sample_ceil(texture: tf.Tensor, uvs: tf.Tensor):
    xy = uv_to_texture(texture, uvs)
    return gather_pixels(texture, tf.cast(tf.math.ceil(xy), tf.int32))

def sample_bilinear(texture: tf.Tensor, uvs: tf.Tensor):
    xy = uv_to_texture(texture, uvs)
    
    xy_top_left = tf.math.floor(xy)
    xy_top_left_int = tf.cast(xy_top_left, tf.int32)
    xy_bottom_right_int = xy_top_left_int + (1, 1)
    xy_top_right_int = xy_top_left_int + (1, 0)
    xy_bottom_left_int = xy_top_left_int + (0, 1)
    
    pixel_top_left = gather_pixels(texture, xy_top_left_int)
    pixel_bottom_right = gather_pixels(texture, xy_bottom_right_int)
    pixel_top_right = gather_pixels(texture, xy_top_right_int)
    pixel_bottom_left = gather_pixels(texture, xy_bottom_left_int)

    delta_x = tf.expand_dims(xy[...,0] - xy_top_left[...,0], -1)
    delta_y = tf.expand_dims(xy[...,1] - xy_top_left[...,1], -1)

    mean_top = (pixel_top_left * (1-delta_x)) + (pixel_top_right * delta_x)
    mean_bottom = (pixel_bottom_left * (1-delta_y)) + (pixel_bottom_right * delta_y)
    mean = (mean_top * (1-delta_y)) + (mean_bottom * delta_y)

    return mean
