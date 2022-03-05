import tensorflow as tf

def in_triangle(triangles: tf.Tensor, points: tf.Tensor):
    v0 = triangles[:, 2] - triangles[:, 0]
    v1 = triangles[:, 1] - triangles[:, 0]
    v2 = points - tf.expand_dims(tf.expand_dims(triangles[:, 0], -2), -2)

    dot00 = tf.reduce_sum(v0 * v0, -1)
    dot01 = tf.reduce_sum(v0 * v1, -1)
    dot02 = tf.reduce_sum(tf.expand_dims(tf.expand_dims(v0, -2), -2) * v2, -1)
    dot11 = tf.reduce_sum(v1 * v1, -1)
    dot12 = tf.reduce_sum(tf.expand_dims(tf.expand_dims(v1, -2), -2) * v2, -1)

    _dot00 = tf.expand_dims(tf.expand_dims(dot00, -1), -1)
    _dot01 = tf.expand_dims(tf.expand_dims(dot01, -1), -1)
    _dot11 = tf.expand_dims(tf.expand_dims(dot11, -1), -1)

    invDenom = 1 / (_dot00 * _dot11 - _dot01 * _dot01)
    u = (_dot11 * dot02 - _dot01 * dot12) * invDenom
    v = (_dot00 * dot12 - _dot01 * dot02) * invDenom

    return (u >= 0) & (v >= 0) & (u + v < 1)

def distances(triangles: tf.Tensor, points: tf.Tensor):
    diffs = (
        tf.expand_dims(tf.expand_dims(triangles, 1), 1)
        - tf.expand_dims(points, -2)
    )
    squared_distances = tf.reduce_sum(tf.square(diffs), -1)
    dists = tf.sqrt(squared_distances)
    return dists

def distanced_weights(triangles: tf.Tensor, points: tf.Tensor):
    raw_weights = 1 / distances(triangles, points)
    raw_weight_totals = tf.reduce_sum(raw_weights, -1)
    weights = raw_weights / tf.expand_dims(raw_weight_totals, -1)
    return weights

def planes(triangles: tf.Tensor):
    v1 = triangles[:, 0] - triangles[:, 1]
    v2 = triangles[:, 0] - triangles[:, 2]
    n = tf.linalg.cross(v1, v2)
    d = - tf.reduce_sum(triangles[:, 0] * n, -1)
    plns = tf.concat(
        (
            n,
            tf.expand_dims(d, -1),
        ),
        -1,
    )
    return plns

def depths(planes: tf.Tensor, points: tf.Tensor):
    _planes = tf.expand_dims(tf.expand_dims(planes, -1), -1)
    A = _planes[:, 0, :, :]
    B = _planes[:, 1, :, :]
    C = _planes[:, 2, :, :]
    D = _planes[:, 3, :, :]
    z = - (D + (A * points[:, :, 0]) + (B * points[:, :, 1])) / C
    return z

def align(val, steps, ceil = False):
    div = val // steps
    modulo = val - div * steps
    rounded = div + tf.sign(modulo) if ceil else div
    aligned = rounded * steps
    return aligned

def align_boundary(val, ceil, width, height, steps):
    rounded = tf.math.ceil(val) if ceil else tf.math.floor(val)
    quantize = tf.cast(tf.minimum(tf.maximum(rounded, (0, 0)), (width, height)), tf.int32)
    aligned = align(quantize, steps, ceil)
    return aligned

def range_2d(w, h, step = 1):
    return tf.stack(
        (
            tf.tile(tf.expand_dims(tf.range(0, w * step, step), 1), (1, h)),
            tf.tile(tf.expand_dims(tf.range(0, h * step, step), 0), (w, 1)),
        ),
        -1,
    )

def gather_images(images, indices):
    batch = tf.transpose(images, (1, 2, 0, 3))
    gathered = tf.gather(batch, indices, batch_dims=2)
    return gathered

def collate_images(images):
    return tf.concat(
        tf.unstack(
            tf.map_fn(
                lambda col: tf.concat(tf.unstack(col, axis=0), axis=1),
                images,
            ),
            axis=0
        ),
        axis=0,
    )
