import tensorflow as tf

@tf.function
def tf_van_der_corput(n, base=2, dtype=tf.float64):
    """
    TensorFlow version of Van der Corput sequence (GPU-compatible).
    """
    i = tf.range(n, dtype=tf.int32)
    vdc = tf.zeros_like(tf.cast(i, dtype))

    inv_base = tf.constant(1.0, dtype=dtype) / tf.cast(base, dtype)

    def body(j, vdc, base_pow):
        digit = tf.math.floormod(i // (base ** j), base)
        vdc += tf.cast(digit, dtype) * base_pow
        base_pow /= tf.cast(base, dtype)
        return j + 1, vdc, base_pow

    cond = lambda j, *_: j < 32  # sufficient for ~2^32 samples

    _, result, _ = tf.while_loop(
        cond,
        body,
        loop_vars=[0, vdc, inv_base],
        maximum_iterations=32
    )
    return result

@tf.function
def tf_hammersley_sequence(n, dim, dtype=tf.float64):
    """
    Generate Hammersley points in [0, 1]^dim using TensorFlow on GPU.
    """
    seq = tf.TensorArray(dtype, size=dim)

    # First dimension is just linspace
    one = tf.constant(1.0, dtype=dtype)
    start = tf.constant(0.0, dtype=dtype)
    stop = one - one / tf.cast(n, dtype)
    seq = seq.write(0, tf.linspace(start, stop, n))


    # Use first dim-1 primes for bases (e.g., [2, 3, 5])
    primes = tf.constant([2, 3, 5, 7, 11, 13, 17], dtype=tf.int32)
    for i in tf.range(1, dim):
        vdc = tf_van_der_corput(n, base=primes[i - 1], dtype=dtype)
        seq = seq.write(i, vdc)

    return tf.transpose(seq.stack())  # shape (n, dim)

@tf.function
def tf_scale_to_domain(points, bounds):
    """
    Scales [0,1]^d points to a given domain.
    bounds: tensor of shape (d, 2), where [:,0] = lower, [:,1] = upper
    """
    bounds = tf.cast(bounds, points.dtype)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return lower + points * (upper - lower)

# Example usage
def tf_hammersley_sampler(n_points, domain_bounds, dtype=tf.float64):
    """
    domain_bounds: list or array-like of [(low1, high1), ..., (lowD, highD)]
    """
    dim = len(domain_bounds)
    bounds_tensor = tf.convert_to_tensor(domain_bounds, dtype=dtype)
    points_unit = tf_hammersley_sequence(n_points, dim, dtype=dtype)
    return tf_scale_to_domain(points_unit, bounds_tensor)
