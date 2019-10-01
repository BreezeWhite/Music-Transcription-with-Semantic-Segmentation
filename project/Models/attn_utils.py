
from keras.initializers import Identity

from keras import layers as L
from keras import backend as K
import tensorflow as tf

def split_heads_2d(x, num_heads):
    m = K.int_shape(x)[-1]
    assert(m % num_heads==0), "Last dim: {}, Number of heads: {}".format(m, num_heads)
    
    ori_shape = L.Lambda(lambda x: K.shape(x))(x)
    extra_dim = K.constant([num_heads, m//num_heads], dtype=tf.int32)
    tar_shape = K.concatenate([ori_shape[:-1], extra_dim])
    out = K.reshape(x, tar_shape)
    return K.permute_dimensions(out, (0, 3, 1, 2, 4))

def combine_heads_2d(x):
    x = K.permute_dimensions(x, [0, 2, 3, 1, 4])
    x_shape = K.shape(x)
    a, b = K.int_shape(x)[-2:]
    tar_shape = K.concatenate([x_shape[:-2], [a*b]])
    return K.reshape(x, tar_shape)

def pad_to_multiple(x, block_shape):
    ori_shape = K.int_shape(x)

    pad_t = ori_shape[1] % block_shape[0]
    pad_f = ori_shape[2] % block_shape[1]
    padding=((0, 0), (0, pad_t), (0, pad_f))
    
    new_x = L.ZeroPadding3D(padding=padding)(x)

    return new_x


def gather_indices_2d(x, block_shape, block_stride):
    kernel = K.eye(block_shape[0]*block_shape[1])
    #kernel = K.reshape(kernel, [block_shape[0], block_shape[1], 1, block_shape[0]*block_shape[1]])
    kernel = reshape_range(kernel, 0, 1, [block_shape[0], block_shape[1], 1])

    x_shape = K.shape(x)
    indices = K.arange(x_shape[2]*x_shape[3])
    indices = K.reshape(indices, [1, x_shape[2], x_shape[3], 1])    
    indices = K.conv2d(tf.cast(indices, tf.float32), kernel, strides=(block_stride[0], block_stride[1]))

    i_shape = K.shape(indices)[:3]
    n_blocks = tf.reduce_prod(i_shape)
    indices = K.reshape(indices, [n_blocks, -1])
    return tf.cast(indices, tf.int32)

def reshape_range(x, i, j, shape):
    cur_shape = K.shape(x)
    tar_shape = K.concatenate([cur_shape[:i], shape, cur_shape[j:]])
    return K.reshape(x, tar_shape)

def gather_blocks_2d(x, indices):
    x_shape = K.shape(x)
    x = reshape_range(x, 2, 4, [tf.reduce_prod(x_shape[2:4])])
    # [length, batch, heads, dim]
    x_t = K.permute_dimensions(x, [2, 0, 1, 3])
    x_new = K.gather(x_t, indices)
    # returns [batch, heads, num_blocks, block_ength**2, dim]
    return K.permute_dimensions(x_new, [2, 3, 0, 1, 4])

def scatter_blocks_2d(x, indices, shape):
    x_shape = K.shape(x)
    x_t = K.reshape(x, [x_shape[0], x_shape[1], -1, x_shape[-1]])
    x_t = K.permute_dimensions(x_t, [2, 0, 1, 3])
    x_t_shape = K.shape(x_t)
    indices = K.reshape(indices, [-1, 1])
    scattered_x = tf.scatter_nd(indices, x_t, x_t_shape)
    scattered_x = K.permute_dimensions(scattered_x, [1, 2, 0, 3])
    
    return K.reshape(scattered_x, shape)
