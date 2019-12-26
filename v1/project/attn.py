from keras.layers.convolutional import Conv2D
from tensor2tensor.layers.common_attention import local_attention_2d, split_heads_2d, combine_heads_2d



def multihead_attention(x, out_channel=64, d_model=32, n_heads=8, query_shape=(128, 24), memory_flange=(8, 8)):
    q = Conv2D(d_model, (3, 3), strides=(1, 1), padding="same", name="gen_q_conv")(x)
    k = Conv2D(d_model, (3, 3), strides=(1, 1), padding="same", name="gen_k_conv")(x)
    v = Conv2D(d_model, (3, 3), strides=(1, 1), padding="same", name="gen_v_conv")(x)

    q = split_heads_2d(q, n_heads)
    k = split_heads_2d(k, n_heads)
    v = split_heads_2d(v, n_heads)

    k_depth_per_head = d_model // n_heads
    q *= k_depth_per_head**-0.5

    
    """
    # local attetion 2d
    v_shape = K.int_shape(v)
    q = pad_to_multiple(q, query_shape)
    k = pad_to_multiple(k, query_shape)
    v = pad_to_multiple(v, query_shape)

    paddings = ((0, 0), (memory_flange[0], memory_flange[1]), (memory_flange[0], memory_flange[1]))
    k = L.ZeroPadding3D(padding=paddings)(k)
    v = L.ZeroPadding3D(padding=paddings)(v)
    
    # Set up query blocks
    q_indices = gather_indices_2d(q, query_shape, query_shape)
    q_new = gather_blocks_2d(q, q_indices)

    # Set up key and value blocks
    memory_shape = (query_shape[0] + 2*memory_flange[0],
                    query_shape[1] + 2*memory_flange[1])
    k_and_v_indices = gather_indices_2d(k, memory_shape, query_shape)
    k_new = gather_blocks_2d(k, k_and_v_indices)
    v_new = gather_blocks_2d(v, k_and_v_indices)

    output = dot_attention(q_new, k_new, v_new)

    # Put output back into original shapes
    padded_shape = K.shape(q)
    output = scatter_blocks_2d(output, q_indices, padded_shape) 

    # Remove padding
    output = K.slice(output, [0, 0, 0, 0, 0], [-1, -1, v_shape[2], v_shape[3], -1])
    """

    output = local_attention_2d(q, k, v, query_shape=query_shape, memory_flange=memory_flange)
    
    output = combine_heads_2d(output)
    output = Conv2D(out_channel, (3, 3), strides=(1, 1), padding="same", use_bias=False)(output)
    
    return output
