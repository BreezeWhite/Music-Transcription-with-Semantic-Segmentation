from keras.engine import Input, Model
from keras.layers import Dense, Reshape, add, TimeDistributed, LSTM, CuDNNLSTM, Dropout, Lambda, concatenate, Multiply
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling2D, UpSampling1D,Conv2D,Conv2DTranspose,MaxPooling2D,Cropping2D
from keras.optimizers import SGD, Adam
from keras import regularizers
#from project.utils import load_model, model_copy

import tensorflow as tf
from tensorflow.python.ops import array_ops

from tensor2tensor.layers.common_attention import local_attention_2d, split_heads_2d, combine_heads_2d
from keras import layers as L



def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_mean(per_entry_cross_ent)


def sparse_loss(yTrue,yPred):
    loss = focal_loss(yPred, yTrue)
    return loss


def conv_block(input_tensor,
               channel, kernel_size,
               strides=(2, 2),
               dilation_rate=1,
               dropout_rate=0.4
               ):

    skip = input_tensor

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(channel, kernel_size, strides=strides, dilation_rate=dilation_rate,
                          padding="same")(input_tensor)

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(channel, kernel_size, strides=(1, 1), dilation_rate=dilation_rate,
                          padding="same")(input_tensor)

    if (strides != (1, 1)):
        skip = Conv2D(channel, (1, 1), strides=strides, padding="same")(skip)
    input_tensor = add([input_tensor, skip])

    return input_tensor


def transpose_conv_block(input_tensor,
                         channel,
                         kernel_size,
                         strides=(2, 2),
                         dropout_rate=0.4
                         ):

    skip = input_tensor

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(channel, kernel_size, strides=(1, 1), padding="same")(input_tensor)

    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2DTranspose(channel, kernel_size, strides=strides, padding="same")(input_tensor)

    if (strides != (1, 1)):
        skip = Conv2DTranspose(channel, (1, 1), strides=strides, padding="same")(skip)
    input_tensor = add([input_tensor, skip])

    return input_tensor

def dot_attention(q, k, v):
    logits = K.batch_dot(q, k, axes=(4, 3))
    weights = K.softmax(logits)
    return K.batch_dot(weights, v)

def MultiHead_Attention(x, out_channel=64, d_model=16, n_heads=8, query_shape=(32, 32), memory_flange=(8, 8)):
    q = Conv2D(d_model, (3, 3), strides=(1, 1), padding="same")(x)
    k = Conv2D(d_model, (3, 3), strides=(1, 1), padding="same")(x)
    v = Conv2D(d_model, (3, 3), strides=(1, 1), padding="same")(x)

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

def seg(feature_num=128,
        timesteps=256,
        multi_grid_layer_n=1,
        multi_grid_n=3,
        input_channel=1,
        prog = False,
        out_class=2
        ):
    layer_out = []

    input_score = Input(shape=(timesteps, feature_num, input_channel), name="input_score_48")
    en = Conv2D(2 ** 5, (7, 7), strides=(1, 1), padding="same")(input_score)
    layer_out.append(en)

    en_l1 = conv_block(en, 2 ** 5, (3, 3), strides=(2, 2))
    en_l1 = conv_block(en_l1, 2 ** 5, (3, 3), strides=(1, 1))
    layer_out.append(en_l1)

    en_l2 = conv_block(en_l1, 2 ** 6, (3, 3), strides=(2, 2))
    en_l2 = conv_block(en_l2, 2 ** 6, (3, 3), strides=(1, 1))
    en_l2 = conv_block(en_l2, 2 ** 6, (3, 3), strides=(1, 1))
    layer_out.append(en_l2)

    en_l3 = conv_block(en_l2, 2 ** 7, (3, 3), strides=(2, 2))
    en_l3 = conv_block(en_l3, 2 ** 7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2 ** 7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2 ** 7, (3, 3), strides=(1, 1))
    layer_out.append(en_l3)

    en_l4 = conv_block(en_l3, 2 ** 8, (3, 3), strides=(2, 2))
    en_l4 = conv_block(en_l4, 2 ** 8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2 ** 8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2 ** 8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2 ** 8, (3, 3), strides=(1, 1))
    layer_out.append(en_l4)

    feature = en_l4
    feature = L.Lambda(MultiHead_Attention)(feature)

    #for i in range(multi_grid_layer_n):
    #    feature = BatchNormalization()(Activation("relu")(feature))
    #    feature = Dropout(0.3)(feature)
    #    m = BatchNormalization()(Conv2D(2 ** 9, (1, 1), strides=(1, 1), padding="same", activation="relu")(feature))
    #    multi_grid = m
    #    for ii in range(multi_grid_n):
    #        m = BatchNormalization()(Conv2D(2 ** 9, (3, 3), strides=(1, 1),
    #                                        dilation_rate=2 ** ii, padding="same", activation="relu"
    #                                        )(feature))
    #        multi_grid = concatenate([multi_grid, m])
    #    multi_grid = Dropout(0.3)(multi_grid)
    #    feature = Conv2D(2 ** 9, (1, 1), strides=(1, 1), padding="same")(multi_grid)
    #    layer_out.append(feature)

    feature = BatchNormalization()(Activation("relu")(feature))

    feature = Conv2D(2 ** 8, (1, 1), strides=(1, 1), padding="same")(feature)
    feature = add([feature, en_l4])
    de_l1 = transpose_conv_block(feature, 2 ** 7, (3, 3), strides=(2, 2))
    layer_out.append(de_l1)

    skip = de_l1
    de_l1 = BatchNormalization()(Activation("relu")(de_l1))
    de_l1 = concatenate([de_l1, BatchNormalization()(Activation("relu")(en_l3))])
    de_l1 = Dropout(0.4)(de_l1)
    de_l1 = Conv2D(2 ** 7, (1, 1), strides=(1, 1), padding="same")(de_l1)
    de_l1 = add([de_l1, skip])
    de_l2 = transpose_conv_block(de_l1, 2 ** 6, (3, 3), strides=(2, 2))
    layer_out.append(de_l2)

    skip = de_l2
    de_l2 = BatchNormalization()(Activation("relu")(de_l2))
    de_l2 = concatenate([de_l2, BatchNormalization()(Activation("relu")(en_l2))])
    de_l2 = Dropout(0.4)(de_l2)
    de_l2 = Conv2D(2 ** 6, (1, 1), strides=(1, 1), padding="same")(de_l2)
    de_l2 = add([de_l2, skip])
    de_l3 = transpose_conv_block(de_l2, 2 ** 5, (3, 3), strides=(2, 2))
    layer_out.append(de_l3)

    skip = de_l3
    de_l3 = BatchNormalization()(Activation("relu")(de_l3))
    de_l3 = concatenate([de_l3, BatchNormalization()(Activation("relu")(en_l1))])
    de_l3 = Dropout(0.4)(de_l3)
    de_l3 = Conv2D(2 ** 5, (1, 1), strides=(1, 1), padding="same")(de_l3)
    de_l3 = add([de_l3, skip])
    de_l4 = transpose_conv_block(de_l3, 2 ** 5, (3, 3), strides=(2, 2))
    layer_out.append(de_l4)

    de_l4 = BatchNormalization()(Activation("relu")(de_l4))
    de_l4 = Dropout(0.4)(de_l4)
    out = Conv2D(out_class, (1, 1), strides=(1, 1), padding="same", name='prediction')(de_l4)

    if(prog):
        model = Model(inputs=input_score,
                      outputs=layer_out)
    else:
        model = Model(inputs=input_score,
                      outputs=out)

    return model

import numpy as np
if __name__ == "__main__":
    model = seg(feature_num=128, input_channel=8, timesteps=256, out_class=3)
    model.compile(optimizer="adam", loss={"prediction": focal_loss}, metrics=["accuracy"])
    
    bs = 16
    a = tf.constant(np.arange(bs*256*128*2, dtype=np.float32), shape=[bs, 256, 128, 8])

    out = model(a)



