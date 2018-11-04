from keras.engine import Input, Model
from keras.layers import Dense, Reshape, add, TimeDistributed, LSTM, CuDNNLSTM, Dropout, Lambda, concatenate, Multiply
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling2D, UpSampling1D,Conv2D,Conv2DTranspose,MaxPooling2D,Cropping2D

from keras.optimizers import SGD, Adam
from keras import regularizers
from project.utils import load_model, model_copy

import tensorflow as tf
from tensorflow.python.ops import array_ops


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
               dropout_rate=0.2
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


def adapter(input_tensor,
            channel,
            kernel_size=(1, 9),
            strides=(1, 3),
            dropout_rate=0.2
            ):
    input_tensor = BatchNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2DTranspose(channel, kernel_size, strides=strides, padding="same")(input_tensor)

    return input_tensor


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

    for i in range(multi_grid_layer_n):
        feature = BatchNormalization()(Activation("relu")(feature))
        feature = Dropout(0.3)(feature)
        m = BatchNormalization()(Conv2D(2 ** 9, (1, 1), strides=(1, 1), padding="same", activation="relu")(feature))
        multi_grid = m
        for ii in range(multi_grid_n):
            m = BatchNormalization()(Conv2D(2 ** 9, (3, 3), strides=(1, 1),
                                            dilation_rate=2 ** ii, padding="same", activation="relu"
                                            )(feature))
            multi_grid = concatenate([multi_grid, m])
        multi_grid = Dropout(0.3)(multi_grid)
        feature = Conv2D(2 ** 9, (1, 1), strides=(1, 1), padding="same")(multi_grid)
        layer_out.append(feature)

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


def seg_pnn(feature_num=128,
            timesteps=256,
            multi_grid_layer_n=5,
            multi_grid_n=3,
            prev_model="melody_transfer_transpose"
            ):

    layer_out = []
    input_score_48 = Input(shape=(timesteps, feature_num, 1), name="input_score_48")
    input_score_12 = Input(shape=(timesteps, feature_num // 3, 1), name="input_score_12")

    me_transfer_seg = seg(multi_grid_layer_n=1, timesteps=timesteps, prog = True)
    me_seg = load_model(prev_model)
    model_copy(me_seg, me_transfer_seg)

    #TODO: move inside model_copy
    for index, layer in enumerate(me_transfer_seg.layers):
        me_transfer_seg.layers[index].trainable = False

    o_p = me_transfer_seg([input_score_12])

    en_l = Conv2D(2 ** 5, (7, 7), strides=(1, 1), padding="same")(input_score_48)

    o = adapter(o_p[0], 2 ** (5), dropout_rate=0.2)
    en_l = add([en_l, o])

    en_l1 = conv_block(en_l, 2 ** 5, (3, 3), strides=(2, 2))
    en_l1 = conv_block(en_l1, 2 ** 5, (3, 3), strides=(1, 1))
    layer_out.append(en_l1)

    o = adapter(o_p[1], 2 ** (5), dropout_rate=0.2)
    en_l1 = add([en_l1, o])

    en_l2 = conv_block(en_l1, 2 ** 6, (3, 3), strides=(2, 2))
    en_l2 = conv_block(en_l2, 2 ** 6, (3, 3), strides=(1, 1))
    en_l2 = conv_block(en_l2, 2 ** 6, (3, 3), strides=(1, 1))
    layer_out.append(en_l2)

    o = adapter(o_p[2], 2 ** (6), dropout_rate=0.2)
    en_l2 = add([en_l2, o])

    en_l3 = conv_block(en_l2, 2 ** 7, (3, 3), strides=(2, 2))
    en_l3 = conv_block(en_l3, 2 ** 7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2 ** 7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2 ** 7, (3, 3), strides=(1, 1))
    layer_out.append(en_l3)

    o = adapter(o_p[3], 2 ** (7), dropout_rate=0.2)
    en_l3 = add([en_l3, o])

    en_l4 = conv_block(en_l3, 2 ** 8, (3, 3), strides=(2, 2))
    en_l4 = conv_block(en_l4, 2 ** 8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2 ** 8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2 ** 8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2 ** 8, (3, 3), strides=(1, 1))
    layer_out.append(en_l4)

    o = adapter(o_p[4], 2 ** (8), dropout_rate=0.2)
    en_l4 = add([en_l4, o])

    feature = en_l4

    for i in range(multi_grid_layer_n):

        feature = BatchNormalization()(Activation("relu")(feature))
        feature = Dropout(0.3)(feature)
        m = BatchNormalization()(Conv2D(2 ** 9, (1, 1), strides=(1, 1), padding="same", activation="relu")(feature))
        multi_grid = m
        for ii in range(multi_grid_n):
            m = BatchNormalization()(
                Conv2D(2 ** 9, (3, 3), strides=(1, 1), dilation_rate=2 ** ii, padding="same", activation="relu")(
                    feature))
            multi_grid = concatenate([multi_grid, m])
        multi_grid = Dropout(0.3)(multi_grid)
        feature = Conv2D(2 ** 9, (1, 1), strides=(1, 1), padding="same")(multi_grid)

        o = adapter(o_p[5], 2 ** (9), dropout_rate=0.3)
        feature = add([feature, o])

    feature = BatchNormalization()(Activation("relu")(feature))

    feature = Dropout(0.4)(feature)
    feature = Conv2D(2 ** 8, (1, 1), strides=(1, 1), padding="same")(feature)
    feature = add([feature, layer_out[3]])
    de_l1 = transpose_conv_block(feature, 2 ** 7, (3, 3), strides=(2, 2))

    o = adapter(o_p[6], 2 ** (7), kernel_size=(1, 5), dropout_rate=0.4)
    de_l1 = add([de_l1, o])

    skip = de_l1
    de_l1 = BatchNormalization()(Activation("relu")(de_l1))
    de_l1 = concatenate([de_l1, BatchNormalization()(Activation("relu")(layer_out[2]))])
    de_l1 = Dropout(0.4)(de_l1)
    de_l1 = Conv2D(2 ** 7, (1, 1), strides=(1, 1), padding="same")(de_l1)
    de_l1 = add([de_l1, skip])
    de_l2 = transpose_conv_block(de_l1, 2 ** 6, (3, 3), strides=(2, 2))

    o = adapter(o_p[7], 2 ** (6), kernel_size=(1, 5), dropout_rate=0.4)
    de_l2 = add([de_l2, o])

    skip = de_l2
    de_l2 = BatchNormalization()(Activation("relu")(de_l2))
    de_l2 = concatenate([de_l2, BatchNormalization()(Activation("relu")(layer_out[1]))])
    de_l2 = Dropout(0.4)(de_l2)
    de_l2 = Conv2D(2 ** 6, (1, 1), strides=(1, 1), padding="same")(de_l2)
    de_l2 = add([de_l2, skip])
    de_l3 = transpose_conv_block(de_l2, 2 ** 5, (3, 3), strides=(2, 2))

    o = adapter(o_p[8], 2 ** (5), kernel_size=(1, 5), dropout_rate=0.4)
    de_l3 = add([de_l3, o])

    skip = de_l3
    de_l3 = BatchNormalization()(Activation("relu")(de_l3))
    de_l3 = concatenate([de_l3, BatchNormalization()(Activation("relu")(layer_out[0]))])
    de_l3 = Dropout(0.4)(de_l3)
    de_l3 = Conv2D(2 ** 5, (1, 1), strides=(1, 1), padding="same")(de_l3)
    de_l3 = add([de_l3, skip])
    de_l4 = transpose_conv_block(de_l3, 2 ** 5, (3, 3), strides=(2, 2))

    o = adapter(o_p[9], 2 ** (5), kernel_size=(1, 5), dropout_rate=0.4)
    de_l4 = add([de_l4, o])

    de_l4 = BatchNormalization()(Activation("relu")(de_l4))
    de_l4 = Dropout(0.4)(de_l4)
    out = Conv2D(2, (1, 1), strides=(1, 1), padding="same", name='prediction')(de_l4)

    model = Model(inputs=[input_score_48, input_score_12],
                  outputs=out)

    return model
