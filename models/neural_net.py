# 2. Set `python` built-in pseudo-random generator at a fixed value
seed_value= 0
import random
import os
random.seed(seed_value)
#
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# # 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization

from tensorflow.keras.models import Sequential, load_model, model_from_json
# from tensorflow.keras.utils import np_utils
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l1
# from keras import backend as K
from tensorflow.keras import optimizers

from tensorflow.keras import metrics

import pickle

def top_k_precion(model, data, data_X, k=20):
    pre = model.predict_proba(data_X)
    pre_and_y = data.loc[:, ['t', 'feature_name', 'acc_bin', 'ig']]
    # pre_and_y.sort_values('t', ascending=False)
    pre_and_y['pre'] = pre
    pre_and_y = pre_and_y.groupby(['feature_name'], as_index=False).mean()
    # pre_and_y = pre_and_y.drop_duplicates('feature_name')
    pre_feat = pre_and_y.sort_values('pre', ascending=False)['feature_name'][:k].values
    t_feat = pre_and_y.sort_values('t', ascending=False)['feature_name'][:k].values
    ig_faet = pre_and_y.sort_values('ig', ascending=False)['feature_name'][:k].values

    top_20 = (np.isin(pre_feat, t_feat).sum(), np.isin(ig_faet, t_feat).sum())
    top_10 = (np.isin(pre_feat[:10], t_feat[:10]).sum(), np.isin(ig_faet[:10], t_feat[:10]).sum())
    top_5 = (np.isin(pre_feat[:5], t_feat[:5]).sum(), np.isin(ig_faet[:5], t_feat[:5]).sum())

    print('fitted: ', np.isin(pre_feat, t_feat).sum())
    print('ig: ', np.isin(ig_faet, t_feat).sum())
    return top_20, top_10, top_5



def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def binary_focal_loss(gamma=5, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)

        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise

        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise

        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """

    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss

    return focal_loss


precision = tf.keras.metrics.Precision()


def nural_net(input_shape, file_for_wights='NN_wights.data', LR=0.0001, focal_loss=True,layers=[15,10,5]):
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(input_shape,), activity_regularizer=l1(0.0001)))
    model.add(Dropout(0.25))
    model.add(Dense(layers[1], activation='relu', activity_regularizer=l1(0.0001)))
    model.add(Dropout(0.25))
    model.add(Dense(layers[2], activation='relu', activity_regularizer=l1(0.0001)))
    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(lr=LR)
    if focal_loss == True: loss = [binary_focal_loss()]
    else: loss = 'binary_crossentropy'
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=[tf.keras.metrics.Precision()]
                  )

    if not os.path.exists(file_for_wights):
        # save initial wights - only once
        with open(file_for_wights, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(model.get_weights(), filehandle)


    with open(file_for_wights, 'rb') as filehandle:
        # read the data as binary data stream
        nuralnet_wights = pickle.load(filehandle)
    model.set_weights(nuralnet_wights)

    return model


