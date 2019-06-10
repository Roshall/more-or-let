from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io.wavfile
import tensorflow as tf
import model as cnnmodel


import keras.backend as K
import numpy as np

from keras.initializers import RandomUniform
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers.merge import Maximum
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model

def tf_calculate_deltas(data, num_deltas, axis=0, target_axis=-1):
    max_filt_width = 4 * num_deltas + 1
    pad_width = (max_filt_width - 1) // 2
    pad_widths = [(0, 0)] * len(data.shape)
    pad_widths = np.array(pad_widths)
    pad_widths[axis] = (1, 1)
    slices = [slice(None)]* len(data.shape)
    slices[axis] = slice(pad_width, -pad_width)
    slices = tuple(slices)
    delta_data_list = [data]
    padded_data = tf.pad(data, pad_widths,'symmetric')
    padded_data = tf.pad(padded_data, pad_widths, 'symmetric')
    pad_widths[axis] *= 2
    padded_data = tf.pad(padded_data, pad_widths, 'symmetric')
    delta_filt = np.asarray((.2, .1, 0., -.1, -.2))
    cur_filt = np.ones(1)
    padded_data = tf.expand_dims(padded_data, -1)
    for i in range(num_deltas):
        cur_filt = np.convolve(cur_filt, delta_filt, 'full')
        conv_filt = cur_filt[..., np.newaxis, np.newaxis, np.newaxis]
        delta_data_list.append(tf.squeeze(tf.nn.conv2d(padded_data,conv_filt,strides=[1,1,1,1],padding='SAME'), -1)[slices])
    return tf.concat(delta_data_list, target_axis)

def tf_get_feature(data,sample_rate=16000):
    # now we know how to comput feature via numpy
    # we do it identically via tensorflow.
    NFFT = 512
    pre_emphasis = 0.97
    frame_length = int(sample_rate * 0.025)
    frame_stride = int(sample_rate * 0.01)
    hanning_widow = np.hanning(frame_length)

    batch_size, data_length = data.get_shape().as_list()
    data = tf.cast(data, tf.float32)
    frames_num = int(np.ceil(float(np.abs(data_length - frame_length)) / frame_stride))+1

    # step 1. pre-emphasis and padding
    data = tf.concat((data[:, :1], data[:, 1:] - pre_emphasis * data[:, :-1],
                      np.zeros((batch_size, frames_num * frame_stride + frame_length-data_length), dtype=np.float32)), 1)
    data_length = frames_num * frame_stride

    # step 2. framing
    windowed = tf.stack([data[:, i:i + frame_length] for i in range(0, data_length, frame_stride)], 1)

    # step 3. add window
    windowed *= hanning_widow

    #step 4.take FFT to convert to frequency space
    ffted = tf.spectral.rfft(windowed,[NFFT])
    pow_frames = tf.square(tf.abs(ffted))

    # step 5. compute mal fbank
    fbanks = np.load("fbank_40.npy").T
    filter_banks = tf.matmul(pow_frames,np.array([fbanks]*batch_size,dtype=np.float32))

    # step 5.2. compute energy
    energy = tf.reduce_sum(pow_frames,axis=2)
    # step 6. concatenate energy to fbank
    feature = tf.concat((tf.reshape(energy, (-1, frames_num, 1)), filter_banks), axis=2) + 1e-30
    feature = tf.log(feature)
    feature = tf_calculate_deltas(feature, 2, 1)

    return feature

def calculate_deltas(data, num_deltas, axis=0, target_axis=-1, concatenate=True):
    '''Calculate deltas for arrays

    Deltas are simply weighted rolling averages; double deltas are the
    rolling averages of rolling averages. This can be done an arbitrary
    number of times. Because most datas are non-zero in silence, the
    data is edge-padded before convolution.

    Parameters
    ----------
    data : array-like
        At least one dimensional
    num_deltas : int
        A nonnegative integer specifying the number of delta calculations
    axis : int
        The axis of `data` to be calculated over (i.e. convolved)
    target_axis : int
        The location where the new axis, for deltas, will be inserted
    concatenate : bool
        Whether to concatenate the deltas to the end of `target_axis` (`True`),
        or create a new axis in this location (`False`)

    Returns
    -------
    array-like
    '''
    max_filt_width = 4 * num_deltas + 1
    pad_width = (max_filt_width - 1) // 2
    pad_widths = [(0, 0)] * len(data.shape)
    pad_widths = np.array(pad_widths)
    pad_widths[axis] = (pad_width, pad_width)
    slices = [slice(None)]
    slices[axis] = slice(pad_width, -pad_width)
    delta_data_list = [data]
    padded_data = np.pad(data, pad_widths,'edge')
    delta_filt = np.asarray((.2, .1, 0., -.1, -.2))
    cur_filt = np.ones(1)
    for i in range(num_deltas):
        cur_filt = np.convolve(cur_filt, delta_filt, 'full')
        delta_data_list.append(np.apply_along_axis(
            np.convolve, axis, padded_data, cur_filt, 'same')[slices])
    if concatenate:
        return np.concatenate(delta_data_list, target_axis)
    else:
        return np.stack(delta_data_list, target_axis)

def get_feature(data,sample_rate=16000):
    NFFT = 512
    pre_emphasis = 0.97
    frame_length = int(sample_rate * 0.025)
    frame_stride = int(sample_rate * 0.01)
    hanning_widow = np.hanning(frame_length)
    data_length = len(data)
    frames_num = int(np.ceil(float(np.abs(data_length - frame_length)) / frame_stride)) + 1

    # step 1. pre-emphasis and padding
    data = np.concatenate((data[:1], data[1:] - pre_emphasis * data[:-1], np.zeros(frames_num * frame_stride + frame_length-data_length, dtype=np.float32)), 0)
    data_length = frames_num * frame_stride

    # step 2. framing
    windowed = np.stack([data[i:i + frame_length] for i in range(0, data_length, frame_stride)], 0)

    # step 3. add window
    windowed *= hanning_widow

    #step 4.take FFT to convert to frequency space
    ffted = np.fft.rfft(windowed, NFFT)
    pow_frames = np.square(np.abs(ffted))

    # step 5. compute mal fbank
    fbank = np.load("fbank_40.npy")
    filter_banks = np.dot(pow_frames, fbank.T)

    # step 5.2. compute energy
    energy = np.sum(pow_frames, 1)

    # step 6. concatenate energy to fbank
    feature = np.concatenate((energy.reshape(-1,1),filter_banks),1) + 1e-30
    feature = np.log(feature)
    feature = calculate_deltas(feature, 2)

    return feature

def load_cnnmodel():
    sample_rate, signal = scipy.io.wavfile.read('data/si1559_adv.wav')
    audio = get_feature(signal, sample_rate)

    # original = tf.Variable(np.zeros((1,len(signal)),dtype=np.float32))
    # audio2  = tf_get_feature(original, sample_rate)
    # with tf.Session() as sess:
    #     sess.run(original.assign(np.array([signal])))
    #     re = sess.run(audio2)
    # print(re)

    model = load_model("../../exp/model/kaldi_41-0300.1.h5", custom_objects={
        '_ctc_loss': cnnmodel._tf_dft_ctc_loss,
        '_y_pred_loss': cnnmodel._y_pred_loss })

    label2id_map = dict()
    with open('data/phn_id.map') as f:
        for line in f:
            label,idee = line.strip().split()
            idee = int(idee)
            label2id_map[idee] = label

    length_batch = np.asarray([[audio.shape[0]]], dtype=np.int32)

    label_out = ctc_decode(
        model.get_layer(name='dense_activation_td').output,
        model.get_layer(name='feat_size_in').output
    )

    decoder = K.function(
        [
            model.get_layer(name='feat_in').output,
            model.get_layer(name='feat_size_in').output,
            K.learning_phase(),
        ],
        label_out,
    )

    ret_id=decoder([audio[np.newaxis,:,:,np.newaxis],length_batch,0])[0][0]
    # ret_id=decoder([audio[np.newaxis,:,:,np.newaxis],length_batch,0])

    # ret_labels = [label2id_map[idee] for idee in ret_id]

    return ret_id

def cnn_model(batch_x):
    # construct an acoustic model from scratch
    config={'filt_time_width': 5, 'filt_freq_width': 3, 'filt_freq_stride': 1,
            'filt_time_stride': 1, 'cur_weight_seed': 1234, 'num_fea': 41, 'delta_order': 0,
            'init_num_filt_channels': 64, 'pool_time_width': 1, 'pool_freq_width': 3,
            'num_dense_hidden': 512, 'num_labels': 62}

    def _layer_kwargs():
        ret = {
            'activation': 'linear',
            'kernel_initializer': RandomUniform(
                minval=-.05,
                maxval=.05,
                seed= config['cur_weight_seed'],
            ),
        }
        config['cur_weight_seed'] = config['cur_weight_seed'] + 1
        return ret

    # convolutional layer pattern
    def _conv_maxout_layer(last_layer, n_filts, name_prefix):
        conv_a = Conv2D(
            n_filts,
            (config['filt_time_width'], config['filt_freq_width']),
            strides=(
                config['filt_time_stride'],
                config['filt_freq_stride'],
            ),
            padding='same',
            name=name_prefix + '_a',
            **_layer_kwargs()
        )(last_layer)
        conv_b = Conv2D(
            n_filts,
            (config['filt_time_width'], config['filt_freq_width']),
            strides=(
                config['filt_time_stride'],
                config['filt_freq_stride'],
            ),
            padding='same',
            name=name_prefix + '_b',
            **_layer_kwargs()
        )(last_layer)
        last = Maximum(name=name_prefix + '_m')([conv_a, conv_b])
        return last

    # inputs
    # feat_input = Input(
    #     shape=(
    #         None,
    #         config['num_feats'] * (1 + config['delta_order']),
    #         1,
    #     ),
    #     name='feat_in',
    # )
    last_layer = tf.expand_dims(batch_x,-1)
    # convolutional layers
    n_filts = config['init_num_filt_channels']
    last_layer = _conv_maxout_layer(
        last_layer, n_filts, 'conv_1')
    last_layer = MaxPooling2D(
        pool_size=(
            config['pool_time_width'],
            config['pool_freq_width']),
        name='conv_1_p')(last_layer)

    for layer_no in range(2, 11):
        if layer_no == 5:
            n_filts *= 2
        last_layer = _conv_maxout_layer(
            last_layer, n_filts, 'conv_{}'.format(layer_no))
    last_layer = Lambda(
        lambda layer: K.max(layer, axis=2),
        output_shape=(None, n_filts),
        name='max_freq_into_channel',
    )(last_layer)
    # dense layers
    for layer_no in range(1, 4):
        name_prefix = 'dense_{}'.format(layer_no)
        dense_a = Dense(
            config['num_dense_hidden'], name=name_prefix + '_a',
            **_layer_kwargs()
        )
        dense_b = Dense(
            config['num_dense_hidden'], name=name_prefix + '_b',
            **_layer_kwargs()
        )
        td_a = TimeDistributed(
            dense_a, name=name_prefix + '_td_a')(last_layer)
        td_b = TimeDistributed(
            dense_b, name=name_prefix + '_td_b')(last_layer)
        last_layer = Maximum(name=name_prefix + '_m')([td_a, td_b])

    activation_dense = Dense(
        config['num_labels'], name='dense_activation',
        **_layer_kwargs()
    )
    activation_layer = TimeDistributed(
        activation_dense, name='dense_activation_td')(last_layer)

    return activation_layer

def get_logits(new_input):
    features = tf_get_feature(new_input)
    logits = cnn_model(features)

    return logits

def ctc_decode(y_pred, input_length, beam_width=100):
    input_length = tf.reshape(input_length, [-1])
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    (decoded,), _ = tf.nn.ctc_beam_search_decoder(
        inputs=y_pred,
        sequence_length=input_length,
        beam_width=beam_width,
        top_paths=1,
    )
    decoded_dense = tf.sparse.to_dense(decoded, default_value=-1)
    return (decoded_dense,)


if __name__ == "__main__":
    ret_id = load_cnnmodel()
    print(ret_id)