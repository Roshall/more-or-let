from tensorflow.core.framework.graph_pb2 import *
import numpy as np
import tensorflow as tf

from helper import *
graph_def = GraphDef()

loaded = graph_def.ParseFromString(open("data/cnn_model.pb", "rb").read())

with tf.Graph().as_default() as graph:
    new_input = tf.placeholder(tf.float32,[None,None,None,None],
                               name="new_input")

    logits, = tf.import_graph_def(
        graph_def,
        input_map={"feat_in:0":new_input},
        return_elements=["dense_activation_td/Reshape_1:0"],
        name="newname"
    )

    with tf.Session(graph=graph) as sess:
        #original model
        sample_rate, signal = scipy.io.wavfile.read('data/si1559_.wav')
        audio = get_feature(signal, sample_rate)

        model = load_model("../../exp/model/kaldi_41-0300.1.h5", custom_objects={
            '_ctc_loss': cnnmodel._tf_dft_ctc_loss,
            '_y_pred_loss': cnnmodel._y_pred_loss})

        length_batch = np.asarray([[audio.shape[0]]], dtype=np.int32)

        decoder = K.function(
            [
                model.get_layer(name='feat_in').output,
                model.get_layer(name='feat_size_in').output,
                K.learning_phase(),
            ],
            [model.get_layer(name='dense_activation_td').output]
        )

        ret_id = np.array(decoder([audio[np.newaxis, :, :, np.newaxis], length_batch, 0])).flatten()

        save_list =[]
        for var in tf.global_variables():
            try:
                sess.run(var.assign(sess.run('newname/'+var.name)))
                save_list.append(var)
            except Exception as e:
                continue

        res = (sess.run(logits,{new_input:audio[np.newaxis, :, :, np.newaxis],
                                'newname/feat_size_in:0':length_batch})).flatten()

        saver = tf.train.Saver()
        saver.save(sess, "data/session_dump")