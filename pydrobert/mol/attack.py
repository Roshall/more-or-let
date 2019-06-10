import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from timeit import default_timer as timer
from helper import get_logits
from model import  _tf_dft_ctc_loss, _tf_dft_ctc_decode



def ctc_lost(target_phrase,target_phrase_lengths, batch_size):
    pass


class Attack:
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                 learning_rate=10, num_iterations=5000, batch_size=1,
                 mp3=False, l2penalty=float('inf')):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """

        self.sess = sess
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len
        self.mp3 = mp3

        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.
        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32),
                                               name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size, 1), dtype=np.float32), name='qq_phrase_lengths')

        # Initially we bound the l_infty norm by 2000, increase this
        # constant if it's not big enough of a distortion for your dataset.
        self.apply_delta = tf.clip_by_value(delta, -2000, 2000) * self.rescale

        # We set the new input to the model to be the abve delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        self.new_input = new_input = self.apply_delta * mask + original

        # We add a tiny bit of noise to help make sure that we can
        # clip our values to 16-bit integers and not break things.
        noise = tf.random_normal(new_input.shape,
                                 stddev=2)
        pass_in = tf.clip_by_value(new_input + noise, -2 ** 15, 2 ** 15 - 1)

        # Feed this final value to get the logits.
        self.logits = logits = get_logits(pass_in)

        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, "data/session_dump")

        # Choose the loss function we want -- either CTC or CW
        self.loss_fn = loss_fn
        if loss_fn == "CTC":

            ctcloss = _tf_dft_ctc_loss(self.target_phrase, logits, lengths, self.target_phrase_lengths)

            # Slight hack: an infinite l2 penalty means that we don't penalize l2 distortion
            # The code runs faster at a slight cost of distortion, and also leaves one less
            # paramaeter that requires tuning.
            if not np.isinf(l2penalty):
                loss = tf.reduce_mean((self.new_input - self.original) ** 2, axis=1) + l2penalty * ctcloss
            else:
                loss = ctcloss
            self.expanded_loss = tf.constant(0)

        elif loss_fn == "CW":
            raise NotImplemented(
                "The current version of this project does not include the CW loss function implementation.")
        else:
            raise ValueError("loss function does not exit.")

        self.loss = loss
        self.ctcloss = ctcloss

        # Set up the Adam optimizer to perform gradient descent for us
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate)

        grad, var = optimizer.compute_gradients(self.loss, [delta])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad), var)])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        sess.run(tf.variables_initializer(new_vars + [delta]))

        # Decoder from the logits, to see how we're doing
        # logits = tf.transpose(logits, perm=[1, 0, 2])
        # self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=True, beam_width=100)
        #new decoded
        self.decoded = _tf_dft_ctc_decode(logits, lengths)[0]
    def attack(self, audio, lengths, target, finetune=None):
        sess = self.sess
        frames_nums=[]
        for length in lengths:
            frames_nums.append(int(np.ceil(float(np.abs(length - 400)) / 160)) +1)
        # Initialize all of the variables
        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        # attack() a bunch of times.
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign(np.array(frames_nums)))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.cwmask.assign(
            np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in np.array(frames_nums)])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t) + [0] * (self.phrase_length - len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size, 1))))

        # Here we'll keep track of the best solution we've found so far
        final_deltas = [None] * self.batch_size

        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune - audio))

        # We'll make a bunch of iterations of gradient descent here
        # now = time.time()
        MAX = self.num_iterations
        for i in range(MAX):
            # iteration = i
            # now = time.time()
            #
            # Print out some debug information every 10 iterations.
            # if i % 10 == 0:
            #     r_out = sess.run((self.decoded))
                # lst = [(r_out, r_logits)]

                # for out, logits in lst:
                #
                #     res = np.zeros(out[0].dense_shape) + 61
                #
                #     for ii in range(len(out[0].values)):
                #         x, y = out[0].indices[ii]
                #         res[x, y] = out[0].values[ii]

                    # Here we print the strings that are recognized.
                # print(r_out)

                    # And here we print the argmax of the alignment.

            feed_dict = {}

            # Actually do the optimization ste
            sess.run(( self.train), feed_dict)

            # Report progress
            # print("%.3f" % np.mean(cl), "\t", "\t".join("%.3f" % x for x in cl))

            # logits = np.argmax(logits, axis=2).T
            if (i + 1) % 100 == 0:
                d, new_input, res, cl = sess.run(( self.delta, self.new_input,self.decoded, self.ctcloss))
                print("%.3f" % np.mean(cl), "\t", "\t".join("%.3f" % x for x in cl))
                # lst = [(r_out, r_logits)]

                # for out, logits in lst:
                #
                #     res = np.zeros(out[0].dense_shape) + 62
                #
                #     for ii in range(len(out[0].values)):
                #         x, y = out[0].indices[ii]
                #         res[x, y] = out[0].values[ii]
                #
                #     print(res)


                for ii in range(self.batch_size):
                    # Every 100 iterations, check if we've succeeded
                    # if we have (or if it's the final epoch) then we
                    # should record our progress and decrease the
                    # rescale constant.
                    print(res[ii])
                    if (self.loss_fn == "CTC" and list(res[ii]) == list(target[ii]) and cl[ii] < 1e-2) \
                            or (i == MAX - 1 and final_deltas[ii] is None):
                        # Get the current constant
                        rescale = sess.run(self.rescale)
                        if rescale[ii] * 2000 > np.max(np.abs(d)):
                            # If we're already below the threshold, then
                            # just reduce the threshold to the current
                            # point and save some time.
                            # print("It's way over", np.max(np.abs(d[ii])) / 2000.0)
                            rescale[ii] = np.max(np.abs(d[ii])) / 2000.0

                        # Otherwise reduce it by some constant. The closer
                        # this number is to 1, the better quality the result
                        # will be. The smaller, the quicker we'll converge
                        # on a result but it will be lower quality.
                        rescale[ii] *= .8

                        # Adjust the best solution found so far
                        final_deltas[ii] = new_input[ii]
                        # print("Worked i=%d ctcloss=%f bound=%f" % (ii, cl[ii], 2000 * rescale[ii][0]))
                        # print('delta',np.max(np.abs(new_input[ii]-audio[ii])))
                        sess.run(self.rescale.assign(rescale))

                        # # Just for debugging, save the adversarial example
                        # # to /tmp so we can see it if we want
                        # wav.write("/tmp/adv.wav", 16000,
                        #           np.array(np.clip(np.round(new_input[ii]),
                        #                            -2 ** 15, 2 ** 15 - 1), dtype=np.int16))

        return final_deltas

def main():
    target = np.arange(1,10)
    iterations = 300
    path = 'data/si1559_adv.wav'
    sample_rate, signal = wav.read('data/si1559_.wav')
    audios = [list(signal)]
    lengths = [len(signal)]
    with tf.Session() as sess:
        maxlen = max(map(len,audios))
        audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])

        start_attack = timer()
        attack = Attack(sess, 'CTC', len(target), maxlen,
                    batch_size=len(audios),
                    num_iterations=iterations)
        delta = attack.attack(audios,
                            lengths,
                            [target]*len(audios))

    wav.write(path, 16000,
              np.array(np.clip(np.round(delta[0][:lengths[0]]),
                               -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
    end_attack = timer() - start_attack
    print('attacking time: %0.3fs' %(end_attack))



if __name__ == "__main__":
    main()