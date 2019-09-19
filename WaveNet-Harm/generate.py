from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import librosa
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, audio_reader

SAMPLES = 16000
TEMPERATURE = 1.0
LOGDIR = './logdir'
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = 20 #100
SILENCE_THRESHOLD = 0.1


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--frames',
        type=int,
        default=SAMPLES,
        help='How many frames to generate')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--clip_id',
        type=str,
        default=None,
        help='the target audio segment id in the NIT dataset to generate, only set when scramble is not set.')
    parser.add_argument(
        '--scramble',
        action='store_true',
        help='Use scramble msfc, ap, f0 and phonemes sequence for local conditioning in generation.')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=False,
        help='Use fast generation')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    parser.add_argument(
        '--gc_channels',
        type=int,
        default=None,
        help='Number of global condition embedding channels. Omit if no '
             'global conditioning.')
    parser.add_argument(
        '--gc_cardinality',
        type=int,
        default=None,
        help='Number of categories upon which we globally condition.')
    parser.add_argument(
        '--gc_id',
        type=int,
        default=None,
        help='ID of category to generate, if globally conditioned.')
    arguments = parser.parse_args()
    if arguments.gc_channels is not None:
        if arguments.gc_cardinality is None:
            raise ValueError("Globally conditioning but gc_cardinality not "
                             "specified. Use --gc_cardinality=377 for full "
                             "VCTK corpus.")

        if arguments.gc_id is None:
            raise ValueError("Globally conditioning, but global condition was "
                              "not specified. Use --gc_id to specify global "
                              "condition.")

    return arguments


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))

def estimate_output(mvn1, mvn2, mvn3, mvn4, w1, w2, w3, w4, mfsc_channels):
    factor = 100
    
    # Empty output, to be concatenated
    output = tf.constant([], dtype = 'float32')
    
    for i in range(mfsc_channels):
        # Sample each multivariate normal w[i] number of times
        sample_1 = mvn1.sample(tf.cast(w1[i]*factor, dtype='int32'))
        sample_2 = mvn2.sample(tf.cast(w2[i]*factor, dtype='int32'))
        sample_3 = mvn3.sample(tf.cast(w3[i]*factor, dtype='int32'))
        sample_4 = mvn4.sample(tf.cast(w4[i]*factor, dtype='int32'))
        
        # Take the ith column of each of the sample_N tensors; returns column vector
        sample_1_sliced = tf.slice(sample_1, [0,i], [-1,1])
        sample_2_sliced = tf.slice(sample_2, [0,i], [-1,1])
        sample_3_sliced = tf.slice(sample_3, [0,i], [-1,1])
        sample_4_sliced = tf.slice(sample_4, [0,i], [-1,1])
        # Concatenate all samples from the different mixtures
        sample_sliced = tf.concat([sample_1_sliced, sample_2_sliced, sample_3_sliced, sample_4_sliced], axis=0)
        op_i = tf.reduce_mean(sample_sliced)
        
        # Concatenate with the other outputs
        output = tf.concat([output, [op_i]], axis = 0)
    
    # Output will be of shape (60,)
    return output


def create_seed(filename,
                sample_rate,
                quantization_channels,
                window_size,
                silence_threshold=SILENCE_THRESHOLD):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio_reader.trim_silence(audio, silence_threshold)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))

    return quantized[:cut_index]

def swap_chunk(input, first_chunk, second_chunk):
    """
    this function swaps two chunks of array.
    first_chunk and second_chunk are indices of the two chunks to be swapped. must have same length
    i.e. first_chunk[1] - first_chunk[0] must equal to second_chunk[1] - second_chunk[0]
    """
    output = np.copy(input)
    output[first_chunk[0]:first_chunk[1]] = input[second_chunk[0]:second_chunk[1]]
    output[second_chunk[0]:second_chunk[1]] = input[first_chunk[0]:first_chunk[1]]

    return output

def load_lc(clip_id=None, scramble=False):
    mfsc_path = "../data/mfsc/"
    ap_path = "../data/ap/"
    f0_path = "../data/f0coded/"
    prev_phone_path = "../data/phones/prev/"
    cur_phone_path = "../data/phones/current/"
    next_phone_path = "../data/phones/next/"
    phone_pos_path = "../data/phones/pos/"

    filename = "nitech_jp_song070_f001_"

    if clip_id:
        F0 = np.load(f0_path + filename + clip_id + ".npy")
        prev_phone = np.load(prev_phone_path + filename + clip_id + ".npy")
        cur_phone = np.load(cur_phone_path + filename + clip_id + ".npy")
        next_phone = np.load(next_phone_path + filename + clip_id + ".npy")
        phone_pos = np.load(phone_pos_path + filename + clip_id + ".npy")
        
        concat = np.concatenate((F0, prev_phone, cur_phone, next_phone, phone_pos), axis=-1)


    if scramble:
        F0 = np.load("../data/F0_scramble.npy")
        prev_phone = np.load("../data/new_phones/prev/concatenated.npy")
        current_phone = np.load("../data/new_phones/current/concatenated.npy")
        next_phone = np.load("../data/new_phones/next/concatenated.npy")
        phone_pos = np.load("../data/phone_scramble.npy")[:, -3:]
        concat = np.concatenate((F0, prev_phone, current_phone, next_phone, phone_pos), axis=-1)

    return concat


def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        use_biases=wavenet_params["use_biases"],
        scalar_input=wavenet_params["scalar_input"],
        initial_filter_width=wavenet_params["initial_filter_width"],
        histograms=False,
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=args.gc_cardinality,
        MFSC_channels=wavenet_params["MFSC_channels"],
        F0_channels=wavenet_params["F0_channels"],
        phone_channels=wavenet_params["phones_channels"],
        phone_pos_channels=wavenet_params["phone_pos_channels"])

    samples = tf.placeholder(tf.float32)
    lc = tf.placeholder(tf.float32)
    
    MFSC_channels = wavenet_params["MFSC_channels"]

    if args.fast_generation:
        next_sample = net.predict_proba_incremental(samples,  args.gc_id)         ########### understand shape of next_sample
    else:
        outputs = net.predict_proba(samples, MFSC_channels, lc, args.gc_id)
        outputs = tf.reshape(outputs, [1, MFSC_channels])

    if args.fast_generation:
        sess.run(tf.global_variables_initializer())
        sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    if args.wav_seed:
        # seed = create_seed(args.wav_seed,
        #                    wavenet_params['sample_rate'],
        #                    quantization_channels,
        #                    net.receptive_field)
        # waveform = sess.run(seed).tolist()
        pass
    else:

        # Silence with a single random sample at the end.
        waveform = np.zeros((net.receptive_field - 1, MFSC_channels))
        waveform = np.append(waveform, np.random.randn(1, MFSC_channels), axis=0)

        lc_array = load_lc(clip_id=args.clip_id, scramble=args.scramble) # clip_id:[003, 004, 007, 010, 012 ...]
        lc_array = np.pad(lc_array, ((net.receptive_field, 0), (0, 0)), 'constant', constant_values=((0, 0),(0,0)))

    if args.fast_generation and args.wav_seed:
        # When using the incremental generation, we need to
        # feed in all priming samples one by one before starting the
        # actual generation.
        # TODO This could be done much more efficiently by passing the waveform
        # to the incremental generator as an optional argument, which would be
        # used to fill the queues initially.
        outputs = [next_sample]
        outputs.extend(net.push_ops)                                   ################# understand net.push_ops, understand shape of outputs

        print('Priming generation...')
        for i, x in enumerate(waveform[-net.receptive_field: -1]):
            if i % 100 == 0:
                print('Priming sample {}'.format(i))
            sess.run(outputs, feed_dict={samples: x})
        print('Done.')

    last_sample_timestamp = datetime.now()
    for step in range(args.frames):
        if args.fast_generation:
            outputs = [next_sample]
            outputs.extend(net.push_ops)
            window = waveform[-1]
        else:

            if len(waveform) > net.receptive_field:
                window = waveform[-net.receptive_field:, :]
            else:
                window = waveform

        # Run the WaveNet to predict the next sample.
        window = window.reshape(1, window.shape[-2], window.shape[-1])
        prediction = sess.run(outputs, feed_dict={samples: window, 
                                                lc: lc_array[step+1: step+1+net.receptive_field, :].reshape(1, net.receptive_field, -1)})
        waveform = np.append(waveform, prediction, axis=0)

        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Frame {:3<d}/{:3<d}'.format(step + 1, args.frames),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

        # If we have partial writing, save the result so far.
        if (args.wav_out_path and args.save_every and
                (step + 1) % args.save_every == 0):
            np.save(args.wav_out_path, waveform)

    # Introduce a newline to clear the carriage return from the progress.
    print()

    # Save the result as a numpy file.
    if args.wav_out_path:
        np.save(args.wav_out_path, waveform)

    print('Finished generating. The result was saved as .npy file.')


if __name__ == '__main__':
    main()
