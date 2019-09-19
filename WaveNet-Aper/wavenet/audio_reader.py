import fnmatch
import os
import random
import re
import threading

import librosa
import numpy as np
import tensorflow as tf

FILE_PATTERN = r'([a-z]+)_([a-z]+)_song([0-9]+)_f([0-9]+)_([0-9]+)\.npy'########################################################
mfsc_path = "../data/mfsc/"
ap_path = "../data/ap/"
f0_path = "../data/f0coded/"
prev_phone_path = "../data/phones/prev/"
cur_phone_path = "../data/phones/current/"
next_phone_path = "../data/phones/next/"
phone_pos_path = "../data/phones/pos/"


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]  
        recording_id, id, clip_id = [int(id_) for id_ in matches[-3:]]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]

def find_files(directory, pattern="*.npy"):  
    
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(filename)          ########## return pure filename, instead of full path
    return files


def load_acous_F0_phones(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    """
    directory is the path of mfsc feat. same file names will be matched under f0, phones paths
    """
    
    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][3])


        mfsc = np.load(mfsc_path + filename)
        ap = np.load(ap_path + filename)
        F0 = np.load(f0_path + filename)

        prev_phone = np.load(prev_phone_path + filename)
        cur_phone = np.load(cur_phone_path + filename)
        next_phone = np.load(next_phone_path + filename)
        phone_pos = np.load(phone_pos_path + filename)
        yield mfsc, ap, F0, prev_phone, cur_phone, next_phone, phone_pos, filename, category_id


def trim_silence(audio, threshold, frame_length=2048):         
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self, 
                 audio_dir,
                 coord,
                 sample_rate,     
                 gc_enabled,
                 receptive_field,
                 sample_size = 32 ,         
                 mfsc_dim = 60,
                 ap_dim = 4,
                 F0_dim = 3,
                 phone_dim = 34,
                 phone_pos_dim = 3,
                 silence_threshold = None,   
                 queue_size = 32):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.ap_dim = ap_dim
        self.mfsc_dim = mfsc_dim
        self.F0_dim=F0_dim
        self.phone_dim = phone_dim
        self.phone_pos_dim = phone_pos_dim

        ##### modify below two lines when training different submodels
        self.acou_feat_dim = self.ap_dim + self.mfsc_dim     ###### non-control inputs to net, e.g. mfsc/ap/uv
        self.lc_dim = F0_dim + 3 * phone_dim + phone_pos_dim   ###### local conditioning inputs to net


        self.gc_enabled = gc_enabled
        self.threads = []
        self.acous_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.lc_placeholder = tf.placeholder(dtype=tf.float32, shape=None) 
        ############################ non-control input queue
        self.acous_queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, self.acou_feat_dim)])
        self.acous_enqueue = self.acous_queue.enqueue([self.acous_placeholder])
        ############################# control input queue
        self.lc_queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, self.lc_dim)])
        self.lc_enqueue = self.lc_queue.enqueue([self.lc_placeholder])
        ############################# add phonetic 
        

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self, num_elements):
        acous_output = self.acous_queue.dequeue_many(num_elements)
        lc_output = self.lc_queue.dequeue_many(num_elements)
        return acous_output, lc_output

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_acous_F0_phones(self.audio_dir, self.sample_rate)
            for mfsc, ap, F0, prev_phone, cur_phone, next_phone, phone_pos, filename, category_id in iterator:
                print (filename)
                if self.coord.should_stop():
                    stop = True
                    print ("threads stopped.")
                    break
                mfsc = mfsc.reshape(-1, self.mfsc_dim)
                ap = ap.reshape(-1, self.ap_dim)
                F0 = F0.reshape(-1, self.F0_dim)
                prev_phone = prev_phone.reshape(-1, self.phone_dim)
                cur_phone = cur_phone.reshape(-1, self.phone_dim)
                next_phone = next_phone.reshape(-1, self.phone_dim)
                phone_pos = phone_pos.reshape(-1, self.phone_pos_dim)

                acous = np.concatenate((ap, mfsc), axis=-1)
                ########### concatnate all control inputs to one array
                lc = np.concatenate((F0, prev_phone, cur_phone, next_phone, phone_pos), axis=-1)
                
                acous = np.pad(acous, [[self.receptive_field, 0], [0, 0]],'constant')    # after padding, shape of audio is [receptive_field + m, 60]
                lc = np.pad(lc, [[self.receptive_field, 0], [0, 0]],'constant')
        
                if self.sample_size:
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    while acous.shape[0] > self.receptive_field:
                        acous_piece = acous[:(self.receptive_field +     
                                        self.sample_size), :]      

                        lc_piece = lc[:(self.receptive_field +     
                                        self.sample_size), :]       

                        sess.run([self.acous_enqueue, self.lc_enqueue],
                                 feed_dict={self.acous_placeholder: acous_piece,\
                                             self.lc_placeholder: lc_piece})

                        acous = acous[self.sample_size:, :]
                        lc = lc[self.sample_size:, :]
                        

                        if self.gc_enabled:
                            sess.run(self.gc_enqueue, feed_dict={
                                self.id_placeholder: category_id})
                else:
                    sess.run([self.acous_enqueue, self.lc_enqueue],
                             feed_dict={self.acous_placeholder: acous,\
                                         self.lc_placeholder: lc})
                    if self.gc_enabled:
                        sess.run(self.gc_enqueue,
                                 feed_dict={self.id_placeholder: category_id})


    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
