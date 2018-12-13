###TODO add bias control

import numpy as np
import tensorflow as tf

from .ops import causal_conv
from .CMG import get_mixture_coef, temp_control, nll_loss
import pdb

def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        return tf.Variable(initial_val, name=name)
    else:
        return create_variable(name, shape)


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)

def add_gaussian_noise(input_tensor, noise_level = 0.4):
    """
    The way to regularize as NPSS paper
    """
    noise = tf.random_normal(shape = tf.shape(input_tensor), 
                            mean=0.0, stddev=1.0,
                            dtype = tf.float32)
    return input_tensor + noise_level * noise

def norm_data(data, norm_app = "min/max"):
    ##### norm to (0,1)
    if norm_app == "min/max":
        min_val = tf.reduce_min(data, axis=1)
        max_val = tf.reduce_max(data, axis=1)
        return (data - min_val) / (max_val - min_val)
    ##### norm to (-1,1)
    elif norm_app == "neg1/pos1":
        return data / tf.max(tf.abs(data)) 
    else:
        print("Unsupported norm method! Your options: min/max or neg1/pos1")
        print("Using min/max instead.")
        min_val = tf.reduce_min(data, axis=1)
        max_val = tf.reduce_max(data, axis=1)
        return (data - min_val) / (max_val - min_val)

class WaveNetModel(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 quantization_channels=2**8,
                 use_biases=False,
                 scalar_input=False,
                 initial_filter_width=10,
                 histograms=False,
                 global_condition_channels=None,
                 global_condition_cardinality=None,
                 MFSC_channels=60,
                 F0_channels=None,
                 phone_channels=None,
                 phone_pos_channels=None):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            scalar_input: Whether to use the quantized waveform directly as
                input to the network instead of one-hot encoding it.
                Default: False.
            initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input. This is only relevant
                if scalar_input=True.
            histograms: Whether to store histograms in the summary.
                Default: False.
            global_condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
            global_condition_cardinality: Number of mutually exclusive
                categories to be embedded in global condition embedding. If
                not None, then this implies that global_condition tensor
                specifies an integer selecting which of the N global condition
                categories, where N = global_condition_cardinality. If None,
                then the global_condition tensor is regarded as a vector which
                must have dimension global_condition_channels.

        '''
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width
        self.histograms = histograms
        self.global_condition_channels = global_condition_channels
        self.global_condition_cardinality = global_condition_cardinality
        self.MFSC_channels = MFSC_channels 
        self.F0_channels = F0_channels
        self.phone_channels = phone_channels
        self.phone_pos_channels = phone_pos_channels   # num of phoneme position channels
        self.lc_channels = F0_channels + 3 * phone_channels + phone_pos_channels
        self.CMG_channels = 4 * MFSC_channels   # each output dimention will be modeled by 4 paras of CMGs

        self.receptive_field = WaveNetModel.calculate_receptive_field(
            self.filter_width, self.dilations, self.scalar_input,
            self.initial_filter_width)
        self.variables = self._create_variables()
        
    @staticmethod
    def calculate_receptive_field(filter_width, dilations, scalar_input,
                                  initial_filter_width):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += initial_filter_width - 1      # modified here
        return receptive_field

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('wavenet'):
            if self.global_condition_cardinality is not None:
                # We only look up the embedding if we are conditioning on a
                # set of mutually-exclusive categories. We can also condition
                # on an already-embedded dense vector, in which case it's
                # given to us and we don't need to do the embedding lookup.
                # Still another alternative is no global condition at all, in
                # which case we also don't do a tf.nn.embedding_lookup.
                with tf.variable_scope('embeddings'):
                    layer = dict()
                    layer['gc_embedding'] = create_embedding_table(
                        'gc_embedding',
                        [self.global_condition_cardinality,
                         self.global_condition_channels])
                    var['embeddings'] = layer          # layer itself is a dict, i.e. var["embeddings"] is a dict

            with tf.variable_scope('causal_layer'):
                layer = dict()
                if self.scalar_input:
                    initial_channels = 1
                    initial_filter_width = self.initial_filter_width
                else:
                    initial_channels = self.MFSC_channels
                    initial_filter_width = self.initial_filter_width
                layer['filter'] = create_variable(
                    'filter',
                    [initial_filter_width,  # initial_filter_width=10
                     initial_channels,      # MFSC_channels=60
                     self.residual_channels])   # residual_channels=130
                var['causal_layer'] = layer

            var['dilated_stack'] = list()     # var["dilated_stack"] is a list of dictionary, has 5 elements
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):  # dilations has 5 elements
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,  # filter_width=2
                             self.residual_channels,   # residual_channels=32
                             self.dilation_channels])   # dilation_channels=32
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])     # skip_channels=512


                        if self.global_condition_channels is not None:
                            current['gc_gateweights'] = create_variable(
                                'gc_gate',
                                [1, self.global_condition_channels,   # 
                                 self.dilation_channels])
                            current['gc_filtweights'] = create_variable(
                                'gc_filter',
                                [1, self.global_condition_channels,
                                 self.dilation_channels])

                        ####### Added for lc control #######
                        current["lc_gateweights"] = create_variable(
                                "lc_gate",
                                [1, self.lc_channels, self.dilation_channels])
                        current["lc_filtweights"] = create_variable(
                                "lc_filt",
                                [1, self.lc_channels, self.dilation_channels])

                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)
            ######### Added the control_skip block
            with tf.variable_scope("control_skip"):
                current = dict()
                current["lc_skipweights"] = create_variable(
                            "lc_skip",
                            [1, self.lc_channels, self.skip_channels])
               

                if self.use_biases:
                    current["lc_skipbias"] = create_variable(
                            "lc_skip",
                            [self.skip_channels])

                var["control_skip"] = current

            with tf.variable_scope('postprocessing'):
                current = dict()

                current["CMG_paras"] = create_variable(
                    "CMG_paras",
                    [1, self.skip_channels, self.CMG_channels])    
                if self.use_biases:
                    current["CMG_paras_bias"] = create_variable(
                    "CMG_paras_bias",
                    [self.CMG_channels])
                var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']   #weights_filter shape: [initial_filter_width,  # filter_width=10
                                                                        #                       initial_channels,      # MFSC_channels=60
                                                                        #                       self.residual_channels])   # residual_channels=130
                                                                        
            
            return causal_conv(input_batch, weights_filter, 1)         # dilation = 1 

    def _create_dilation_layer(self, input_batch, lc_batch, layer_index, dilation,
                               global_condition_batch, output_width):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             global_conditioning_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        variables = self.variables['dilated_stack'][layer_index]    # variables['dilated_stack'][i] is a dictionary

        weights_filter = variables['filter']   #[filter_width, residual_channels, dilation_channels]
        weights_gate = variables['gate']
        
        conv_filter = causal_conv(input_batch, weights_filter, dilation)    # [batch_size, m, dilation_channels]       
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        lc_filter = variables["lc_filtweights"]
        lc_gate = variables["lc_gateweights"]
    

        lc_batch_cut = tf.shape(lc_batch)[1] - tf.shape(conv_filter)[1]
        lc_batch = tf.slice(lc_batch, [0,lc_batch_cut,0], [-1, -1, -1])
        

        ######## convolve control inputs filter with control inputs ##########
        conv_filter += tf.nn.conv1d(lc_batch, lc_filter, stride=1, padding="SAME", name="lc_filt") 
                        
        ######## convolve control inputs gate with control inputs ##########
        conv_gate += tf.nn.conv1d(lc_batch, lc_gate, stride=1, padding="SAME", name="lc_gate") 
                        
        # global_condition_batch shape: [batch_size(1), 1, gc_channels]
        # input_batch shape: [batch_size(1), m, residual_channels]

        if global_condition_batch is not None:
            weights_gc_filter = variables['gc_filtweights']                         # TODO understand the broadcasting through time
            conv_filter = conv_filter + tf.nn.conv1d(global_condition_batch,    
                                                     weights_gc_filter,   # [batch_size(1), 1, gc_channels] * [1, self.global_condition_channels, self.dilation_channels]
                                                     stride=1,            # = [batch_size, 1, dilation_channels]. the '+' broadcast this through time
                                                     padding="SAME",
                                                     name="gc_filter") 
            weights_gc_gate = variables['gc_gateweights']
            conv_gate = conv_gate + tf.nn.conv1d(global_condition_batch,
                                                 weights_gc_gate,
                                                 stride=1,
                                                 padding="SAME",
                                                 name="gc_gate")

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])             # cut off the proceeding chunk, so that out_skip.shape[1] = output_width
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        if self.histograms:
            layer = 'layer{}'.format(layer_index)
            tf.histogram_summary(layer + '_filter', weights_filter)
            tf.histogram_summary(layer + '_gate', weights_gate)
            tf.histogram_summary(layer + '_dense', weights_dense)
            tf.histogram_summary(layer + '_skip', weights_skip)
            if self.use_biases:
                tf.histogram_summary(layer + '_biases_filter', filter_bias)
                tf.histogram_summary(layer + '_biases_gate', gate_bias)
                tf.histogram_summary(layer + '_biases_dense', dense_bias)
                tf.histogram_summary(layer + '_biases_skip', skip_bias)

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]        
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed

    def _generator_conv(self, input_batch, state_batch, weights):
        '''Perform convolution for a single convolutional processing step.'''
        # TODO generalize to filter_width > 2
        past_weights = weights[0, :, :]                 # filter width is 2 here, 0 means the first coefficient, [quantization_channels, residual_channels]
        curr_weights = weights[1, :, :]                 # 1 means the second coefficient
        output = tf.matmul(state_batch, past_weights) + tf.matmul(
            input_batch, curr_weights)
        return output

    def _generator_causal_layer(self, input_batch, state_batch):
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            output = self._generator_conv(
                input_batch, state_batch, weights_filter)
        return output

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
                                  dilation, global_condition_batch):
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        output_filter = self._generator_conv(
            input_batch, state_batch, weights_filter)
        output_gate = self._generator_conv(
            input_batch, state_batch, weights_gate)

        if global_condition_batch is not None:
            global_condition_batch = tf.reshape(global_condition_batch,
                                                shape=(1, -1))
            weights_gc_filter = variables['gc_filtweights']
            weights_gc_filter = weights_gc_filter[0, :, :]   # shape [self.global_condition_channels, self.dilation_channels]
            output_filter += tf.matmul(global_condition_batch,  # [1, gc_channels]
                                       weights_gc_filter)
            weights_gc_gate = variables['gc_gateweights']
            weights_gc_gate = weights_gc_gate[0, :, :]       # shape [self.global_condition_channels, self.dilation_channels]
            output_gate += tf.matmul(global_condition_batch,
                                     weights_gc_gate)

        if self.use_biases:
            output_filter = output_filter + variables['filter_bias']
            output_gate = output_gate + variables['gate_bias']

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = variables['dense']
        transformed = tf.matmul(out, weights_dense[0, :, :])
        if self.use_biases:
            transformed = transformed + variables['dense_bias']

        weights_skip = variables['skip']
        skip_contribution = tf.matmul(out, weights_skip[0, :, :])
        if self.use_biases:
            skip_contribution = skip_contribution + variables['skip_bias']

        return skip_contribution, input_batch + transformed

    def _create_network(self, input_batch, 
                        lc_batch,
                        global_condition_batch):
        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch  # shape = 51

        # Pre-process the input with a regular convolution
        # input_batch = [batch_size(1), receptive_field + sample_size, quantization_channels]
        # converted to current_layer = [batch_size, receptive_field + sample_size - 1, residual_channels]
        # this is the "input_batch" argument for all following functions
        current_layer = self._create_causal_layer(current_layer)  # shape=42 
        # current_layer1 = current_layer         

        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1   ########################################################

        lc_batch_cut = tf.shape(lc_batch)[1] - tf.shape(current_layer)[1]
        lc_batch_dilated = tf.slice(lc_batch, [0,lc_batch_cut,0], [-1, -1, -1])
      

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    # global_condition_batch shape: [batch_size(1), 1, gc_channels]
                    # input_batch shape: [batch_size(1), receptive_field + sample_size, quantization_channels]
                    output, current_layer = self._create_dilation_layer(      
                        current_layer, lc_batch_dilated, layer_index, dilation,
                        global_condition_batch, output_width)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            CMG_weights = self.variables['postprocessing']['CMG_paras']

            if self.use_biases:
                CMG_bias = self.variables['postprocessing']['CMG_paras_bias']

            if self.histograms:
                tf.histogram_summary('postprocess1_weights', w1)
                tf.histogram_summary('postprocess2_weights', w2)
                if self.use_biases:
                    tf.histogram_summary('postprocess1_biases', b1)
                    tf.histogram_summary('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)

            variables = self.variables["control_skip"]
            lc_skipweights = variables["lc_skipweights"]
            
            output_cut = tf.shape(lc_batch)[1] - output_width
            lc_batch_skip = tf.slice(lc_batch, [0,output_cut,0], [-1, -1, -1])
            

            ####### Add the control inputs skip connections
            total += tf.nn.conv1d(lc_batch_skip, lc_skipweights, stride=1, padding="SAME") 
  

            transformed2 = tf.nn.tanh(total)

            CMG = tf.nn.conv1d(transformed2, CMG_weights, stride=1, padding="SAME")
            if self.use_biases:
                CMG = tf.add(CMG, CMG_bias)

        return CMG

    def _create_generator(self, input_batch, global_condition_batch):
        '''Construct an efficient incremental generator.'''
        init_ops = []
        push_ops = []
        outputs = []
        current_layer = input_batch

        q = tf.FIFOQueue(
            1,
            dtypes=tf.float32,
            shapes=(self.batch_size, self.quantization_channels))
        init = q.enqueue_many(
            tf.zeros((1, self.batch_size, self.quantization_channels)))

        current_state = q.dequeue()
        push = q.enqueue([current_layer])
        init_ops.append(init)
        push_ops.append(push)

        current_layer = self._generator_causal_layer(
                            current_layer, current_state)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):

                    q = tf.FIFOQueue(
                        dilation,
                        dtypes=tf.float32,
                        shapes=(self.batch_size, self.residual_channels))
                    init = q.enqueue_many(
                        tf.zeros((dilation, self.batch_size,
                                  self.residual_channels)))

                    current_state = q.dequeue()
                    push = q.enqueue([current_layer])
                    init_ops.append(init)
                    push_ops.append(push)

                    output, current_layer = self._generator_dilation_layer(      ###### output is the skip contrinution, current_layer is the residual sum
                        current_layer, current_state, layer_index, dilation,
                        global_condition_batch)
                    outputs.append(output)
        self.init_ops = init_ops
        self.push_ops = push_ops

        with tf.name_scope('postprocessing'):
            variables = self.variables['postprocessing']
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = variables['postprocess1']
            w2 = variables['postprocess2']
            if self.use_biases:
                b1 = variables['postprocess1_bias']
                b2 = variables['postprocess2_bias']

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)

            conv1 = tf.matmul(transformed1, w1[0, :, :])
            if self.use_biases:
                conv1 = conv1 + b1
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.matmul(transformed2, w2[0, :, :])
            if self.use_biases:
                conv2 = conv2 + b2

        return conv2

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.

        convert shape [receptive_field + sample_size, 1] to shape [receptive_field + sample_size, quantization_channels]
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.quantization_channels,                         
                dtype=tf.float32)
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)
        return encoded                                                      # encoded audio shape: [batch_size(1), receptive_field + sample_size, quantization_channels]

    def _embed_gc(self, global_condition):
        '''Returns embedding for global condition.
        :param global_condition: Either ID of global condition for
               tf.nn.embedding_lookup or actual embedding. The latter is
               experimental.
        :return: Embedding or None
        '''
        embedding = None
        if self.global_condition_cardinality is not None:
            # Only lookup the embedding if the global condition is presented
            # as an integer of mutually-exclusive categories ...
            embedding_table = self.variables['embeddings']['gc_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               global_condition)
        elif global_condition is not None:
            # ... else the global_condition (if any) is already provided
            # as an embedding.

            # In this case, the number of global_embedding channels must be
            # equal to the the last dimension of the global_condition tensor.
            gc_batch_rank = len(global_condition.get_shape())
            dims_match = (global_condition.get_shape()[gc_batch_rank - 1] ==
                          self.global_condition_channels)
            if not dims_match:
                raise ValueError('Shape of global_condition {} does not'
                                 ' match global_condition_channels {}.'.
                                 format(global_condition.get_shape(),
                                        self.global_condition_channels))
            embedding = global_condition

        if embedding is not None:
            embedding = tf.reshape(
                embedding,
                [self.batch_size, 1, self.global_condition_channels])    # embedding shape: [batch_size(1), 1, gc_channels]

        return embedding

    def predict_proba(self, waveform, mfsc_channels, lc_batch, global_condition=None, name='wavenet'):
        '''Computes the probability distribution of the next sample based on
        all samples in the input waveform.
        If you want to generate audio by feeding the output of the network back
        as an input, see predict_proba_incremental for a faster alternative.'''
        with tf.name_scope(name):
            if self.scalar_input:
                encoded = tf.cast(waveform, tf.float32)
                encoded = tf.reshape(encoded, [-1, 1])
            else:
                encoded = waveform

            gc_embedding = self._embed_gc(global_condition)
            raw_output = self._create_network(encoded, lc_batch, gc_embedding)
            out = tf.reshape(raw_output, [-1, self.CMG_channels])
            lc_batch  = tf.reshape(lc_batch, [-1, self.lc_channels])
            # temperature control coef
            tau = np.concatenate((np.array([0.02] * 3),
                                    np.linspace(0.02, 0.2, 6),
                                    np.array([0.2] * 51)),
                                    axis = 0)
            tau = tf.constant(tau, dtype=tf.float32)

            mu1, mu2, mu3, mu4, sigma1, sigma2, sigma3, sigma4, w1, w2, w3, w4 = get_mixture_coef(out)
            mu1_hat, mu2_hat, mu3_hat, mu4_hat, sigma1_hat, sigma2_hat, sigma3_hat, sigma4_hat = temp_control(mu1, mu2, mu3, mu4, 
                                                                                                            sigma1, sigma2, sigma3, sigma4, 
                                                                                                            w1, w2,w3, w4, 
                                                                                                            tau)
            # take last row of mu's and sigma's as the distribution of the predicted frame
            mvn1 = tf.contrib.distributions.MultivariateNormalDiag(loc = mu1_hat[-1,:], scale_diag = sigma1_hat[-1,:])
            mvn2 = tf.contrib.distributions.MultivariateNormalDiag(loc = mu2_hat[-1,:], scale_diag = sigma2_hat[-1,:])
            mvn3 = tf.contrib.distributions.MultivariateNormalDiag(loc = mu3_hat[-1,:], scale_diag = sigma3_hat[-1,:])
            mvn4 = tf.contrib.distributions.MultivariateNormalDiag(loc = mu4_hat[-1,:], scale_diag = sigma4_hat[-1,:])
            
            # Adding estimate_output() content here to check if this speeds up generation
            w1 = w1[-1, :]
            w2 = w2[-1, :]
            w3 = w3[-1, :]
            w4 = w4[-1, :]
            
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
            
            return output

    def predict_proba_incremental(self, waveform, global_condition=None,
                                  name='wavenet'):
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''
        if self.filter_width > 2:
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")
        if self.scalar_input:
            raise NotImplementedError("Incremental generation does not "
                                      "support scalar input yet.")
        with tf.name_scope(name):
            encoded = tf.one_hot(waveform, self.quantization_channels)
            encoded = tf.reshape(encoded, [-1, self.quantization_channels])
            gc_embedding = self._embed_gc(global_condition)
            raw_output = self._create_generator(encoded, gc_embedding)
            out = tf.reshape(raw_output, [-1, self.quantization_channels])
            proba = tf.cast(
                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.quantization_channels])
            return tf.reshape(last, [-1])


    def loss(self,
             input_batch,
             lc_batch,
             global_condition_batch=None,
             l2_regularization_strength=None,
             name='wavenet_loss'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.name_scope(name):

            ###### add gaussian noise and then norm data
            input_batch_corrupted = add_gaussian_noise(input_batch)
            input_batch_norm = norm_data(input_batch_corrupted)

            gc_embedding = self._embed_gc(global_condition_batch)    # embedding shape: [batch_size(1), 1, gc_channels]
            if self.scalar_input:
                network_input = tf.reshape(
                    tf.cast(input_batch, tf.float32),
                    [self.batch_size, -1, 1])
            else:
                network_input = input_batch_norm

            # Cut off the last sample of network input to preserve causality.
            network_input_width = tf.shape(network_input)[1] - 1      
            network_input = tf.slice(network_input, [0, 0, 0],
                                     [-1, network_input_width, -1])
        
            
            raw_output = self._create_network(network_input, lc_batch, gc_embedding)
           
            with tf.name_scope('loss'):
                # Cut off the samples corresponding to the receptive field
                # for the first predicted sample.
                target_output = tf.slice(
                    tf.reshape(
                        input_batch,
                        [self.batch_size, -1, self.MFSC_channels]),    # target_output has dim 60
                    [0, self.receptive_field, 0],
                    [-1, -1, -1])
                
                target_output = tf.reshape(target_output,
                                           [-1, self.MFSC_channels])
                prediction = tf.reshape(raw_output,
                                        [-1, self.CMG_channels])   # raw_output has dim 240

                # temperature control coef
                tau = np.concatenate((np.array([0.05] * 3),
                                        np.linspace(0.05, 0.5, 6),
                                        np.array([0.5] * 51)),
                                        axis = 0)
                tau = tf.constant(tau, dtype=tf.float32)

                mu1, mu2, mu3, mu4, sigma1, sigma2, sigma3, sigma4, w1, w2, w3, w4 = get_mixture_coef(prediction)
                mu1_hat, mu2_hat, mu3_hat, mu4_hat, sigma1_hat, sigma2_hat, sigma3_hat, sigma4_hat = temp_control(mu1, mu2, mu3, mu4, 
                                                                                                                sigma1, sigma2, sigma3, sigma4, 
                                                                                                                w1, w2,w3, w4, 
                                                                                                                tau)
                loss = nll_loss(mu1_hat, mu2_hat, mu3_hat, mu4_hat, 
                                    sigma1_hat, sigma2_hat, sigma3_hat, sigma4_hat, 
                                    w1, w2, w3, w4, 
                                    target_output)


                tf.summary.scalar('loss', loss)

                return loss
