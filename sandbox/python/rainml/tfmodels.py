import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers, activations, regularizers
#Sequential, 
#from keras import backend as KerasBackend
#from keras.saving import register_keras_serializable, serialize_keras_object, deserialize_keras_object
from tensorflow.keras.utils import register_keras_serializable, serialize_keras_object, deserialize_keras_object
from tensorflow.keras.initializers import Initializer

def model_basic(N, K, fcshape, input_sequence_length=None, input_batch_size=None):
    """Rain Drop Counter basic convolutional model (paper V3)

    :param N: detector filter length
    :param K: number of detector filters
    :param fcshape: fully connected layer size(s) (integer or tuple) excluded the last softmax layer.
    :param input_sequence_length: Optional input sequence length. Should be set to >=K when generating TFLite models.
    :param input_batch_size: Optional input batch size parameter. Optional for TFLite model generation, but should be set to the desired number of batches when generating TFLite models. Defaults to 1.
    :returns: tf.keras.Model
    """
    assert type(fcshape) == tuple, "resolver shape not a tuple (of integers)"
    
    inputs = layers.Input(shape=(input_sequence_length, 1), batch_size=input_batch_size, name='input')
    bottleneck = layers.GlobalMaxPooling1D(keepdims=True, name='bottleneck')
    flatten = layers.Flatten(name='flatten')
    dect = layers.Conv1D(
        filters=N, kernel_size=K,
        # kernel_initializer=random_diff_kernel,
        kernel_initializer=StepKernel(),
        name='step_detector',
        use_bias=True, strides=1, padding='same',
        activation=activations.relu
    )

    mask = sum([t > 1 for t in fcshape])
    shp = fcshape[:mask]
    fc = FCResolver(
        shp,
        use_bias=True,
        activation=activations.relu
    )
    x = dect(inputs)
    x = bottleneck(x)
    x = flatten(x)
    x = fc(x)
    m = Model(inputs=inputs, outputs=x)
    _ = m(np.random.random_sample((1, K, 1)).astype('float'))
    return m


def model_residual(dspshape, rdsp, K, M, fcshape, input_sequence_length=None, input_batch_size=None):
    """Rain drop counter model with Residual denoiser (V3). All convolutional layers have the same length K.
    It is modelled by a ResNet residual block. Only one such block is used with a customizable number and shapes of DSP layers.
    
    :param dspshape: (tuple) Filter numbers for each DSP layers.
    :param rdsp: weight regularization parameter. Applied to all DSP layers.
    :param K: length of the convolutional filters.
    :param M: multiplier for the detector depthwise convolutional layer.
    :param fcshape: a FCResolver initialization parameter, a tuple of fully connected layer widths
    :param input_sequence_length: Optional input sequence length. Should be set to >=K when generating TFLite models.
    :param input_batch_size: Optional input batch size parameter. Optional for TFLite model generation, but should be set to the desired number of batches when generating TFLite models. Defaults to 1.
    :returns: keras.Model

    """
    relu = layers.ReLU(name='relu')
    add = layers.Add(name='add')
    flatten = layers.Flatten(name='flatten')
    bottleneck = layers.GlobalMaxPooling1D(keepdims=True, name='bottleneck')
    inputs = layers.Input(shape=(input_sequence_length, 1), batch_size=input_batch_size, name='input')
    bn = [layers.BatchNormalization(name=f'bn{i:02d}') for i in range(len(dspshape))]
    dsp = [
        layers.Conv1D(
            name=f'dsp{i:02d}',
            filters=dspshape[i],
            kernel_size=K,
            padding='same',
            use_bias=True,
            activation=None,
            kernel_regularizer=None if rdsp is None else regularizers.L2(rdsp),
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        ) for i in range(len(dspshape))
    ]

    dect = layers.DepthwiseConv1D(
        K,
        name='step_detector',
        strides=1, padding='same',
        depth_multiplier=M,
        activation=activations.relu,
        use_bias=True,
        # depthwise_initializer=random_diff_kernel,
        depthwise_initializer=StepKernel(),
        bias_initializer=None,
        # kernel_regularizer=None # regularizers.L1(rdsp),  -removed from v2.16? 
        depthwise_regularizer=None
    )

    fc = FCResolver(fcshape, use_bias=True, activation=activations.relu)

    x = relu(bn[0](dsp[0](inputs)))
    for i in range(1, len(dspshape)):
        x = relu(bn[i](dsp[i](x)))
    # Add the input (1 channel) to each channel of x. 
    # Leverage the default Keras action when one of the inputs has exactly 1 channel.
    x = add([x, inputs])
    x = dect(x)
    x = bottleneck(x)
    x = flatten(x)
    x = fc(x)
    
    m = Model(inputs=inputs, outputs=x)
    _ = m(np.random.random_sample((1, K, 1)).astype('float'))
    return m


def model_densnet(dspshape, rdsp, K, M, fcshape, input_sequence_length=None, input_batch_size=None):
    """Rain drop counter model with DensNet-like denoiser.

    :param dspshape: (tuple) Filter numbers for each DSP layers.
    :param rdsp: weight regularization parameter. Applied to all DSP layers.
    :param K: length of the convolutional filters.
    :param M: multiplier for the detector depthwise convolutional layer.
    :param fcshape: a FCResolver initialization parameter, a tuple of fully connected layer widths
    :param input_sequence_length: Optional input sequence length. Should be set to >=K when generating TFLite models.
    :param input_batch_size: Optional input batch size parameter. Optional for TFLite model generation, but should be set to the desired number of batches when generating TFLite models. Defaults to 1.
    :returns: tf.keras.Model

    """
    
    bn1 = layers.BatchNormalization(name='bn1')
    bn2 = layers.BatchNormalization(name='bn2')
    relu = layers.ReLU(name='relu')
    relu1 = layers.ReLU(name='relu1')
    relu2 = layers.ReLU(name='relu2')
    add = layers.Add(name='add')
    flatten = layers.Flatten(name='flatten')
    bottleneck = layers.GlobalMaxPooling1D(keepdims=True, name='bottleneck')
    stack = layers.Concatenate(axis=-1, name='stack')
    
    inputs = layers.Input(shape=(input_sequence_length, 1), batch_size=input_batch_size, name='input')
    dsp1 = layers.Conv1D(
        name='dsp1',
        filters=dspshape[0],
        kernel_size=K,
        padding='same',
        use_bias=True,
        activation=None,
        kernel_regularizer=None if rdsp is None else regularizers.L2(rdsp),
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    )

    dsp2 = layers.Conv1D(
        name='dsp2',
        filters=dspshape[1],
        kernel_size=K,
        padding='same',
        use_bias=True,
        activation=None,
        kernel_regularizer=None if rdsp is None else regularizers.L2(rdsp),
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    )

    dect = layers.DepthwiseConv1D(
        K,
        name='step_detector',
        strides=1, padding='same',
        depth_multiplier=M,
        activation=activations.relu,
        use_bias=True,
        # depthwise_initializer=random_diff_kernel,
        depthwise_initializer=StepKernel(),
        bias_initializer=None,
        # kernel_regularizer=None - removed from v2.16?
        depthwise_regularizer=None
    )

    fc = FCResolver(fcshape, use_bias=True, activation=activations.relu)

    x1 = relu(bn1(dsp1(inputs)))
    x2 = relu(bn2(dsp2(stack([x1, inputs]))))
    x = dect(stack([x2, x1, inputs]))
    x = bottleneck(x)
    x = flatten(x)
    x = fc(x)

    m = Model(inputs=inputs, outputs=x)
    _ = m(np.random.random_sample((1, K, 1)).astype('float'))
    return m


def model_RDN(dspshape, rdsp, K, M, fcshape, input_sequence_length=None, input_batch_size=None):
    """Rain drop counter model with Residual DenseNet denoiser (V3).

    :param dspshape: (tuple) Filter numbers for each DSP layers.
    :param rdsp: weight regularization parameter. Applied to all DSP layers.
    :param K: length of the convolutional filters.
    :param M: multiplier for the detector depthwise convolutional layer.
    :param fcshape: a FCResolver initialization parameter, a tuple of fully connected layer widths
    :param input_sequence_length: Optional input sequence length. Should be set to >=K when generating TFLite models.
    :param input_batch_size: Optional input batch size parameter. Optional for TFLite model generation, but should be set to the desired number of batches when generating TFLite models. Defaults to 1.
    :returns: tf.keras.Model

    """
    add = layers.Add(name='add')
    relu = layers.ReLU(name='relu')
    stack = layers.Concatenate(axis=-1, name='stack')
    inputs = layers.Input(shape=(input_sequence_length, 1), batch_size=input_batch_size, name='input')
    bn1 = layers.BatchNormalization(name='bn1')
    bn2 = layers.BatchNormalization(name='bn2')
    bn3 = layers.BatchNormalization(name='bn3')
    bn4 = layers.BatchNormalization(name='bn4')
    flatten = layers.Flatten(name='flatten')
    bottleneck = layers.GlobalMaxPooling1D(keepdims=True, name='bottleneck')
    dsp1 = layers.Conv1D(
        name='dsp1',
        filters=dspshape[0],
        kernel_size=K,
        padding='same',
        use_bias=True,
        activation=None,
        kernel_regularizer=None if rdsp is None else regularizers.L2(rdsp),
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    )

    dsp2 = layers.Conv1D(
        name='dsp2',
        filters=dspshape[1],
        kernel_size=K,
        padding='same',
        use_bias=True,
        activation=None,
        kernel_regularizer=None if rdsp is None else regularizers.L2(rdsp),
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    )

    dsp3 = layers.Conv1D(
        name='dsp3',
        filters=dspshape[2],
        kernel_size=K,
        padding='same',
        use_bias=True,
        activation=None,
        kernel_regularizer=None if rdsp is None else regularizers.L2(rdsp),
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    )

    dsp4 = layers.Conv1D(
        name='dsp4',
        filters=dspshape[3],
        kernel_size=K,
        padding='same',
        use_bias=True,
        activation=None,
        kernel_regularizer=None if rdsp is None else regularizers.L2(rdsp),
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    )

    dect = layers.DepthwiseConv1D(
        K,
        name='step_detector',
        strides=1, padding='same',
        depth_multiplier=M,
        activation=activations.relu,
        use_bias=True,
        depthwise_initializer=StepKernel(),
        bias_initializer=None,
        # kernel_regularizer=None
        depthwise_regularizer=None
    )

    fc = FCResolver(fcshape, use_bias=True, activation=activations.relu)

    x1 = relu(bn1(dsp1(inputs)))
    x2 = relu(bn2(dsp2(stack([x1, inputs]))))
    x3 = relu(bn3(dsp3(stack([x2, x1, inputs]))))
    x4 = relu(bn4(dsp4(stack([x3, x2, x1, inputs]))))
    x = dect(stack([x4, x3, x2, x1]))
    x = add([x, inputs])
    x = bottleneck(x)
    x = flatten(x)
    x = fc(x)

    m = Model(inputs=inputs, outputs=x)
    _ = m(np.random.random_sample((1, K, 1)).astype('float'))
    return m




@register_keras_serializable(package="rainml", name="random_diff_kernel")
def random_diff_kernel(shape, dtype=None):
    """Initializer for the naive detectors that can be used with Conv1D and DepthwiseConv1D layers.

    Make N random kernels difference kernels with K_pos positive side, and K_neg, negative side.
    The kernels are reshaped so that they are of the correct shape for the keras layer objects.
    Time-stamp: <2023-09-11 09:03:35 rytis>

    """
    K = shape[0]
    N = shape[1]*shape[2]
    _dtype = dtype
    if isinstance(dtype, tf.dtypes.DType):
        _dtype = dtype.as_numpy_dtype

    assert N > 0 and K > 1
    # This is our heuristic for choosing K_neg and K_pos
    # It will generate random K_neg. Examples (K=min(K_neg)-max(K_neg)) :
    # K=6:2-3 K=12:3-5 K=24:5-9 K=30:7-11
    K_neg = np.random.randint(1 + (K//5), 2 + (K//3))
    # An alternative, non random K_neg:
    # K_neg = 1 + (K // 4)
    K_pos = K - K_neg
    t_neg = np.random.rand(N, K_neg).astype(_dtype)
    t_pos = np.random.rand(N, K_pos).astype(_dtype)
    tmp = np.hstack(
        (t_neg/(-np.sum(t_neg, 1, keepdims=True)),
         t_pos/np.sum(t_pos, 1, keepdims=True)))
    tmp = np.reshape(tmp, (shape[1], shape[2], K))
    return np.transpose(tmp, (2, 0, 1))



@register_keras_serializable(package="rainml", name="FCResolver")
class FCResolver(layers.Layer):
    """A simple Resolver consisting of one or more Dense layers. The last layer is a softmax classifier.

    Optional arguments will be passed to all layers (all are dense). 
    Frequency detector in a sense that we use only
    one dense layer.
    The dense layer use the same number of neurons as the number of
    filters.
    It is followed by the classifier layer.

    Time-stamp:  <2023-10-15 11:01:17 rytis>

    neurons: specify the size of the dense layer. Default neurons=filters
    """
    def __init__(self, units, **kwargs):
        super(FCResolver, self).__init__()
        self.units = []
        assert (units is None) or (type(units) in [int, list, tuple]), "wrong units parameter"
        # We can override some parameters, like 'name'
        # self.__name__ = 'fc'
        # if 'name' in kwargs.keys():
        #     self.__name__ = kwargs.pop('name')
        for t in ['name']:
            if t in kwargs.keys():
                print("[FCResolver] warning: overriding the option", t)
                kwargs.pop(t)

        if type(units) == int:
            units = [units]

        if units is not None:
            self.units = units
            self.dense = [
                layers.Dense(
                    units[i],
                    name=f'fc_{i+1:02}',
                    **kwargs
                    # use_bias=True,
                    # activation=activations.relu
                ) for i in range(len(units))
            ]
        else:
            self.dense = None

        # self.classifier = layers.Dense(1, name='fc_softmax', **kwargs)
        # Above is a big problem! Softmax should not use ReLU!
        if 'activation' in kwargs.keys():
            kwargs.pop('activation')
        if 'use_bias' in kwargs.keys():
            kwargs.pop('use_bias')
        self.classifier = layers.Dense(1, name='fc_softmax', use_bias=True, activation=None, **kwargs)

    def call(self, inputs):
        x = inputs
        if self.dense:
            for i in range(len(self.dense)):
                x = self.dense[i](x)
        return self.classifier(x)

    def get_config(self):
        # base_config = super(FCResolver, self).get_config()
        # config = {'units': serialize_keras_object(list(self.units))}
        # return {**base_config, **config}
        config = super(FCResolver, self).get_config()
        config.update({'units': list(self.units)})
        return config

    @classmethod
    def from_config(cls, config):
        units_config = config.pop('units')
        units = deserialize_keras_object(units_config)
        return cls(units, **config)





@register_keras_serializable(package="rainml", name="StepKernel")
class StepKernel(Initializer):
    
    def __init__(self):
        super(Initializer, self).__init__()
        
    def __call__(self, shape, dtype=None):
        """Initializer for the naive detectors that can be used with Conv1D and DepthwiseConv1D layers.

        Make N random kernels difference kernels with K_pos positive side, and K_neg, negative side.
        The kernels are reshaped so that they are of the correct shape for the keras layer objects.
        Time-stamp: <2023-09-11 09:03:35 rytis>
        
        """
        K = shape[0]
        N = shape[1]*shape[2]
        _dtype = dtype
        if isinstance(dtype, tf.dtypes.DType):
            _dtype = dtype.as_numpy_dtype

        assert N > 0 and K > 1
        # This is our heuristic for choosing K_neg and K_pos
        # It will generate random K_neg. Examples (K=min(K_neg)-max(K_neg)) :
        # K=6:2-3 K=12:3-5 K=24:5-9 K=30:7-11
        K_neg = np.random.randint(1 + (K//5), 2 + (K//3))
        # An alternative, non random K_neg:
        # K_neg = 1 + (K // 4)
        K_pos = K - K_neg
        t_neg = np.random.rand(N, K_neg).astype(_dtype)
        t_pos = np.random.rand(N, K_pos).astype(_dtype)
        tmp = np.hstack(
            (t_neg/(-np.sum(t_neg, 1, keepdims=True)),
             t_pos/np.sum(t_pos, 1, keepdims=True)))
        tmp = np.reshape(tmp, (shape[1], shape[2], K))
        return np.transpose(tmp, (2, 0, 1))


# def model_residual(dspshape, rdsp, K, M, fcshape):
#     """TODO describe function

#     :param dspshape: a pair of filter numbers to use in two convolutional DSP layers
#     :param rdsp: regularization parameter for the DSP block
#     :param K: length of the filters (all)
#     :param M: multiplier for the depthwise convolutional layer
#     :param fcshape: a FCResolver initialization parameter, which is a tuple of fully connected layer widths
#     :returns: keras.Model

#     """
    
#     bn1 = layers.BatchNormalization(name='bn1')
#     bn2 = layers.BatchNormalization(name='bn2')
#     relu1 = layers.ReLU(name='relu1')
#     relu2 = layers.ReLU(name='relu2')
#     add = layers.Add(name='add')
#     flatten = layers.Flatten(name='flatten')
#     bottleneck = layers.GlobalMaxPooling1D(keepdims=True, name='bottleneck')
    
#     inputs = layers.Input(shape=(None, 1), name='input')
#     dsp1 = layers.Conv1D(
#         name='dsp1',
#         filters=dspshape[0],
#         kernel_size=K,
#         padding='same',
#         use_bias=True,
#         activation=None,
#         kernel_regularizer=None if rdsp is None else regularizers.L2(rdsp),
#         kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
#     )

#     dsp2 = layers.Conv1D(
#         name='dsp2',
#         filters=dspshape[1],
#         kernel_size=K,
#         padding='same',
#         use_bias=True,
#         activation=None,
#         kernel_regularizer=None if rdsp is None else regularizers.L2(rdsp),
#         kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
#     )

#     dect = layers.DepthwiseConv1D(
#         K,
#         name='step_detector',
#         strides=1, padding='same',
#         depth_multiplier=M,
#         activation=activations.relu,
#         use_bias=True,
#         # depthwise_initializer=random_diff_kernel,
#         depthwise_initializer=StepKernel(),
#         bias_initializer=None,
#         kernel_regularizer=None # regularizers.L1(rdsp),
#     )

#     fc = FCResolver(fcshape, use_bias=True, activation=activations.relu)
    
#     x = relu1(bn1(dsp1(inputs)))
#     x = relu2(bn2(dsp2(x)))
#     # Add the input (1 channel) to each channel of x. 
#     # Leverage the default Keras action when one of the inputs has exactly 1 channel.
#     x = add([x, inputs])
#     x = dect(x)
#     x = bottleneck(x)
#     x = flatten(x)
#     x = fc(x)
    
#     m = Model(inputs=inputs, outputs=x)
#     _ = m(np.random.random_sample((1, K, 1)).astype('float'))
#     return m



# This is the ChatGPT suggestion of how to tf.data.Dataset.map ...
# Time-stamp: <2024-01-28 11:51:46 rytis>
# import tensorflow as tf
# # Example function to apply to the first channel
# def process_first_channel(data):
#     # Assuming 'data' is a tuple (channel1, channel2)
#     channel1, channel2 = data
#     # Apply some processing to channel1
#     processed_channel1 = some_transformation(channel1)
#     return processed_channel1, channel2
# # I think that some transformation could be inspired by this:
# const_to_log10 = - 1.0/tf.math.log(tf.constant(10, dtype='float'))
# inputs = layers.Input(shape=(None, 1))
# loglayer = layers.Lambda(lambda x:  const_to_log10*tf.math.log(x), output_shape=(None, 1))
# # Something like so (untested):
# processed_channel1 = const_to_log10*tf.math.log(channel1)



# # Assume 'dataset' is your initial tf.data.Dataset
# # Each element of 'dataset' is a two-channel sequence (channel1, channel2)
# dataset = # your dataset initialization here

# # Map the dataset using the defined function
# processed_dataset = dataset.map(process_first_channel)

# # Now 'processed_dataset' has your function applied to the first channel of each element



# class differentialFilters(layers.Layer):
#     """The naive detector, including the bottleneck.

#     This implementation groups the 'naive' detector, the GlobalMaxPooling1D and Flatten layers.
    
#     Developed in the context of RainDropCounterV2.
#     Time-stamp: <2023-10-15 10:47:04 rytis>
#     """
    
#     def __init__(self, filters, kernel_size, **kwargs):
#         """
#         filters - number of filters
#         kernel_size - filter size

#         Optional arguments will be passed to the Conv1D layer.
#         The options 'kernel_initializer' and 'name' are ignored. We use our own :)

#         The 'standard' use case could be like so:
#         use_bias=True, strides=1
#         padding='valid' (or 'same')
#         bias_initializer=None,
#         activation=activations.relu,
#         kernel_regularizer=None
#         """

#         super(differentialFilters, self).__init__()
#         assert (filters > 0) and (kernel_size > 1), "incorrect input parameters"
#         # Override some options:
#         for t in ['kernel_initializer', 'name']:
#             if t in kwargs.keys():
#                 print("[differentialFilters] warning: overriding the option", t)
#                 kwargs.pop(t)
#         # self._dtype = kwargs.pop('dtype', tf.as_dtype('float'))
#         self.detector = layers.Conv1D(
#             filters=filters, kernel_size=kernel_size,
#             kernel_initializer=random_diff_kernel,
#             name='step_detector',
#             **kwargs
#         )
#         self.bottleneck = layers.GlobalMaxPooling1D(
#             keepdims=True,
#             name='frequency_bottleneck'
#             # dtype=self._dtype
#         )
#         self.flatten = layers.Flatten(name='frequency_flattener', dtype=self._dtype)

#     def call(self, inputs):
#         x = self.detector(inputs)
#         x = self.bottleneck(x)
#         return self.flatten(x)



# class residualDSP(layers.Layer):
#     def __init__(self, conv_spec):
#         """conv_spec = [ (filters1, kernel_size1), (filters2, kernel_size2), ...]
#         """
#         assert (conv_spec is not None) and isinstance(conv_spec, list), "conv spec not a list"
#         assert len(conv_spec) > 0, "conv spec empty list"
#         for k in conv_spec:
#             assert isinstance(k, tuple), f"{k} not a tuple"
#             assert len(k) == 3, "spec not a pair"
#             assert isinstance(k[0], int) and isinstance(k[1], int), "not a pair of the form (filters, kernel_size)"
#         super(residualDSP, self).__init__()
#         self.conv = [
#             layers.SeparableConv1D(
#                 s[0], s[1], strides=1, padding='same', name=f'DSP_filter_{i}',
#                 activation=activations.relu,
#                 depth_multiplier=s[2],
#                 depthwise_initializer=initializers.RandomUniform(
#                     minval=-1.0/s[1], maxval=1.0/s[1], seed=None
#                 ),
#                 depthwise_regularizer=None,
#                 pointwise_initializer=initializers.RandomNormal(mean=0., stddev=1.),
#                 pointwise_regularizer=None,
#                 use_bias=True,
#                 bias_initializer=initializers.RandomNormal(mean=0., stddev=1.),
#                 bias_regularizer=None
#             ) for i, s in enumerate(conv_spec)
#         ]
#         self.join = layers.Conv1D(
#             1, 1, use_bias=True, strides=1, padding='same',
#             kernel_initializer=initializers.RandomNormal(mean=0., stddev=1.),
#             bias_initializer=initializers.RandomNormal(mean=0., stddev=1.),
#             activation=activations.relu,
#             kernel_regularizer=None,
#             name='DSP_join'
#         )
#         self.add = layers.Add()

#     def call(self, inputs):
#         x = self.conv[0](inputs)
#         for c in self.conv[1:]:
#             x = c(x)
#         x = self.join(x)
#         x = self.add([x, inputs])
#         return x




# def model_super( regu1, regu2):


#     flatten = layers.Flatten(name='flatten')
#     inputs = layers.Input(shape=(None, 1))

#     dnse = layers.Conv1D(
#         filters=N, kernel_size=K1,
#         padding='same',
#         use_bias=False,
#         kernel_regularizer=regularizers.L2(regu1),
#         kernel_initializer=None
#     )

#     difkern = layers.DepthwiseConv1D(
#         K2, strides=1, padding='valid', depth_multiplier=M, activation=activations.relu,
#         use_bias=True, depthwise_initializer=random_diff_kernel, bias_initializer=None,
#         kernel_regularizer=regularizers.L2(regu2),

#     )

#     pool = layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')

#     c_args = dict({'use_bias': True,
#                    'strides': 1,
#                    'padding': 'valid',
#                    'activation': activations.relu,
#                    'kernel_regularizer': regularizers.L2(1.e-3)
#                    })
#     p_args = dict({'pool_size': 2, 'strides': 2, 'padding': 'valid'})
#     c1 = layers.Conv1D(N, K, **c_args)
#     c2 = layers.Conv1D(N, K, **c_args)
#     # p2 = layers.MaxPooling1D(**p_args)
#     p2 = layers.GlobalMaxPooling1D(keepdims=True)
#     dense = DLogit(C)


#     x = dnse(inputs)
#     x = layers.Add()([x, inputs])
#     x = difkern(x)
#     x = pool(x)
#     # Add a VGG-like unit
#     x = c1(x)
#     x = c2(x)
#     x = flatten(x)
#     x = p2(x)
#     x = dense(x)
#     m = Model(inputs=inputs, outputs=x)
#     return m
    

    #     tf.keras.layers.DepthwiseConv1D(
    #     kernel_size,
    #     strides=1,
    #     padding='valid',
    #     depth_multiplier=1,
    #     data_format=None,
    #     dilation_rate=1,
    #     activation=None,
    #     use_bias=True,
    #     depthwise_initializer='glorot_uniform',
    #     bias_initializer='zeros',
    #     depthwise_regularizer=None,
    #     bias_regularizer=None,
    #     activity_regularizer=None,
    #     depthwise_constraint=None,
    #     bias_constraint=None,
    #     **kwargs
    # )


    

    

# def F1(y_true, y_pred):
#     def recall(y_true, y_pred):
#         """Recall metric.

#         Only computes a batch-wise average of recall.

#         Computes the recall, a metric for multi-label classification of
#         how many relevant items are selected.
#         """
#         true_positives = KerasBackend.sum(KerasBackend.round(KerasBackend.clip(y_true * y_pred, 0, 1)))
#         possible_positives = KerasBackend.sum(KerasBackend.round(KerasBackend.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + KerasBackend.epsilon())
#         return recall

#     def precision(y_true, y_pred):
#         """Precision metric.

#         Only computes a batch-wise average of precision.

#         Computes the precision, a metric for multi-label classification of
#         how many selected items are relevant.
#         """
#         true_positives = KerasBackend.sum(KerasBackend.round(KerasBackend.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = KerasBackend.sum(KerasBackend.round(KerasBackend.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + KerasBackend.epsilon())
#         return precision
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+KerasBackend.epsilon()))

# def FalseAlarm(y_true, y_pred):
#     """False alarm metric: TP/(TP + FN=N)
#         Only computes a batch-wise average of false alarm.
#     """
#     possible_negatives = KerasBackend.sum(KerasBackend.round(KerasBackend.clip(1-y_true, 0, 1)))
#     false_positives = KerasBackend.sum(KerasBackend.round(KerasBackend.clip((1-y_true) * y_pred, 0, 1)))
#     return false_positives/(possible_negatives+KerasBackend.epsilon())


# K=9
# N=5
# dsp = residualDSP([(3,3),(5,5)])
# dect = differentialFilters(N,K)
# fc = DLogit(N)

# input_shape = (None, 1)
# inputs = layers.Input(shape=input_shape)
# outputs = fc(dect(dsp(inputs)))
# m = Model(inputs=inputs, outputs=outputs)
# _ = m(np.random.random_sample((1, K+1, 1)).astype('float'))

# m.compile(
#     optimizer=optimizers.SGD(learning_rate=0.01),
#     loss=losses.BinaryCrossentropy(from_logits=True),
#     metrics=METRICS
# )



# # a failed model
# def model_VGGlike(N, K, fcshape, l2par=0.01):

#     # for t in ['stride'] in kwargs.keys():
#     #     if t in kwargs.keys():
#     #         print("[model_VGGlike] warning: ignoring option", t)
#     #         kwargs.pop(t)
#     scope = 2*(1+2*(K-1)) + 2*(K-1)
#     c_args = dict({'use_bias': True,
#                    'strides': 1,
#                    'padding': 'valid',
#                    'activation': None,
#                    'kernel_regularizer': regularizers.L2(l2par)
#                    })
#     p_args = dict({'pool_size': 2, 'strides': 2, 'padding': 'valid'})
#     d_args = dict({'use_bias': True,
#                    'activation': activations.relu,
#                    'kernel_regularizer': regularizers.L2(l2par)
#                    })
#     bottleneck = layers.GlobalMaxPooling1D(keepdims=True)
#     flatten = layers.Flatten(name='flatten')
#     relu1 = layers.ReLU()
#     relu2 = layers.ReLU()
#     relu3 = layers.ReLU()
#     relu4 = layers.ReLU()
#     bn1 = layers.BatchNormalization()
#     bn2 = layers.BatchNormalization()
#     bn3 = layers.BatchNormalization()
#     bn4 = layers.BatchNormalization()
#     pool = layers.MaxPooling1D(**p_args)
#     inputs = layers.Input(shape=(None, 1))
#     c1 = layers.Conv1D(filters=N, kernel_size=K, **c_args)
#     c2 = layers.Conv1D(filters=N, kernel_size=K, **c_args)
#     K2 = max(1, N//2)
#     c3 = layers.Conv1D(filters=K2, kernel_size=K, **c_args)
#     c4 = layers.Conv1D(filters=K2, kernel_size=K, **c_args)
#     fc = FCResolver(fcshape, **d_args)
    
#     x = relu1(bn1(c1(inputs)))
#     x = pool(relu2(bn2(c2(x))))
#     x = relu3(bn3(c3(x)))
#     x = bottleneck(relu4(bn4(c4(x))))

#     m = Model(inputs=inputs, outputs=fc(flatten(x)))
#     _ = m(np.random.random_sample((1, scope, 1)).astype('float'))
#     return m



# # marked for removal
# def model_dsp(Ndsp, Kdsp, rdsp, Mdect, Kdect):
#     bn = layers.BatchNormalization(name='bn')
#     relu = layers.ReLU(name='relu')
#     add = layers.Add(name='add')
#     flatten = layers.Flatten(name='flatten')
#     bottleneck = layers.GlobalMaxPooling1D(keepdims=True, name='bottleneck')
    
#     inputs = layers.Input(shape=(None, 1))
#     dsp = layers.Conv1D(
#         filters=Ndsp,
#         kernel_size=Kdsp,
#         name='dsp',
#         padding='same',
#         use_bias=True,
#         activation=None,
#         kernel_regularizer=regularizers.L2(rdsp),
#         # bias_regularizer=None, #regularizers.L2(rdsp),
#         # kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None),
#         kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
#     )

#     dect = layers.DepthwiseConv1D(
#         Kdect,
#         name='step_detector',
#         strides=1,
#         padding='same',
#         depth_multiplier=Mdect,
#         activation=activations.relu,
#         use_bias=True,
#         depthwise_initializer=random_diff_kernel,
#         bias_initializer=None,
#         kernel_regularizer=None # regularizers.L1(rdsp),
#     )

#     fc = FCResolver(Ndsp*Mdect, use_bias=True, activation=activations.relu)
    
#     x = relu(bn(dsp(inputs)))
#     x = add([x, inputs])
#     x = dect(x)
#     x = bottleneck(x)
#     x = flatten(x)
#     x = fc(x)

#     m = Model(inputs=inputs, outputs=x)
#     _ = m(np.random.random_sample((1, max(Kdect, Kdsp), 1)).astype('float'))
#     return m

