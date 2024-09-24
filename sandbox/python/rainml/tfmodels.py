import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers, activations, regularizers
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
