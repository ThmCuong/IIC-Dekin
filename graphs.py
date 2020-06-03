# inclue some graph, neural net using in IIC
# VGG A, B, C
import tensorflow as tf 

KERNEL_INIT = tf.keras.initializers.he_uniform()
WEIGHT_INIT = tf.random_normal_initializer(mean=0.0, stddev=0.01) 
BIAS_INIT = tf.constant_initializer(0.0) 
# convolution block with option batch_norm
def convolution_layer(x, kernel_size, num_out_channels, activation,batch_norm, is_training,name):
    x = tf.keras.layers.Conv2D(
                                filters = num_out_channels, 
                                kernel_size = [kernel_size]*2,
                                strides = [1, 1],
                                padding = 'same',
                                activation = None,
                                kernel_initializer = BIAS_INIT,
                                name =name           
                                )(x)
    if(batch_norm):
        x = tf.keras.layers.BatchNormalization()(x)
    x = activation(x)
    return x
                                                

# maxpooling
def max_pooling_layer(x, pool_size, strides, name):
    x = tf.keras.layers.MaxPool2D(pool_size = pool_size, strides = strides,name =name)(x)
    return x 

# full connected
def fully_connected_layer(x, num_outputs, activation, is_training, name):
    x = tf.keras.layers.Dense(units = num_outputs, activation = None,name = name)
    x = tf.keras.layers.BatchNormalization()(x)
    x = activation(x)
    return x
# IIC graph
class IICGraph(object):
    def __init__(self, config = 'B', batch_norm =True,fan_out_init = 64):
        super().__init__()
        self.activation = tf.nn.relu 
        self.config = config 
        self.batch_norm = batch_norm 
        self.fan_out_init = fan_out_init

    def __architecture_b(self, x, is_training):
                              
        #layer 1
        num_out_channels = self.fan_out_init # 64 
        x = convolution_layer(x= x, kernel_size= 5, num_out_channels= num_out_channels, activation= self.activation,
                            batch_norm= self.batch_norm, is_training= is_training, name = 'conv1')
        x = max_pooling_layer(x= x, pool_size=2, name= 'pool1')

        #layer 2 
        num_out_channels *= 2 # 128 
        x = convolution_layer(x= x, kernel_size= 5, num_out_channels= num_out_channels, activation= self.activation,
                            batch_norm= self.batch_norm, is_training= is_training, name = 'conv2')
        x = max_pooling_layer(x= x, pool_size=2, name= 'pool2')

        # layer 3 
        num_out_channels *= 2 # 256
        x = convolution_layer(x= x, kernel_size= 5, num_out_channels= num_out_channels, activation= self.activation,
                            batch_norm= self.batch_norm, is_training= is_training, name = 'conv3')
        x = max_pooling_layer(x= x, pool_size=2, name= 'pool3')

        # layer 4 
        num_out_channels *= 2 # 512
        x = convolution_layer(x= x, kernel_size= 5, num_out_channels= num_out_channels, activation= self.activation,
                            batch_norm= self.batch_norm, is_training= is_training, name = 'conv4')

        x = tf.keras.layers.Flatten()(x)

        return x

    def evaluate(self, x, is_training):
        if self.config == 'B':
            return self.__architecture_b(x, is_training)
        else:
            raise Exception( 'Unknown graph configuration')
    

class VGG(object):
    1
