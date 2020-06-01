import tensorflow as tf 
import tensorflow_datasets as tfds 
import tensorflow_addons as tfa 
import numpy as np 

def mnist_x(x_org, mdl_input_dim, is_training):
    '''
    x_orig: batch of images [bathc, height, weight, channels]
    mdl_input_dim: size of input dim we need to resize
    is_training: is this training set ?
    '''
    # rescale to [0, 1] 
    x_org = tf.cast(x_org, tf.float32) / 255 
    #get image size 
    height_width = mdl_input_dim[:-1] 
    n_chans = mdl_input_dim[-1] 

    # crop and resize
    if is_training:
        # moi anh su dung center crop and random crop
        # lay ngau nhien 1 trong 2 anh lam anh de train
        x1 = tf.image.central_crop(x_org, np.mean(20/np.array(x_org.shape.as_list()[1:-1])))
        x2 = tf.image.random_crop(x_org, tf.concat((tf.shape(x_org)[:1], [20, 20],[n_chans]), axis = 0))
        x = tf.stack([x1, x2])
        x = tf.transpose(x, [1,0,2,3,4]) #(N, 2, 20, 20, 1)
        i = tf.squeeze(tf.random.categorical([[1., 1.]], tf.shape(x)[0])) 
        x = tf.map_fn(lambda y: y[0][y[1]], (x, i), dtype = tf.float32)
        # resize 
        x = tf.image.resize(x,height_width)

    else:
        x = tf.image.central_crop(x_org, np.mean(20/np.array(x_org.shape.as_list()[1:-1])))
        x = tf.image.resize(x, height_width)
    return x 

def mnist_gx(x_org, mdl_input_dim, is_training, sample_repeats):
    '''
    x_org: batch of input image [N, height, width, channel]
    mdl_input_dim: expected dim
    is_training: is this data for training ?
    sample_repeats: number of repeat we want for this data
    '''
    # if not training, return a constant value--it 
    # will unused but needs to be same shape to avoid TensorFlow errors
    if not is_training: 
        return tf.zeros([0]+ mdl_input_dim) #[None, 24, 24, 1]

    # rescale to [0, 1] 
    x_org = tf.cast(x_org, dtype= tf.float32)/255

    # repeat sample (increase our sample) -- 5 times 
    x_org = tf.tile(x_org, [sample_repeats]+[1]*len(x_org.shape.as_list()[1:]))

    # get common shapes
    height_width = mdl_input_dim[:-1] 
    n_chans = mdl_input_dim[-1]

    # random rotation between [-rad, rad]
    rad = 2 * np.pi *25/360 # 25 degree
    x_rot = tfa.image.rotate(x_org, tf.random.uniform(shape= tf.shape(x_org)[:1], minval= -rad, maxval= rad))
    
    gx = tf.stack([x_org, x_rot])
    gx = tf.transpose(gx, [1, 0, 2, 3, 4])

    # we only need one between x_org and x_rot
    i = tf.squeeze(tf.random.categorical([[1., 1.]], tf.shape(gx)[0]))
    gx = tf.map_fn(lambda y: y[0][y[1]], (gx, i), tf.float32)

    # crop and chose one of them for train
    x1 = tf.image.random_crop(gx, tf.concat((tf.shape(gx)[:1], [16, 16],[n_chans]), axis=0))
    x2 = tf.image.random_crop(gx, tf.concat((tf.shape(gx)[:1], [20, 20],[n_chans]), axis=0))
    x3 = tf.image.random_crop(gx, tf.concat((tf.shape(gx)[:1], [24, 24],[n_chans]), axis=0))

    gx = tf.stack([x1, x2, x3])
    gx = tf.transpose(gx, [1, 0, 2, 3, 4])

    # we only need one between x1, x2, x3
    i = tf.squeeze(tf.random.categorical([[1., 1., 1.]], tf.shape(gx)[0]))
    gx = tf.map_fn(lambda y: y[0][y[1]], (gx, i), tf.float32) 

    # some color adjustment 
    def rand_adjust(img):
        # brightness
        img = tf.image.random_brightness(gx, 0.4)

        # contrast
        img = tf.image.random_contrast(gx,0.6, 1.4)

        # if it is color image then we adjust saturation, hue 
        if img.shape.as_list()[-1] == 3:
            img = tf.image.random_saturation(img,0.6, 1.4)
            img = tf.image.random_hue(img, 0.125) 
        return img 

    # color adjust
    gx = tf.map_fn(lambda y: rand_adjust(y), gx, dtype = tf.float32)
    return gx
 

def pre_process_data(ds, info, is_training, **kwargs):
    if info.name == 'nmist':
        return ds.map(lambda d: {
                                'x': mnist_x(d['image'], 
                                            kwargs['mdl_input_dim'],
                                            is_training = is_training),
                                'gx': mnist_gx(d['image'],
                                                kwargs['mdl_input_dim'],
                                                is_training),
                                'label': d['label']
                                }, 
                        num_parralel_calls = tf.data.experimental.AUTOTUNE)

def configure_data_set(ds, info, is_training, batch_size, **kwargs):

    #shuffle the data 
    ds = ds.shuffle(10* batch_size, reshuffle_each_iteration = True).repeat(1) 

    # batch the data before pre-processing
    ds = ds.batch(batch_size) # elm in ds: 700, 28, 28, 1
    # each batch: 700
    #preprocess data
    with tf.device("/cpu:0"):
        ds = pre_process_data(ds, info, is_training, **kwargs)
    ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    return ds

# rescale ve [0, 1]
def normalize(images, labels): 
    images = tf.cast(images, tf.float32)
    images /= 255 
    return images, labels 

# load va xu li data 
def data_load(db_name, with_info, **kwargs):
    # load data
    dataset, metadata = tfds.load(db_name, split = 'train + test',with_info = with_info)
    # train_dataset, test_dataset = dataset['train'], dataset['test'] 
    # image size : (28, 28, 1) -- mnist
    # # scale to [0, 1]
    # train_dataset = train_dataset.map(normalize)
    # test_dataset = test_dataset.map(normalize) 

    # save to cache, make training faster 
    # train_dataset = train_dataset.cache()
    # test_dataset = test_dataset.cache() 

    if 'train' in metadata.splits:
        train_ds = configure_data_set(ds = dataset, info = metadata, is_training= True, **kwargs)
    else:
        train_ds = None 

    if 'test' in metadata.splits: 
        test_ds = configure_data_set(ds = dataset, info= metadata, is_training= False, **kwargs)
    else:
        test_ds = None 

    return train_ds, test_ds, metadata

