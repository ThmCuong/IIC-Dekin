from data import data_load
import tensorflow_datasets as tfds
import tensorflow as tf 
# pick a data sets
DATA_SET = 'mnist'

# define splits
DS_CONFIG = {
    # mnist data set parameters
    'mnist': {
        'batch_size': 700,
        'num_repeats': 5,
        'mdl_input_dims': [24, 24, 1]}
}

# load the data set
TRAIN_SET, TEST_SET, SET_INFO = data_load(db_name=DATA_SET, with_info= True, **DS_CONFIG[DATA_SET])
# dataset, info = tfds.load('mnist', with_info = True)
# print(typeof(dataset))
print("done!!!")