import tensorflow as tf
elements =  ([{"a": 1, "b": "foo"},
              {"a": 2, "b": "bar"},
              {"a": 3, "b": "baz"},
              {"a": 4, "b": "baz4"},
              {"a": 5, "b": "baz5"},
              {"a": 6, "b": "baz6"},
              {"a": 7, "b": "baz7"},
              {"a": 8, "b": "baz8"}])
dataset = tf.data.Dataset.from_generator(
    lambda: elements, {"a": tf.int32, "b": tf.string})

# `map_func` takes a single argument of type `dict` with the same keys
# as the elements.
# result = dataset.map(lambda d:   d["b"])
# print(list(result.as_numpy_iterator()))

def fixMap(ass):
    # print(" va day la : ", len(ass.shape.as_list()))
    print(" tf.shape : ", tf.shape(ass))
    ass = tf.map_fn(lambda y: y+2, ass, dtype = tf.int32)
    return ass
ds = dataset.batch(3)
print(list(ds.as_numpy_iterator()))
t = ds.map(lambda d: { 'x':fixMap(d['a'])})
print(list(t.as_numpy_iterator()))
# print(" tf.shape emls: ", tf.shape(dataset))



