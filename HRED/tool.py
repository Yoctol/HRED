import tensorflow as tf


def check_layer_format(layer):
    if not isinstance(layer, dict):
        raise TypeError('layer should be a dict')
    for key, type_ in zip(['name', 'params'], [str, dict]):
        if key not in layer:
            raise KeyError('layer should have {}'.format(key))
        if not isinstance(layer[key], type_):
            raise TypeError(
                'layer {} should have type {}, now receive {}'.format(
                    key, type_, type(layer[key])))


def cast_to_float32(tensor_list):
    for num, tensor in enumerate(tensor_list):
        tensor_list[num] = tf.cast(tensor, tf.float32)
    return tensor_list
