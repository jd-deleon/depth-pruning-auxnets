import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

import argparse
import os

def get_flops(ckpt_path):
    model = tf.keras.models.load_model(ckpt_path)
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops

# model = tf.keras.models.Sequential()
# model.add(tf.keras.Input(shape=(10 ,10 ,3)))
# model.add(tf.keras.layers.Conv2D(2, 3, activation='relu'))
# model.summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_name',
        type=str)
    parser.add_argument(
        '--skip_parser',
        type=str,
        default="False")
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.skip_parser == "True":
        subfolder = FLAGS.ckpt_name
    else:
        subfolders = [f.path for f in os.scandir(FLAGS.ckpt_name) if f.is_dir()]
        latest_model = -1
        for sub_path in subfolders:
            split_path = sub_path.split("_")
            if latest_model < float(split_path[-1]):
                latest_model=float(split_path[-1])

        subfolder = FLAGS.ckpt_name + "/" + FLAGS.ckpt_name.split("/")[1] +"_" + ('%.4f' % latest_model)
    # get_flops()
    print("The {} FLOPs is: {}".format(subfolder, get_flops(subfolder)) ,flush=True )

    # get model information
    # model = tf.keras.models.load_model(FLAGS.ckpt_name)
    # model.summary()