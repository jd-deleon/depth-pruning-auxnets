import tensorflow as tf
"""
Definition of auxiliary networks using different architectures
"""


def dense_aux_model(model_input, num_classes, net_config):
  layers = net_config[0]
  x = model_input

  head_pool = tf.keras.layers.GlobalAveragePooling2D()
  flat = tf.keras.layers.Flatten()
  x = head_pool(x)
  x = flat(x)

  for i in range(layers):
    units = net_config[i+1]

    dense1 = tf.keras.layers.Dense(units, name='aux_block'+str(i)+'_dense')
    dense1_act = tf.keras.layers.ReLU(name='aux_block'+str(i)+'_relu')
    x = dense1(x)
    x = dense1_act(x)

  head_fc = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, name='aux_pred')
  x = head_fc(x)
  return x

def conv_aux_model(model_input, num_classes, net_config):
  layers = net_config[0]
  x = model_input

  for i in range(layers):
    # {number of conv features, conv filter height, width and stride in y,x dir.}
    n_feat = net_config[i*3+1]
    filter_height = net_config[i*3+2]
    filter_width = net_config[i*3+3]

    conv1 = tf.keras.layers.Conv2D(n_feat, (filter_height,filter_width), name='aux_block'+str(i)+'_conv')
    conv1_bn = tf.keras.layers.BatchNormalization(name='aux_block'+str(i)+'_bn')
    conv1_act = tf.keras.layers.ReLU(name='aux_block'+str(i)+'_relu')

    x = conv1(x)
    x = conv1_bn(x)
    x = conv1_act(x)

  # Head Layers
  head_pool = tf.keras.layers.GlobalAveragePooling2D(name='aux_avgpool')
  # head_flat = tf.keras.layers.Reshape([-1], name='aux_flat')
  head_flat = tf.keras.layers.Flatten()
  head_fc = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, name='aux_pred')

  x = head_pool(x)
  x = head_flat(x)
  x = head_fc(x)
  return x

def dscnn_aux_model(model_input, num_classes, net_config):
  layers = net_config[0]

  x = model_input

  #DW Conv Blocks
  for i in range(layers):
    # {number of conv features, conv filter height, width and stride in y,x dir.}
    n_feat = net_config[i*5+1]
    filter_height = net_config[i*5+2]
    filter_width = net_config[i*5+3]
    stride_y = net_config[i*5+4]
    stride_x = net_config[i*5+5]

    # Build Layers
    dw_conv = tf.keras.layers.DepthwiseConv2D([filter_height, filter_width],strides=[stride_y,stride_x], padding='SAME', name='aux_block'+str(i)+'_dwconv')
    dw_bn = tf.keras.layers.BatchNormalization(name='aux_block'+str(i)+'_bn1')
    dw_act = tf.keras.layers.ReLU(name='aux_block'+str(i)+'_relu1')

    pw_conv = tf.keras.layers.Conv2D(n_feat,[1, 1],padding='same', name='aux_block'+str(i)+'_pwconv')
    pw_bn = tf.keras.layers.BatchNormalization(name='aux_block'+str(i)+'_bn2')
    pw_act = tf.keras.layers.ReLU(name='aux_block'+str(i)+'_relu2')

    # Add layers to functional model
    x = dw_conv(x)
    x = dw_bn(x)
    x = dw_act(x)
    x = pw_conv(x)
    x = pw_bn(x)
    x = pw_act(x)

  # Head Layers
  head_pool = tf.keras.layers.GlobalAveragePooling2D(name='aux_avgpool')
  # head_flat = tf.keras.layers.Reshape([-1], name='aux_flat')
  head_flat = tf.keras.layers.Flatten()
  head_fc = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, name='aux_pred')

  x = head_pool(x)
  x = head_flat(x)
  x = head_fc(x)
  return x