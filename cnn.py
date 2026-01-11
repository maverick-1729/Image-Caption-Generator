import tensorflow as tf

def CNN_encoder():

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    output = base_model.output   
    output = tf.keras.layers.Reshape((-1, output.shape[-1]))(output)

    cnn_model = tf.keras.models.Model(base_model.input, output)
    return cnn_model