import tensorflow as tf

def preprocess_image(img):
    img = tf.convert_to_tensor(img)
    # img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(224, 224)(img)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img