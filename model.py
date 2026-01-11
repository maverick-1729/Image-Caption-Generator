import tensorflow as tf

class Embedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_dim, max_len, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.token_embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(max_len, embed_dim)

    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]
        position_ids = tf.range(start = 0, limit = length, delta = 1)
        position_ids = tf.expand_dims(position_ids, axis=0)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        return token_embeddings + position_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "max_len": self.max_len,
        })
        return config

class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation='relu')

    def call(self, x, training):
        x = self.layer_norm_1(x)
        x = self.dense(x)

        attn_out = self.attention(
            query = x,
            key = x,
            value = x,
            attention_mask = None,
            training = training
        )

        x = self.layer_norm_2(x + attn_out)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, units, num_heads, vocab_size, max_len, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.units = units
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        self.embedding = Embedding(vocab_size, embed_dim, max_len)

        self.encoder_proj = tf.keras.layers.Dense(embed_dim)

        key_dim = embed_dim // num_heads

        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=0.1
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()

        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)

        # self.out = tf.keras.layers.Dense(vocab_size, activation="softmax")

        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.dropout_2 = tf.keras.layers.Dropout(0.2)

    def get_casual_attn_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_length = input_shape[0], input_shape[1]
        i = tf.range(seq_length)[:, tf.newaxis]
        j = tf.range(seq_length)
        mask = tf.cast(i>=j, dtype='int32')
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        
        return tf.tile(mask, mult)

    def call(self, embeddings, encoder_output, training, causal_mask=None, padding_mask=None):
        # embeddings = self.embedding(input_ids)

        # causal_mask = None
        # causal_mask = self.get_casual_attn_mask(embeddings)
        # causal_mask = tf.cast(causal_mask, tf.float32)
        # padding_mask = None
        # combined_mask = causal_mask

        # if mask is not None:
        #     # causal_mask = self.get_casual_attn_mask(embeddings)
        #     padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.float32)
        #     padding_mask_2d = tf.cast(mask[:, tf.newaxis, :], dtype=tf.float32)
            # combined_mask = tf.minimum(padding_mask_2d, causal_mask)

        attn_output_1 = self.attention_1(
            query=embeddings,
            value=embeddings,
            key=embeddings,
            attention_mask=causal_mask,
            training=training
        )

        out_1 = self.layernorm_1(embeddings + attn_output_1)

        enc_proj = self.encoder_proj(encoder_output)

        attn_output_2 = self.attention_2(
            query=out_1,
            value=enc_proj,
            key=enc_proj,
            attention_mask=padding_mask,
            training=training
        )

        out_2 = self.layernorm_2(out_1 + attn_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        # preds = self.out(ffn_out)
        return ffn_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "units": self.units,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
        })
        return config

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, 
                 num_layers,
                 embed_dim, 
                 units, 
                 num_heads,
                 vocab_size,
                 max_len,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.units = units
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_len = max_len

        # shared embedding
        self.embedding = Embedding(vocab_size, embed_dim, max_len)

        # create N decoder blocks
        self.layers_list = [
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                units=units,
                num_heads=num_heads,
                vocab_size=vocab_size,
                max_len=max_len
            )
            for _ in range(num_layers)
        ]

        # final projection layer
        self.final_dense = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def get_causal_attn_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_length = input_shape[0], input_shape[1]
        i = tf.range(seq_length)[:, tf.newaxis]
        j = tf.range(seq_length)
        mask = tf.cast(i >= j, dtype='int32')
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        return tf.tile(mask, mult)

    def call(self, input_ids, encoder_output, training, mask=None):

        # token + positional embeddings
        embeddings  = self.embedding(input_ids)

        causal_mask = self.get_causal_attn_mask(embeddings)
        causal_mask = tf.cast(causal_mask, tf.float32)
        
        # Create combined mask if padding mask provided
        combined_mask = causal_mask
        padding_mask = None
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.float32)
            padding_mask_2d = tf.cast(mask[:, tf.newaxis, :], dtype=tf.float32)
            combined_mask = tf.minimum(padding_mask_2d, causal_mask)
        
        # Pass through each decoder block
        x = embeddings

        # pass through each decoder block
        for layer in self.layers_list:
            x = layer(x, encoder_output, training=training, 
                     causal_mask=combined_mask, padding_mask=padding_mask)

        # final prediction
        return self.final_dense(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "units": self.units,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "max_len": self.max_len
        })
        return config

class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, cnn_model, encoder, decoder, image_aug=None, **kwargs):
        super().__init__(**kwargs)
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")


    def calculate_loss(self, y_true, y_pred, mask):
        # loss = self.loss(y_true, y_pred)
        y_true_one_hot = tf.one_hot(y_true, depth=self.decoder.vocab_size)
        loss = self.loss(y_true_one_hot, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
    

    def compute_loss_and_acc(self, img_embed, captions, training=True):
        encoder_output = self.encoder(img_embed, training=training)
        y_input = captions[:, :-1]
        y_true = captions[:, 1:]
        mask = (y_input != 0)
        y_pred = self.decoder(
            y_input, encoder_output, training=training, mask= mask
        )
        loss = self.calculate_loss(y_true, y_pred, mask)
        acc = self.calculate_accuracy(y_true, y_pred, mask)
        return loss, acc

    
    def train_step(self, batch):
        imgs, captions = batch

        if self.image_aug:
            imgs = self.image_aug(imgs)
        
        img_embed = self.cnn_model(imgs)

        with tf.GradientTape() as tape:
            loss, acc = self.compute_loss_and_acc(
                img_embed, captions
            )
    
        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
    

    def test_step(self, batch):
        imgs, captions = batch

        img_embed = self.cnn_model(imgs)

        loss, acc = self.compute_loss_and_acc(
            img_embed, captions, training=False
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def call(self, inputs, training=False):
        # inputs is expected to be a list: [images, captions]
        images, captions = inputs
        
        # 1. extract features from image
        x = self.cnn_model(images, training=False)
        
        # 2. pass through encoder
        x = self.encoder(x, training=training)
        
        # 3. pass through decoder (captions + encoded images)
        # Note: Depending on your decoder, it might also need training=True/False
        y_pred = self.decoder(captions, x, training=training)
        
        return y_pred

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]
