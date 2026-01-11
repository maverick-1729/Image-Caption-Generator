import tensorflow as tf
import numpy as np
import pickle
from preprocess import preprocess_image
from model import TransformerEncoder, TransformerDecoder, ImageCaptioningModel
from cnn import CNN_encoder

embed_dim=256
max_len=40
vocab_size=8002

cnn_model = CNN_encoder()
encoder = TransformerEncoder(embed_dim, 1)
decoder = TransformerDecoder(
    num_layers=4,          
    embed_dim=256,
    units=512,
    num_heads=4,
    vocab_size=vocab_size,
    max_len=max_len
)

caption_model = ImageCaptioningModel(
    cnn_model=cnn_model,
    encoder=encoder,
    decoder=decoder,
    image_aug=None
)

dummy_img = tf.zeros((1, 224, 224, 3))
dummy_txt = tf.zeros((1, 40))

_ = caption_model([dummy_img, dummy_txt], training=False)

caption_model.load_weights("my_caption_model.weights.h5")

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

vocab = tokenizer.get_vocabulary()
clean_vocab = [token for token in vocab if token not in ['', '[UNK]']]

word2idx = tf.keras.layers.StringLookup(
    vocabulary = clean_vocab,
    invert=False,
    mask_token = '',           
    oov_token = '[UNK]' 
)

idx2word = tf.keras.layers.StringLookup(
    vocabulary = clean_vocab,
    invert = True,
    mask_token = '',            
    oov_token = '[UNK]' 
)

def get_image_features(img_path):
    img = preprocess_image(img_path)            
    img = tf.expand_dims(img, axis=0)                
    img_embed = caption_model.cnn_model(img)        
    return caption_model.encoder(img_embed, training=False)

def decode_ids_to_sentence(ids, idx2word):
   
    words = []
    for i in ids:
        if i == 0:               
            continue
        w = idx2word(tf.constant([i]))[0].numpy().decode("utf-8")
        if w in ("[start]", "[end]"):
            continue
        words.append(w)
    return " ".join(words)

def advanced_generate(encoder_output,
                      method="beam",     
                      beam_width=3,
                      top_k=40,
                      temperature=1.0,
                      repetition_penalty=1.2, 
                      alpha=0.6,             
                      max_len=None):

    if max_len is None:
        max_len = 40

    # --- Preprocessing & Encoding ---
    # img = preprocess_image(img_path)            
    # img = tf.expand_dims(img, axis=0)                
    # img_embed = caption_model.cnn_model(img)        
    # encoder_output = caption_model.encoder(img_embed, training=False)

    start_id = int(tokenizer(["[start]"])[0,0].numpy())  
    end_id   = int(tokenizer(["[end]"])[0,0].numpy())
    unk_id   = 1 

    def process_logits(logits, current_seq):
        logits[unk_id] = -1e10
        
        if temperature != 1.0:
            logits = logits / float(temperature)
            
        if repetition_penalty != 1.0:
            for tok in set(current_seq):
                if logits[tok] > 0:
                    logits[tok] /= repetition_penalty
                else:
                    logits[tok] *= repetition_penalty
        return logits

    if method == "greedy":
        y = [start_id]
        for _ in range(max_len-1):
            inp = tf.constant([y], dtype=tf.int32)
            mask = tf.cast(inp != 0, tf.int32)
            preds = caption_model.decoder(inp, encoder_output, training=False, mask=mask)
            
            logits = np.log(np.clip(preds[0, -1, :].numpy(), 1e-9, 1.0))
            logits = process_logits(logits, y)
            
            next_id = int(np.argmax(logits))
            if next_id == end_id:
                break
            y.append(next_id)
        return decode_ids_to_sentence(y, idx2word)

    # ---------- BEAM SEARCH ----------
    if method == "beam":
        sequences = [([start_id], 0.0)]
        
        for _step in range(max_len-1):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == end_id:
                    all_candidates.append((seq, score))
                    continue

                inp = tf.constant([seq], dtype=tf.int32)
                mask = tf.cast(inp != 0, tf.int32)
                preds = caption_model.decoder(inp, encoder_output, training=False, mask=mask)
                
                logits = np.log(np.clip(preds[0, -1, :].numpy(), 1e-9, 1.0))
                logits = process_logits(logits, seq)

                top_ids = np.argsort(logits)[-beam_width:]
                for tok in top_ids:
                    candidate = seq + [int(tok)]
                    new_score = score + float(logits[tok])
                    all_candidates.append((candidate, new_score))

            def get_lp_score(c_seq, c_score):
                lp = ((5 + len(c_seq))**alpha) / ((5 + 1)**alpha)
                return c_score / lp

            sequences = sorted(all_candidates, key=lambda x: get_lp_score(x[0], x[1]), reverse=True)[:beam_width]

            if all(seq[-1] == end_id for seq, sc in sequences):
                break

        return decode_ids_to_sentence(sequences[0][0], idx2word)

    # ---------- SAMPLING ----------
    if method == "sample":
        y = [start_id]
        for _ in range(max_len-1):
            inp = tf.constant([y], dtype=tf.int32)
            mask = tf.cast(inp != 0, tf.int32)
            preds = caption_model.decoder(inp, encoder_output, training=False, mask=mask)
            
            logits = np.log(np.clip(preds[0, -1, :].numpy(), 1e-9, 1.0))
            logits = process_logits(logits, y)

            topk_idx = np.argsort(logits)[-top_k:]
            topk_logits = logits[topk_idx]
            
            exp_logits = np.exp(topk_logits - np.max(topk_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            next_id = int(np.random.choice(topk_idx, p=probs))

            if next_id == end_id:
                break
            y.append(next_id)

        return decode_ids_to_sentence(y, idx2word)

    raise ValueError("Unknown method: choose 'greedy', 'beam', or 'sample'")