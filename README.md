# Image-Caption-Generator

Built an end-to-end image captioning model using EfficientNetB0 for visual feature extraction and a custom Transformer encoder–decoder for sequence generation. Trained on a subset of the MS COCO dataset with multi-caption supervision and evaluated using BLEU and CIDEr metrics. Implemented multiple decoding strategies including greedy decoding and beam search.

## Future Improvements
- Fine-tune the CNN encoder for better visual grounding
- Train with reinforcement learning (CIDEr-optimized loss)
- Replace the CNN with a Vision Transformer (ViT)
- Improve caption diversity using advanced sampling strategies
