import gradio as gr
import traceback
from inference import advanced_generate, get_image_features
 
def caption_image(img, method):
    try:
        if img is None:
            return "Please upload an image."
        
        encoder_output = get_image_features(img)

        if method=='Beam':
            return advanced_generate(encoder_output, method="beam", beam_width=3, temperature=1.0)
        elif method=='Sample':
            captions = []
            for i in range(5):
                caption = advanced_generate(
                    encoder_output,
                    method="sample",
                    temperature=1.1
                )
                captions.append(f"{i+1}. {caption}")
            return "\n".join(captions)
        
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=caption_image,
    inputs=[
        gr.Image(type="numpy", label="Upload an image"),
        gr.Radio(
            choices=["Beam", "Sample"],
            value="beam",
            label="Captioning Method"
        )
    ],
    outputs=gr.Textbox(label="Generated Caption", lines=5),
    title="Image Captioning Model",
    description="Upload an image and choose a decoding method to generate a caption."
)

demo.launch(share=True)
