import gradio as gr
import torch
from PIL import Image
import numpy as np
import os
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline

# Verifică versiunea torch
print(f"PyTorch version: {torch.__version__}")

# Setează device-ul (CUDA dacă este disponibil, altfel CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Folosim device-ul: {device}")

# Definește model_id
model_id = "RunDiffusion/Juggernaut-XL-v9"

# Funcție pentru a încărca modelul în mod sigur
def load_model_safely():
    try:
        # Încarcă modelul cu StableDiffusionXLImg2ImgPipeline
        try:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None
            )
        except Exception as e:
            print(f"Eroare la încărcarea modelului cu StableDiffusionXLImg2ImgPipeline: {e}")
            # Fallback la DiffusionPipeline generic
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None
            )
            
        pipe.to(device)
        return pipe
    except Exception as e:
        print(f"Eroare la încărcarea modelului: {e}")
        raise e

# Încarcă modelul
pipe = load_model_safely()

def process_image(
    init_image, 
    prompt, 
    negative_prompt, 
    strength, 
    guidance_scale, 
    num_inference_steps, 
    seed, 
    keep_original_size, 
    width, 
    height
):
    if init_image is None:
        return None, "Nu a fost încărcată nicio imagine."
    
    # Convertește la formatul PIL
    if isinstance(init_image, np.ndarray):
        init_image = Image.fromarray(init_image)
    
    # Asigură-te că imaginea este RGB
    if init_image.mode != "RGB":
        init_image = init_image.convert("RGB")
    
    # Setează dimensiunile
    if keep_original_size:
        width, height = init_image.size
    else:
        # Asigură-te că width și height sunt multipli de 8 (cerință pentru SDXL)
        width = (width // 8) * 8
        height = (height // 8) * 8
        init_image = init_image.resize((width, height), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC)
    
    # Setează seed-ul pentru reproducibilitate
    if seed == -1:
        seed = torch.randint(0, 2147483647, (1,)).item()
    
    try:
        # Creează generator cu compatibilitate pentru versiuni mai vechi de PyTorch
        if hasattr(torch.Generator, device):
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            # Fallback pentru versiuni mai vechi de PyTorch
            generator = torch.Generator().manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
    
        # Procesează imaginea cu tratarea excepțiilor
        try:
            # Verifică dacă pipe are metode specifice
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
            
            # Verifică pentru avertismente de siguranță
            if hasattr(result, "nsfw_content_detected") and result.nsfw_content_detected[0]:
                return None, "Conținutul generat a fost marcat ca potențial nepotrivit."
            
            # Extrage imaginea rezultată
            output_image = result.images[0]
            
        except AttributeError as attr_err:
            # Tratează cazul când pipe nu are metodele așteptate
            return None, f"Eroare de compatibilitate: {str(attr_err)}. Modelul încărcat nu suportă toate funcțiile necesare."
        
        return output_image, f"Imagine procesată cu succes. Seed: {seed}"
    
    except Exception as e:
        return None, f"Eroare în timpul procesării: {str(e)}"

# Creează interfața Gradio
with gr.Blocks(title="Editor Imagini Juggernaut-XL") as app:
    gr.Markdown("# Editor Imagini cu Juggernaut-XL-v9")
    gr.Markdown("Încărcați o imagine și ajustați parametrii pentru a o edita folosind modelul Juggernaut-XL-v9.")
    
    with gr.Row():
        with gr.Column():
            # Intrări
            init_image = gr.Image(label="Imagine de intrare", type="pil")
            prompt = gr.Textbox(label="Prompt pozitiv", placeholder="Descrieți ce doriți să vedeți în imagine")
            negative_prompt = gr.Textbox(label="Prompt negativ", placeholder="Descrieți ce NU doriți să vedeți în imagine", value="low quality, blurry, distorted")
            
            with gr.Accordion("Setări avansate", open=False):
                strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Strength (puterea modificării)")
                guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=7.5, label="Guidance Scale")
                num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Pași de inferență")
                seed = gr.Slider(minimum=-1, maximum=2147483647, step=1, value=-1, label="Seed (-1 pentru aleatoriu)")
                
                keep_original_size = gr.Checkbox(label="Păstrează dimensiunea originală", value=True)
                with gr.Group(visible=False) as size_group:
                    width = gr.Slider(minimum=256, maximum=1280, step=8, value=512, label="Lățime")
                    height = gr.Slider(minimum=256, maximum=1280, step=8, value=512, label="Înălțime")
                
                # Actualizează vizibilitatea setărilor de dimensiune
                keep_original_size.change(
                    fn=lambda x: gr.Group.update(visible=not x),
                    inputs=keep_original_size,
                    outputs=size_group
                )
            
            # Buton de procesare
            process_btn = gr.Button("Procesează Imaginea")
        
        with gr.Column():
            # Ieșiri
            output_image = gr.Image(label="Imagine rezultată")
            output_text = gr.Textbox(label="Mesaj", interactive=False)
    
    # Conectează funcția de procesare
    process_btn.click(
        fn=process_image,
        inputs=[
            init_image, 
            prompt, 
            negative_prompt, 
            strength, 
            guidance_scale, 
            num_inference_steps, 
            seed, 
            keep_original_size, 
            width, 
            height
        ],
        outputs=[output_image, output_text]
    )
    
    gr.Markdown("""
    ## Instrucțiuni de utilizare
    1. Încărcați o imagine
    2. Introduceți un prompt pozitiv descriind modificările dorite
    3. (Opțional) Introduceți un prompt negativ pentru a evita anumite aspecte
    4. Ajustați setările avansate dacă este necesar
    5. Apăsați butonul "Procesează Imaginea"
    
    ### Sfaturi pentru prompturi eficiente:
    - Fiți cât mai specifici posibil în descriere
    - Pentru rezultate mai bune, includeți detalii despre stil, lumină, compoziție
    - În promptul negativ, includeți defectele pe care doriți să le evitați
    """)

# Lansează aplicația
if __name__ == "__main__":
    app.launch(share=True)
