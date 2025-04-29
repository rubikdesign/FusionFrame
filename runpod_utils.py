import argparse

def add_runpod_arguments(create_interface_function):
    """
    Decorator pentru a adăuga suport pentru argumente în linie de comandă pentru RunPod.
    Modifică funcția create_interface pentru a accepta argumente CLI.
    """
    def wrapper():
        parser = argparse.ArgumentParser(description='Platformă Gradio de Generare Imagini')
        parser.add_argument('--headless', action='store_true', help='Rulează în mod headless (fără interfață web)')
        parser.add_argument('--port', type=int, default=7860, help='Port pentru interfața web')
        parser.add_argument('--share', action='store_true', help='Permite accesul public la interfață')
        parser.add_argument('--model', type=str, default='SDXL 1.0', help='Modelul de utilizat')
        
        # Parametri pentru generare
        parser.add_argument('--prompt', type=str, help='Prompt pozitiv pentru generare')
        parser.add_argument('--negative_prompt', type=str, help='Prompt negativ pentru generare')
        parser.add_argument('--steps', type=int, default=30, help='Număr de inference steps')
        parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale')
        parser.add_argument('--seed', type=int, default=-1, help='Seed pentru generare')
        parser.add_argument('--width', type=int, default=768, help='Lățime imagine')
        parser.add_argument('--height', type=int, default=768, help='Înălțime imagine')
        parser.add_argument('--num_outputs', type=int, default=1, help='Număr de imagini de generat')
        parser.add_argument('--output_dir', type=str, help='Director pentru salvare imagini')
        
        # Parametri pentru imagini de input
        parser.add_argument('--woman_image', type=str, help='Cale către imaginea cu modelul')
        parser.add_argument('--clothing_image', type=str, help='Cale către imaginea cu îmbrăcămintea')
        parser.add_argument('--background_image', type=str, help='Cale către imaginea cu fundalul')
        
        # Parametri pentru ControlNet și IP-Adapter
        parser.add_argument('--controlnet_type', type=str, default='None', help='Tipul de ControlNet')
        parser.add_argument('--controlnet_scale', type=float, default=0.5, help='Scală pentru ControlNet')
        parser.add_argument('--ip_adapter_model', type=str, default='None', help='Modelul IP-Adapter')
        parser.add_argument('--ip_adapter_scale', type=float, default=0.6, help='Scală pentru IP-Adapter')
        
        args = parser.parse_args()
        
        # Dacă suntem în modul headless, generăm direct imaginile
        if args.headless and args.prompt:
            from app import generate_images  # Import lazily to avoid circular imports
            results, seed, message = generate_images(
                model_name=args.model,
                woman_image=args.woman_image,
                clothing_image=args.clothing_image,
                background_image=args.background_image,
                positive_prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                lora1="None", lora1_weight=0.7,
                lora2="None", lora2_weight=0.7,
                lora3="None", lora3_weight=0.7,
                lora4="None", lora4_weight=0.7,
                lora5="None", lora5_weight=0.7,
                controlnet_type=args.controlnet_type,
                controlnet_conditioning_scale=args.controlnet_scale,
                ip_adapter_name=args.ip_adapter_model,
                ip_adapter_scale=args.ip_adapter_scale,
                denoising_strength=0.75,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                image_width=args.width,
                image_height=args.height,
                scheduler_name="DPM++ 2M Karras",
                vae_name="Default",
                num_outputs=args.num_outputs,
                seed=args.seed
            )
            print(f"Generated images: {results}")
            print(f"Seed: {seed}")
            print(f"Message: {message}")
            return None  # Nu returnăm interfața în modul headless
        
        # Altfel, creăm interfața normal
        app = create_interface_function()
        app.queue(concurrency_count=1, max_size=20)
        app.launch(
            server_port=args.port,
            share=args.share
        )
        return app
    
    return wrapper

# Exemplu de utilizare:
"""
# În app.py, adăugați aceste modificări:

# 1. Importați decoratorul
from runpod_utils import add_runpod_arguments

# 2. Aplicați decoratorul funcției create_interface
@add_runpod_arguments
def create_interface():
    # Codul existent pentru crearea interfeței
    with gr.Blocks(...) as app:
        # ...
    return app

# 3. Modificați main pentru a folosi funcția decorată
if __name__ == "__main__":
    # În loc de:
    # app = create_interface()
    # app.queue(concurrency_count=1, max_size=20)
    # app.launch(share=True, server_port=7860)
    
    # Folosiți doar:
    create_interface()
"""
