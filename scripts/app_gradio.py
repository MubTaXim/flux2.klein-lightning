"""
FLUX.2 Klein 9B Gradio WebUI
Leverages the official BFL inference code from src/flux2/

No NSFW safety filters - full creative control.
"""
import sys
import os

# Ensure flux2 module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from pathlib import Path

import gradio as gr
import torch
from einops import rearrange
from PIL import Image

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import load_ae, load_flow_model, load_text_encoder, FLUX2_MODEL_INFO

# =============================================================================
# Global State
# =============================================================================
MODEL_NAME = "flux.2-klein-9b"
model = None
text_encoder = None
ae = None
is_loaded = False


def get_model_info():
    """Get model configuration info."""
    info = FLUX2_MODEL_INFO.get(MODEL_NAME, {})
    return {
        "num_steps": info.get("defaults", {}).get("num_steps", 4),
        "guidance": info.get("defaults", {}).get("guidance", 1.0),
        "fixed_params": info.get("fixed_params", set()),
    }


def load_models(cpu_offload: bool = True, progress=None):
    """Load all model components.
    
    Optimized strategy for L4 (24GB):
    - Text encoder on CPU (9GB) - runs slower but avoids PCIe transfer
    - Flow model on GPU (18GB) - stays permanently, never moves
    - VAE on GPU (0.3GB) - stays permanently
    Total GPU: 18.3GB - fits comfortably!
    """
    global model, text_encoder, ae, is_loaded
    
    if is_loaded:
        return
    
    if progress:
        progress(0.1, desc="Loading text encoder (CPU)...")
    print("Loading text encoder (on CPU to save GPU VRAM)...")
    # Text encoder stays on CPU - we'll run inference on CPU
    # This is slower but avoids expensive PCIe transfers
    text_encoder = load_text_encoder(MODEL_NAME, device="cpu")
    
    if progress:
        progress(0.4, desc="Loading flow model (GPU)...")
    print("Loading flow model (on GPU - stays permanently)...")
    # Flow model goes directly to GPU and stays there
    model = load_flow_model(MODEL_NAME, device="cuda")
    
    if progress:
        progress(0.7, desc="Loading autoencoder (GPU)...")
    print("Loading autoencoder (on GPU - stays permanently)...")
    # VAE on GPU - only 0.3GB
    ae = load_ae(MODEL_NAME, device="cuda")
    ae.eval()
    
    is_loaded = True
    if progress:
        progress(0.9, desc="Models loaded!")
    
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"✓ All models loaded! GPU VRAM: {vram:.1f}GB")



def generate_image(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    input_images: list = None,
    progress=gr.Progress()
):
    """Generate image using FLUX.2 Klein 9B."""
    global model, text_encoder, ae
    import time
    
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt"
    
    start_time = time.time()
    
    def log(msg):
        elapsed = time.time() - start_time
        vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"[{elapsed:6.1f}s] [VRAM: {vram:.1f}GB] {msg}")
    
    log(f"Starting generation: {prompt[:50]}...")
    
    # Load models if needed
    if not is_loaded:
        progress(0.05, desc="Loading models (first run)...")
        load_models(progress=progress)
    
    # Get fixed parameters for Klein 9B distilled
    model_info = get_model_info()
    num_steps = model_info["num_steps"]
    guidance = model_info["guidance"]
    
    # Handle seed
    actual_seed = seed if seed >= 0 else random.randrange(2**31)
    log(f"Seed: {actual_seed}, Size: {width}x{height}, Steps: {num_steps}")
    
    try:
        with torch.no_grad():
            # Step 1: Encode reference images if provided
            ref_tokens, ref_ids = None, None
            if input_images and len(input_images) > 0:
                log("Encoding reference images...")
                progress(0.15, desc="Encoding reference images...")
                img_ctx = []
                for img_file in input_images:
                    if img_file is not None:
                        if hasattr(img_file, 'name'):
                            img_ctx.append(Image.open(img_file.name))
                        else:
                            img_ctx.append(Image.open(img_file))
                if img_ctx:
                    # AE is already on GPU with our optimized loading strategy
                    ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)
                log(f"Encoded {len(img_ctx)} reference images")
            
            # Step 2: Encode text (ON CPU - slower but no transfer!)
            log("Encoding prompt (on CPU)...")
            progress(0.2, desc="Encoding prompt (CPU)...")
            
            # Text encoder runs on CPU, then we move just the embeddings to GPU
            # Embeddings are tiny (~1MB) so transfer is instant
            encode_start = time.time()
            ctx = text_encoder([prompt])  # Runs on CPU
            ctx = ctx.to(device="cuda", dtype=torch.bfloat16)  # Move tiny output to GPU
            ctx, ctx_ids = batched_prc_txt(ctx)
            log(f"Prompt encoded in {time.time() - encode_start:.1f}s (CPU inference)")
            
            # Step 3: No model movement needed - flow model already on GPU!
            
            # Step 4: Create noise
            progress(0.3, desc="Creating noise...")
            shape = (1, 128, height // 16, width // 16)
            generator = torch.Generator(device="cuda").manual_seed(actual_seed)
            randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
            x, x_ids = batched_prc_img(randn)
            
            # Step 5: Denoise (model already on GPU!)
            log(f"Denoising ({num_steps} steps)...")
            progress(0.4, desc=f"Denoising ({num_steps} steps)...")
            timesteps = get_schedule(num_steps, x.shape[1])
            denoise_start = time.time()
            x = denoise(
                model,  # Already on GPU!
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
            log(f"Denoising complete in {time.time() - denoise_start:.1f}s")
            
            # Step 6: Decode (VAE already on GPU!)
            progress(0.8, desc="Decoding...")
            log("Decoding latents...")
            decode_start = time.time()
            x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
            x = ae.decode(x).float()  # Already on GPU!
            log(f"Decode complete in {time.time() - decode_start:.1f}s")
        
        # Convert to PIL
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        
        # Save to outputs
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"flux2_klein_{actual_seed}.png"
        img.save(output_path, quality=95)
        
        total_time = time.time() - start_time
        log(f"✓ Complete! Total time: {total_time:.1f}s")
        
        progress(1.0, desc="Done!")
        return img, f"✓ Saved to {output_path} | Seed: {actual_seed} | Size: {width}x{height} | Time: {total_time:.1f}s"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


# =============================================================================
# Gradio UI
# =============================================================================
def create_ui():
    """Create the Gradio interface."""
    
    model_info = get_model_info()
    
    with gr.Blocks(
        title="FLUX.2 Klein 9B",
    ) as demo:
        
        gr.Markdown("""
        # 🎨 FLUX.2 Klein 9B - Image Generator
        
        **Sub-second generation on consumer GPUs** | Built on the [official BFL repository](https://github.com/black-forest-labs/flux2)
        
        > This model uses **fixed parameters**: `steps=4`, `guidance=1.0` (distilled model)
        """)
        
        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="✍️ Prompt",
                    placeholder="A cat holding a sign that says 'hello world'",
                    lines=4,
                    max_lines=10,
                )
                
                with gr.Row():
                    width = gr.Slider(
                        minimum=256,
                        maximum=1536,
                        value=1024,
                        step=64,
                        label="Width"
                    )
                    height = gr.Slider(
                        minimum=256,
                        maximum=1536,
                        value=1024,
                        step=64,
                        label="Height"
                    )
                
                with gr.Row():
                    seed = gr.Number(
                        value=-1,
                        label="Seed (-1 = random)",
                        precision=0
                    )
                
                with gr.Accordion("📷 Reference Images (Optional)", open=False):
                    gr.Markdown("Upload reference images for image-to-image editing")
                    ref_images = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Reference Images"
                    )
                
                generate_btn = gr.Button(
                    "🚀 Generate",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"]
                )
                
                # Model info display
                gr.Markdown(f"""
                ---
                **Model Info:**
                - Steps: `{model_info['num_steps']}` (fixed)
                - Guidance: `{model_info['guidance']}` (fixed)
                - GPU VRAM: ~18.5GB (optimized for L4)
                - Text encoder runs on CPU
                """)
            
            # Right column - Output
            with gr.Column(scale=3):
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=512
                )
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["status-box"]
                )
        
        # Example prompts
        gr.Examples(
            examples=[
                ["A serene Japanese garden with cherry blossoms, koi pond, and a small wooden bridge, morning light"],
                ["Portrait of a cyberpunk samurai, neon lights, rain, cinematic lighting, 8k detailed"],
                ["A cozy cabin in the mountains during winter, warm light from windows, northern lights in sky"],
                ["Macro photography of a dewdrop on a rose petal, golden hour lighting, extreme detail"],
            ],
            inputs=[prompt],
            label="Example Prompts"
        )
        
        # Connect events
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, width, height, seed, ref_images],
            outputs=[output_image, status]
        )
        
        # Also generate on Enter key in prompt
        prompt.submit(
            fn=generate_image,
            inputs=[prompt, width, height, seed, ref_images],
            outputs=[output_image, status]
        )
    
    return demo


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  FLUX.2 Klein 9B - Gradio WebUI")
    print("=" * 50)
    print(f"  Model: {MODEL_NAME}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 50)
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
