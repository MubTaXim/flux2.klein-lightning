# FLUX.2 Klein 9B on Lightning AI

Deploy FLUX.2 Klein 9B with a Gradio WebUI on Lightning AI's free GPU tier.

![FLUX.2 Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/images/preview.png)

## Features

- 🚀 **Sub-second inference** - 4-step distilled model
- 🎨 **Text-to-Image** - Generate images from text prompts
- 🖼️ **Image Editing** - Use reference images for guided generation
- 💾 **CPU Offloading** - Run on 24GB VRAM GPUs (L4, A10G)
- 🌐 **Gradio WebUI** - Easy-to-use web interface
- ⚡ **No Safety Filters** - Full creative control

## Prerequisites

### 1. HuggingFace Access

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
3. **Accept the license agreement** (click "Agree and access repository")
4. Create an access token at [Settings > Tokens](https://huggingface.co/settings/tokens)

### 2. Lightning AI Account

1. Sign up at [lightning.ai](https://lightning.ai) (free tier available)
2. You get **15 free credits/month** (~31 GPU hours on L4)

## Quick Start

### Step 1: Create a GPU Studio

1. Go to [Lightning AI Studios](https://lightning.ai/studios)
2. Click **"New Studio"**
3. Select **L4 GPU** (24GB VRAM) - works with free tier
4. Wait for the Studio to start (~1-2 minutes)

### Step 2: Clone Repository

Open the terminal in your Studio and run:

```bash
# Clone the repository
git clone https://github.com/black-forest-labs/flux2.git
cd flux2

# Or if you have a fork with the Gradio app already:
git clone https://github.com/YOUR_USERNAME/flux2.git
cd flux2
```

### Step 3: Set Environment

```bash
# Set your HuggingFace token
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx

# Alternative: Add to Lightning AI Environment Variables
# Go to: Studio Settings > Environment Variables > Add HF_TOKEN
```

### Step 4: Run Setup

```bash
# Make setup script executable and run
chmod +x lightning_setup.sh
./lightning_setup.sh
```

This will:
- Install all dependencies
- Configure persistent storage for model cache
- Download FLUX.2 Klein 9B (~15GB, cached after first run)
- Create convenience scripts

### Step 5: Start the App

```bash
# Option 1: Use the convenience script
./start_gradio.sh

# Option 2: Run directly
python scripts/app_gradio.py
```

### Step 6: Access the UI

**Option A - Gradio Plugin (Recommended):**
1. Click the Gradio icon in the left sidebar of Lightning AI
2. This creates a secure tunnel to your app

**Option B - Share Link:**
1. The terminal will print a public URL like: `https://xxxxx.gradio.live`
2. Open this URL in your browser

## GPU Recommendations

| GPU | VRAM | CPU Offload | Speed | Cost |
|-----|------|-------------|-------|------|
| **L4** | 24GB | ✅ Required | ~3-5 sec | Free tier eligible |
| **A10G** | 24GB | ✅ Required | ~2-4 sec | ~$0.50/hr |
| **A100-40G** | 40GB | ❌ Optional | ~1-2 sec | ~$1.50/hr |
| **A100-80G** | 80GB | ❌ Optional | ~1 sec | ~$2.00/hr |

## Usage Tips

### Prompt Tips

FLUX.2 Klein excels at:
- Detailed scene descriptions
- Specific artistic styles
- Text rendering in images
- Photorealistic generation

Example prompts:
```
A cat holding a sign that says "Hello World", photorealistic

Portrait of a cyberpunk samurai, neon lights, rain, cinematic lighting, 8k

A cozy cabin in the mountains during winter, warm light from windows, aurora borealis
```

### Reference Image Editing

1. Click "Reference Images (Optional)" accordion
2. Upload one or more reference images
3. Describe what you want in the prompt
4. The model will use the references as guidance

### Memory Management

If you get "Out of Memory" errors:
1. ✅ Enable "CPU Offload" checkbox (enabled by default)
2. ✅ Reduce image dimensions (try 768x768)
3. ✅ Restart the kernel to clear cached memory

## Troubleshooting

### "Repository not found" or 401 Error

```
huggingface_hub.errors.RepositoryNotFoundError
```

**Solution:**
1. Ensure you've accepted the license at the [model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
2. Check your `HF_TOKEN` is set correctly
3. Verify your token has "Read" access

### CUDA Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:**
1. Enable CPU Offload in the UI
2. Lower image resolution
3. Run `torch.cuda.empty_cache()` in terminal

### Slow First Generation

First generation takes longer (~30-60 seconds) because:
- Models are loaded from disk to GPU
- Subsequent generations are much faster (~3-5 seconds)

### Models Re-downloading

If models download every time:
1. Check persistent storage is set up: `echo $HF_HOME`
2. Should be `/teamspace/studios/this_studio/.cache/huggingface`
3. Re-run `./lightning_setup.sh` if needed

## File Structure

```
flux2/
├── scripts/
│   ├── cli.py              # Original CLI (from BFL)
│   └── app_gradio.py       # Gradio WebUI (added)
├── src/flux2/              # Core model code (from BFL)
├── lightning_setup.sh      # Setup script (added)
├── start_gradio.sh         # Start script (created by setup)
├── .env.example            # Environment template (added)
├── README_LIGHTNING.md     # This file (added)
└── outputs/                # Generated images (symlinked)
```

## Model Information

| Property | Value |
|----------|-------|
| **Model** | FLUX.2 Klein 9B |
| **Parameters** | 9 billion |
| **Steps** | 4 (fixed, distilled) |
| **Guidance** | 1.0 (fixed, distilled) |
| **License** | [FLUX Non-Commercial](https://github.com/black-forest-labs/flux2/blob/main/model_licenses/LICENSE-FLUX-NON-COMMERICAL) |
| **Text Encoder** | Qwen3-8B |
| **Output Formats** | PNG (saved automatically) |

## Credits

- **Model**: [Black Forest Labs](https://blackforestlabs.ai/)
- **Repository**: [github.com/black-forest-labs/flux2](https://github.com/black-forest-labs/flux2)
- **Platform**: [Lightning AI](https://lightning.ai/)

## License

This Gradio interface is provided for use with the FLUX.2 model under its [Non-Commercial License](https://github.com/black-forest-labs/flux2/blob/main/model_licenses/LICENSE-FLUX-NON-COMMERICAL).

**For commercial use**, see [Black Forest Labs API](https://docs.bfl.ai/).
