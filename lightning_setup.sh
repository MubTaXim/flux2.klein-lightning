#!/bin/bash
# =============================================================================
#  FLUX.2 Klein 9B - Lightning AI Studio Setup Script
# =============================================================================
#
#  This script sets up everything needed to run FLUX.2 Klein 9B on Lightning AI
#
#  Usage:
#    1. Create a new Lightning AI Studio with L4 or A10G GPU
#    2. Clone this repository
#    3. Set your HuggingFace token: export HF_TOKEN=your_token
#    4. Run this script: ./lightning_setup.sh
#
# =============================================================================

set -e  # Exit on any error

echo ""
echo "============================================"
echo "  FLUX.2 Klein 9B - Lightning AI Setup"
echo "============================================"
echo ""

# =============================================================================
# 1. Check prerequisites
# =============================================================================
echo "🔍 Checking prerequisites..."

# Check if running on Linux (Lightning AI uses Linux)
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  Warning: This script is designed for Lightning AI (Linux)"
    echo "   You appear to be running on: $OSTYPE"
fi

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python version: $PYTHON_VERSION"

# =============================================================================
# 2. Set up persistent storage
# =============================================================================
echo ""
echo "📁 Setting up persistent storage..."

# Use Lightning AI's persistent storage for HuggingFace cache
if [[ -d "/teamspace/studios/this_studio" ]]; then
    # We're on Lightning AI
    export HF_HOME="/teamspace/studios/this_studio/.cache/huggingface"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    OUTPUTS_DIR="/teamspace/studios/this_studio/outputs"
    echo "✓ Using Lightning AI persistent storage"
else
    # Local development
    export HF_HOME="$HOME/.cache/huggingface"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    OUTPUTS_DIR="./outputs"
    echo "✓ Using local cache directory"
fi

mkdir -p "$HF_HOME"
mkdir -p "$OUTPUTS_DIR"
echo "  HF_HOME: $HF_HOME"
echo "  Outputs: $OUTPUTS_DIR"

# Create symlink for outputs in project directory
if [[ ! -L "./outputs" ]] && [[ ! -d "./outputs" ]]; then
    ln -sf "$OUTPUTS_DIR" "./outputs"
    echo "✓ Created outputs symlink"
fi

# =============================================================================
# 3. Check HuggingFace token
# =============================================================================
echo ""
echo "🔑 Checking HuggingFace authentication..."

if [[ -z "$HF_TOKEN" ]]; then
    echo ""
    echo "⚠️  HF_TOKEN environment variable is not set!"
    echo ""
    echo "   The FLUX.2 Klein 9B model requires authentication."
    echo "   Please set your token:"
    echo ""
    echo "   Option 1 - Export in terminal:"
    echo "     export HF_TOKEN=hf_xxxxxxxxxx"
    echo ""
    echo "   Option 2 - Lightning AI Environment Variables:"
    echo "     Go to Studio Settings > Environment Variables"
    echo ""
    echo "   Get your token at: https://huggingface.co/settings/tokens"
    echo ""
    read -p "Enter your HuggingFace token now (or press Enter to skip): " token_input
    if [[ -n "$token_input" ]]; then
        export HF_TOKEN="$token_input"
        echo "✓ Token set for this session"
    else
        echo "⚠️  Continuing without token. Model download may fail."
    fi
else
    echo "✓ HF_TOKEN is set"
fi

# =============================================================================
# 4. Install dependencies
# =============================================================================
echo ""
echo "📦 Installing dependencies..."

# Determine CUDA version for PyTorch
CUDA_VERSION="cu124"  # Default for Lightning AI
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | cut -d',' -f1)
    echo "  CUDA version: $NVCC_VERSION"
fi

# Install the flux2 package in editable mode
pip install -e . --extra-index-url https://download.pytorch.org/whl/$CUDA_VERSION --no-cache-dir --quiet

# Install Gradio
pip install gradio>=4.0.0 --quiet

echo "✓ Dependencies installed"

# =============================================================================
# 5. Pre-download models (optional but recommended)
# =============================================================================
echo ""
echo "📥 Pre-downloading models..."
echo "   This may take 10-15 minutes on first run."
echo ""

python3 << 'PYTHON_SCRIPT'
import os
import sys

# Set token if available
hf_token = os.environ.get('HF_TOKEN', '')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token

try:
    from flux2.util import load_flow_model, load_ae, load_text_encoder
    
    print("  Loading text encoder (Qwen3-8B)...")
    load_text_encoder('flux.2-klein-9b', device='cpu')
    print("  ✓ Text encoder cached")
    
    print("  Loading flow model (Klein 9B)...")
    load_flow_model('flux.2-klein-9b', device='cpu')
    print("  ✓ Flow model cached")
    
    print("  Loading autoencoder...")
    load_ae('flux.2-klein-9b', device='cpu')
    print("  ✓ Autoencoder cached")
    
    print("\n✓ All models downloaded and cached!")
    
except Exception as e:
    print(f"\n⚠️  Model download failed: {e}")
    print("   You may need to:")
    print("   1. Accept the license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B")
    print("   2. Ensure your HF_TOKEN has access to the model")
    sys.exit(1)
PYTHON_SCRIPT

# =============================================================================
# 6. Create convenience scripts
# =============================================================================
echo ""
echo "📝 Creating convenience scripts..."

# Create start script
cat > start_gradio.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="src:$PYTHONPATH"
python scripts/app_gradio.py
EOF
chmod +x start_gradio.sh
echo "✓ Created start_gradio.sh"

# =============================================================================
# 7. Done!
# =============================================================================
echo ""
echo "============================================"
echo "  ✅ Setup Complete!"
echo "============================================"
echo ""
echo "  To start the Gradio app:"
echo ""
echo "    ./start_gradio.sh"
echo ""
echo "  Or manually:"
echo ""
echo "    python scripts/app_gradio.py"
echo ""
echo "  Then:"
echo "    - Use the Gradio plugin in Lightning AI for a public URL"
echo "    - Or use the share link printed in the terminal"
echo ""
echo "============================================"
