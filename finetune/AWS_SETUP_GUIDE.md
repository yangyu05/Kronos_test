# AWS EC2 GPU Setup Guide for Kronos

This guide helps you set up PyTorch with CUDA on AWS EC2 GPU instances.

## Option 1: Use AWS Deep Learning AMI (Recommended - Easiest)

AWS Deep Learning AMIs come pre-configured with PyTorch, CUDA, and all dependencies.

### Step 1: Launch Instance with Deep Learning AMI

1. Go to EC2 Console → Launch Instance
2. Search for "Deep Learning AMI" in the AMI marketplace
3. Select: **Deep Learning AMI GPU PyTorch (Ubuntu)** or **Deep Learning AMI GPU PyTorch (Amazon Linux)**
4. Choose your GPU instance type (e.g., `g5.2xlarge`, `p3.8xlarge`)
5. Launch the instance

### Step 2: Connect and Verify

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Verify CUDA
nvidia-smi

# Verify PyTorch with CUDA
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

If PyTorch is already installed, you're done! Skip to "Setting Up Kronos" section.

## Option 2: Install PyTorch Manually on Ubuntu/Debian

If you're using a standard Ubuntu AMI, follow these steps:

### Step 1: Install NVIDIA Drivers and CUDA

```bash
# Update system
sudo apt-get update

# Install NVIDIA drivers (for Ubuntu 20.04/22.04)
sudo apt-get install -y nvidia-driver-535  # or latest version
sudo reboot  # Reboot to load drivers

# After reboot, verify
nvidia-smi
```

### Step 2: Install CUDA Toolkit

```bash
# Download and install CUDA 11.8 (recommended for PyTorch)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Install PyTorch with CUDA

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1 (newer)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Option 3: Quick Install Script (All-in-One)

Create and run this script on your AWS instance:

```bash
#!/bin/bash
# save as install_pytorch_cuda.sh

set -e

echo "=== Installing PyTorch with CUDA on AWS ==="

# Check if running on Ubuntu/Debian
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS"
    exit 1
fi

# Install Python and pip if not present
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 (most compatible)
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
echo ""
echo "=== Verification ==="
python3 -c "import torch; print('✓ PyTorch version:', torch.__version__); print('✓ CUDA available:', torch.cuda.is_available()); print('✓ CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('✓ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "=== Installation Complete ==="
echo "Activate virtual environment: source venv/bin/activate"
```

Run it:
```bash
chmod +x install_pytorch_cuda.sh
./install_pytorch_cuda.sh
```

## Setting Up Kronos on AWS

After PyTorch is installed:

### Step 1: Clone/Upload Kronos Code

```bash
# Option A: Clone from GitHub
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos

# Option B: Upload your local code using SCP
# From your local machine:
# scp -i your-key.pem -r /path/to/Kronos ubuntu@your-instance-ip:~/
```

### Step 2: Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # or use your DLAMI conda environment

# Install requirements
pip install -r requirements.txt

# Install qlib
pip install git+https://github.com/microsoft/qlib.git
```

### Step 3: Verify Everything Works

```bash
# Check PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Check qlib
python -c "import qlib; print('Qlib installed')"

# Check Kronos imports
python -c "from model import Kronos, KronosTokenizer; print('Kronos imports OK')"
```

## Common Issues and Solutions

### Issue: "CUDA not available" after installing PyTorch

**Solution**: 
1. Check NVIDIA drivers: `nvidia-smi`
2. If drivers missing, install them:
   ```bash
   sudo apt-get install -y nvidia-driver-535
   sudo reboot
   ```
3. Verify CUDA path: `echo $LD_LIBRARY_PATH`

### Issue: "No module named 'torch'"

**Solution**: 
1. Ensure virtual environment is activated: `source venv/bin/activate`
2. Reinstall PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Issue: Wrong CUDA version

**Solution**: 
- Check CUDA version: `nvcc --version` or `nvidia-smi`
- Install matching PyTorch version:
  - CUDA 11.8: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
  - CUDA 12.1: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Issue: Out of memory during training

**Solution**: 
1. Reduce batch size in `finetune/config.py`: `self.batch_size = 25`
2. Use gradient accumulation: `self.accumulation_steps = 2`

## Quick Start Commands

Once everything is set up:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Verify GPU
python -c "import torch; assert torch.cuda.is_available(); print('GPU OK')"

# 3. Run training
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```

## Recommended AWS Setup

**For best experience:**
1. Use **Deep Learning AMI GPU PyTorch** (saves time)
2. Instance: `g5.2xlarge` or `p3.8xlarge`
3. Storage: At least 50GB for models and data
4. Security: Open port 22 (SSH) in security group

## Cost Optimization Tips

1. **Use Spot Instances**: Can save 70-90% on GPU instances
2. **Stop when not training**: GPU instances are expensive
3. **Use smaller instance for testing**: `g4dn.xlarge` for initial setup
4. **Monitor costs**: Set up CloudWatch billing alerts

## Next Steps

After PyTorch is installed:
1. Process your data: `python finetune/qlib_data_preprocess.py`
2. Start training: `torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py`

