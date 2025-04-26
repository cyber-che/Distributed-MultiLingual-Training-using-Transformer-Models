#!/bin/bash

# Usage:
# 1. Make the script executable: chmod +x nvidia_setup.sh
# 2. Run the script: ./nvidia_setup.sh

# Define the recommended NVIDIA driver version
DRIVER_VERSION="535"

# Step 1: Update package lists
echo "Updating package lists..."
sudo apt-get update

# Step 2: Remove any existing NVIDIA drivers to prevent conflicts
echo "Removing existing NVIDIA drivers..."
sudo apt-get purge -y nvidia-*
sudo apt-get autoremove -y
sudo apt-get clean

# Step 3: Install required dependencies
echo "Installing build tools and kernel headers..."
sudo apt-get install -y build-essential dkms linux-headers-$(uname -r)

# Step 4: Install the NVIDIA driver
echo "Installing NVIDIA driver version $DRIVER_VERSION..."
sudo apt-get install -y nvidia-driver-$DRIVER_VERSION

# Step 5: Rebuild initramfs to include NVIDIA kernel modules
echo "Rebuilding initramfs..."
sudo update-initramfs -u

# Step 6: Load the NVIDIA kernel module
echo "Loading the NVIDIA kernel module..."
sudo modprobe nvidia

# Step 7: Verify that the NVIDIA module is loaded
echo "Checking loaded modules..."
lsmod | grep nvidia

# Step 8: Install the CUDA Toolkit
echo "Installing CUDA Toolkit..."
sudo apt-get install -y nvidia-cuda-toolkit

# Step 9: Verify CUDA installation
echo "Verifying CUDA installation..."
nvcc --version

# Step 10: Display driver and GPU information
echo "NVIDIA Driver and GPU Info:"
nvidia-smi

echo "IMPORTANT: Installation complete. Please reboot your system for changes to take effect."

#python3 t5_small_v1.py --rank 0 --world_size 4
#python3 t5_small_v1.py --rank 1 --world_size 4
#python3 t5_small_v1.py --rank 2 --world_size 4
#python3 t5_small_v1.py --rank 3 --world_size 4


#install dependences
sudo apt install -y python3-pip
sudo pip install transformers datasets --quiet
sudo pip install torch
sudo pip install "transformers[torch]"
sudo pip install evaluate rouge-score absl-py nltk sacrebleu sentencepiece
sudo pip install protobuf

#id, conversation, language, task, no_of_turns, evolved_user_prompt, output_assistant_reply
#seed_prompt, task_evol_prompt, task_evol_type , evolved_multiturn_prompt,
#multiturn_evol_type, multiturn_evol_prompt

#vim t5_small_v1.py



