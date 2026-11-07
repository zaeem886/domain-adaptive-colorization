# ==============================================================================
# Cell 1: Environment Setup & Installations (Run Once)
# ==============================================================================
# In Kaggle Notebook settings (right sidebar):
# 1. Turn Accelerator to GPU T4 x2 or P100.
# 2. Turn Internet to ON.

!pip install -q opencv-python segment-anything timm transformers einops
!pip install -q ddcolor || echo "ddcolor install checked"

# Download pretrained SAM weights if not already present
import os
import urllib.request

SAM_CHECKPOINT = "sam_vit_b.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if not os.path.exists(SAM_CHECKPOINT):
    print("Downloading SAM ViT-B checkpoint...")
    urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT)
    print("Downloaded SAM checkpoint successfully!")
