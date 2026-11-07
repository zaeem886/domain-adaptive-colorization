# ==============================================================================
# Domain-Adaptive Colorization: Kaggle Notebook Cells
#
# IMPORTANT SETTINGS in Kaggle (right panel):
# 1. Accelerator: GPU T4 x2 (or P100)
# 2. Internet: ON (required for pip install & model weights)
# ==============================================================================

# ------------------------------------------------------------------------------
# Cell 1: Clone Repository & CD into Directory
# ------------------------------------------------------------------------------
import os
if not os.path.exists("colorize"):
    !git clone -b B https://github.com/zaeem886/domain-adaptive-colorization.git
    %cd domain-adaptive-colorization
else:
    print("Already inside workspace:", os.getcwd())
    !git pull origin B

# ------------------------------------------------------------------------------
# Cell 2: Install Dependencies
# Note: --no-build-isolation is required for basicsr in modern Python (PEP 517)
# ------------------------------------------------------------------------------
!pip install -q opencv-python segment-anything timm transformers einops
!pip install -q basicsr --no-build-isolation || echo "basicsr wheel build skipped (only needed for full DDColor training)"

# ------------------------------------------------------------------------------
# Cell 3: Run Unit Tests (Person A & Person B Handover)
# ------------------------------------------------------------------------------
!PYTHONPATH=. python tests/run_tests.py

# ------------------------------------------------------------------------------
# Cell 4: Run Interactive Pipeline Simulation (Manual Test)
# ------------------------------------------------------------------------------
!python manual_test.py pics/dog.jpeg

# ------------------------------------------------------------------------------
# Cell 5: Display 4-Panel Visual Comparison
# ------------------------------------------------------------------------------
from IPython.display import Image, display
display(Image(filename="manual_test_output/05_comparison_side_by_side.png"))

# ------------------------------------------------------------------------------
# Cell 6: Download SAM Weights for Grounded-SAM
# ------------------------------------------------------------------------------
import os
import urllib.request

SAM_CHECKPOINT = "sam_vit_b.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if not os.path.exists(SAM_CHECKPOINT):
    print("Downloading SAM ViT-B checkpoint (~375MB)...")
    urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT)
    print("SAM checkpoint downloaded!")
else:
    print("SAM checkpoint already exists.")

# ------------------------------------------------------------------------------
# Cell 7: Run Real Grounded-SAM (Person A) on GPU
# ------------------------------------------------------------------------------
import torch
from colorize.config import DecompConfig
from colorize.decomposition import GroundedSAM, load_gray

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

decomp_cfg = DecompConfig(sam_checkpoint=SAM_CHECKPOINT)
grounded_sam = GroundedSAM(cfg=decomp_cfg, device=device)

gray_img = load_gray("pics/dog.jpeg", max_side=600)
decomp = grounded_sam(gray_img)

print(f"[Person A] Routed {len(decomp.regions)} regions.")
print(f"[Person A] Domain coverage: {decomp.coverage * 100:.1f}%")
print(f"[Person A] Domain breakdown: {decomp.domain_share}")
