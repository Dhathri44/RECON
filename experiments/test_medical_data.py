import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.medical_data import find_brats_pairs, load_nifti_pair
import matplotlib.pyplot as plt

ROOT="data/medical_images"
pairs=find_brats_pairs(ROOT)
if not pairs: raise SystemExit(f"No matching *_flair and *_seg files found under {ROOT}")
print("Matched patients:",len(pairs))
image,mask,index=load_nifti_pair(*pairs[0])
print("Selected axial slice:",index,"image:",image.shape,"mask:",mask.shape,
      "labels:",set(mask.ravel().tolist()))
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(image,cmap="gray"); plt.title("T2-FLAIR"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(mask,cmap="tab10"); plt.title("Segmentation annotation"); plt.axis("off")
plt.tight_layout(); plt.show()
