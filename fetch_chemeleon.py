"""Download the pretrained CheMeleon encoder checkpoint."""

from pathlib import Path
from urllib.request import urlretrieve

Path("checkpoints").mkdir(exist_ok=True)
urlretrieve(
    "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
    "checkpoints/chemeleon_mp.pt",
)
print("Saved checkpoints/chemeleon_mp.pt")
