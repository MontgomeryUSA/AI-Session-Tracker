from pathlib import Path
import os


def setup_hf_cache(hf_home_path: str, offline: bool = False) -> Path:
    hf_home = Path(hf_home_path).expanduser().resolve()
    hf_home.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home)
    os.environ["TORCH_HOME"] = str(hf_home / "torch")

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    return hf_home