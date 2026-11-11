"""Download + extract the Khmer ASR Cultural Dataset from Mozilla Data Collective."""
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from ..utils import log

DATASET_ID = "cmkcy8in2004umo0775mye43g"  # https://mozilladatacollective.com/datasets/cmkcy8in2004umo0775mye43g


def extract(archive, out_dir):
    """Extract a .tar.gz, using pigz when available (faster), else Python's tarfile."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("pigz") and shutil.which("tar"):
        subprocess.run(["tar", "-I", "pigz", "-xf", str(archive), "-C", str(out_dir)], check=True)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            try:
                tf.extractall(out_dir, filter="data")
            except TypeError:  # Python < 3.12
                tf.extractall(out_dir)
    return out_dir


def download(args):
    """Requires `pip install datacollective`, an API key in $MDC_API_KEY, and the dataset terms
    accepted on the dataset page with the same account."""
    archive = args.archive
    if not archive:
        if not os.environ.get("MDC_API_KEY"):
            raise SystemExit("Set MDC_API_KEY to your Mozilla Data Collective API key "
                             "(Profile -> API / Credentials), e.g.  export MDC_API_KEY=...")
        try:
            from datacollective import download_dataset
        except ImportError:
            raise SystemExit("pip install datacollective   (or: pip install 'khmer-vits-tts[download]')")
        log(f"downloading dataset {args.dataset_id} (~12.6 GB; resumes if interrupted) ...")
        try:
            archive = download_dataset(args.dataset_id, download_directory=str(args.cache_dir), show_progress=True)
        except PermissionError as e:
            raise SystemExit("Access denied: accept the dataset terms on mozilladatacollective.com "
                             f"with the same account as your API key. ({e})")
    log(f"extracting {archive} -> {args.out}")
    extract(archive, args.out)
    meta = sorted(Path(args.out).rglob("metadata.csv"))
    log(f"done. metadata: {meta[0] if meta else 'NOT FOUND'}")
    log(f"next: khmer-vits prepare --src {args.out} --out data/prep")
