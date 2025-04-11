from pathlib import Path
from typing import List


def get_checkpoint_paths(path: Path):
    all_checkpoint_paths = []

    if not path.is_absolute():
        path = Path.cwd() / path
    else:
        path = path

    if path.is_dir():
        all_checkpoint_paths = all_checkpoint_paths + [file.resolve() for file in Path(path).rglob("*.ckpt")]

    elif path.is_file() and path.suffix == '.ckpt':
        print(f"Checkpoint file: {path}")
        all_checkpoint_paths.append(path)
    else:
        print(f"Warning: {path} is neither a directory containing .ckpt files nor a valid .ckpt file.")

    print("\nFinal list of checkpoints used:")
    for checkpoint in all_checkpoint_paths:
        print(f"  - {checkpoint}")

    return all_checkpoint_paths