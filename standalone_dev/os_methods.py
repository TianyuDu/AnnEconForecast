import os


def create_record(base: str) -> dict:
    """
    Create the necessary directory structure for an experiment record.
    """
    path_col = {
        "base": base,  # Parent directory
        "tb": base + "/tensorboard",  # Tensorboard Files
        "fig": base + "/fig",  # Figures
        "outputs": base + "/outputs"  # Numericals, datasets
    }
    for p in path_col.values():
        print(f"Creating directory: {p}")
        os.mkdir(p)
    return path_col
