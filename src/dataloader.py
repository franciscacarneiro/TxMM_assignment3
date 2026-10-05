from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "A3data_2627_pan2020"
FILES = {
    "train": "pan2627_train_data.csv",
    "dev": "pan2627_dev_data.csv",
    "test": "pan2627_test_data.csv",
}


def load_split(name, data_dir=DATA_DIR):
    df = pd.read_csv(Path(data_dir) / FILES[name], encoding="utf-8-sig", index_col=0)
    return df["text"].tolist(), df["author"].tolist()


def load_all(data_dir=DATA_DIR):
    return {name: load_split(name, data_dir) for name in FILES}
