#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from typing import List, Tuple, Optional

import requests

BASE_URL = 'https://raw.githubusercontent.com/disrpt/sharedtask2025/refs/heads/master/data'
PREFIX = '# newdoc id = '


def download_split(corpus: str, split: str, out_path: str) -> None:
    """Download a .conllu split if it doesn’t already exist."""
    if os.path.isfile(out_path):
        return
    url = f"{BASE_URL}/{corpus}/{corpus}_{split}.conllu"
    resp = requests.get(url)
    resp.raise_for_status()
    with open(out_path, 'w', encoding='utf8') as f:
        f.write(resp.text)


def read_newdoc_ids(path: str) -> List[str]:
    """Return list of all newdoc IDs in a .conllu file."""
    with open(path, 'r', encoding='utf8') as f:
        return [
            line.strip()[len(PREFIX):]
            for line in f
            if line.strip().startswith(PREFIX)
        ]


def find_rs3_docs(rs3_dir: str, corpus: str) -> List[str]:
    """List all .rs3 basenames under rs3_dir/corpus/."""
    pattern = os.path.join(rs3_dir, corpus, '*.rs3')
    return [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(pattern)]


def load_splits(
    corpus: str,
    rs3_dir: Optional[str] = None,
    download: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (train_ids, dev_ids, test_ids) for `corpus`.
    If download=True it will fetch missing .conllu files.
    If rs3_dir is provided, infer test_ids from its <corpus>/*.rs3.
    """
    dev_path = f"{corpus}_dev.conllu"
    train_path = f"{corpus}_train.conllu"

    if download:
        try:
            download_split(corpus, 'dev', dev_path)
            download_split(corpus, 'train', train_path)
        except requests.HTTPError as e:
            raise RuntimeError(f"Failed to download splits: {e}")

    train_ids = read_newdoc_ids(train_path)
    dev_ids = read_newdoc_ids(dev_path)

    test_ids = []
    if rs3_dir:
        all_rs3 = find_rs3_docs(rs3_dir, corpus)
        test_ids = [d for d in all_rs3 if d not in train_ids + dev_ids]

    return train_ids, dev_ids, test_ids


def main():
    parser = argparse.ArgumentParser(
        description="Extract train/dev newdoc IDs and infer test split from RS3 files."
    )
    parser.add_argument('corpus',
                        help="Corpus name (e.g. 'eng.rst.umuc' or 'deu.rst.pcc')")
    parser.add_argument('--rs3-dir',
                        default='data/raw_rs3',
                        help="If set, infer test IDs from <rs3-dir>/<corpus>/*.rs3")
    args = parser.parse_args()

    try:
        train_ids, dev_ids, test_ids = load_splits(
            args.corpus,
            rs3_dir=args.rs3_dir
        )
    except Exception as e:
        sys.exit(str(e))

    print(f"Found {len(train_ids)} train docs.")
    print("\n".join(train_ids))
    print(f"\nFound {len(dev_ids)} dev docs.")
    print("\n".join(dev_ids))
    if test_ids:
        print(f"\nFound {len(test_ids)} test docs.")


if __name__ == '__main__':
    main()
