import argparse
import csv
import hashlib
import io
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image


def prepare_data(data_root, task_root):
    data_root.mkdir(parents=True, exist_ok=True)
    archives = data_root / 'archives'
    archives.mkdir(exist_ok=True)
    for name in ['horse-or-human', 'validation-horse-or-human']:
        archive = archives / f'{name}.zip'
        if not archive.exists():
            url = f'https://storage.googleapis.com/learning-datasets/{name}.zip'
            urllib.request.urlretrieve(url, archive)
        target = data_root / name
        if not target.exists():
            with zipfile.ZipFile(archive) as source:
                source.extractall(target)

    happy_archive = data_root / 'data.zip'
    train_archive = data_root / 'train.zip'
    if not happy_archive.exists() or not train_archive.exists():
        supplied = next((task_root / 'lab7').glob('Datasets-*.zip'))
        with zipfile.ZipFile(supplied) as source:
            if not happy_archive.exists():
                with source.open('Datasets/HappyVsSad/data.zip') as src, happy_archive.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            outer = data_root / 'dogs-vs-cats.zip'
            if not outer.exists():
                with source.open('Datasets/dogs-vs-cats.zip') as src, outer.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        if not train_archive.exists():
            with zipfile.ZipFile(outer) as source:
                with source.open('train.zip') as src, train_archive.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    if not (data_root / 'happy-sad').exists():
        with zipfile.ZipFile(happy_archive) as source:
            source.extractall(data_root / 'happy-sad')

    manifest = Path(__file__).parent / 'split.csv'
    if manifest.exists():
        with manifest.open() as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = []
        seen = set()
        rng = random.Random(42)
        # Split once, excluding byte-identical files before assigning a subset.
        with zipfile.ZipFile(train_archive) as source:
            for label in ['cat', 'dog']:
                names = sorted(n for n in source.namelist() if n.startswith(f'train/{label}.'))
                rng.shuffle(names)
                selected = []
                for name in names:
                    raw = source.read(name)
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest in seen:
                        continue
                    try:
                        with Image.open(io.BytesIO(raw)) as image:
                            image.verify()
                    except (OSError, ValueError):
                        continue
                    seen.add(digest)
                    selected.append((name, digest))
                    if len(selected) == 2000:
                        break
                for i, (name, digest) in enumerate(selected):
                    subset = 'train' if i < 1000 else 'validation' if i < 1500 else 'test'
                    rows.append({'subset': subset, 'class': label, 'source': name, 'sha256': digest})
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['subset', 'class', 'source', 'sha256'])
            writer.writeheader()
            writer.writerows(rows)

    with zipfile.ZipFile(train_archive) as source:
        for row in rows:
            target = data_root / 'cats-dogs' / row['subset'] / row['class'] / Path(row['source']).name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(row['source']))
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != row['sha256']:
                raise ValueError(f'Image changed: {target}')
    for subset in ['train', 'validation', 'test']:
        print(subset, {label: sum(r['subset'] == subset and r['class'] == label for r in rows)
                       for label in ['cat', 'dog']})
    return data_root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[2] / '.lab-data')
    parser.add_argument('--task-root', type=Path, default=Path(__file__).resolve().parents[2] / 'tasks')
    args = parser.parse_args()
    prepare_data(args.data_root.resolve(), args.task_root.resolve())
