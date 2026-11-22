"""Recover the paper's Schaefer 200 + Tian S2 atlas from verified upstream files.

The packaged atlas is ready to use. This script reproduces the label preparation
using canonical RAS reorientation only, with no spatial interpolation.
"""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

import nibabel as nib
import numpy as np


def main():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path.home() / '.cache/narrative-paper/atlas-sources')
    p.add_argument('--output', type=Path, default=repo / 'outputs/atlas')
    args = p.parse_args()
    metadata = json.loads((repo / 'assets/atlas/provenance.json').read_text())
    args.cache.mkdir(parents=True, exist_ok=True)
    files = []
    for record in metadata['source_files']:
        path = args.cache / record['filename']
        if not path.exists():
            with urlopen(record['url'], timeout=120) as response:
                content = response.read()
            if sha256(content).hexdigest() != record['sha256']:
                raise ValueError(f"Upstream checksum changed: {path.name}")
            path.write_bytes(content)
        if sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f"Checksum mismatch: {path}")
        files.append(path)
    combined, lut, schaefer, tian, tian_lut = files
    reference = nib.load(repo / 'assets/atlas/labels.nii.gz')
    arrays = []
    for path in (combined, schaefer, tian):
        image = nib.as_closest_canonical(nib.load(path))
        if image.shape != reference.shape or not np.array_equal(image.affine, reference.affine):
            raise ValueError(f"Atlas grid mismatch: {path}")
        arrays.append(np.asarray(image.dataobj, dtype=np.int16))
    labels, cortex, deep = arrays
    np.testing.assert_array_equal(np.unique(labels), np.arange(233))
    np.testing.assert_array_equal(labels[labels > 32] - 32, cortex[labels > 32])
    mask = (labels > 0) & (labels <= 32)
    np.testing.assert_array_equal(labels[mask], deep[mask])
    np.testing.assert_array_equal(labels, np.asarray(reference.dataobj))
    lines = [line.strip() for line in lut.read_text().splitlines() if line.strip()]
    names = {int(info.split()[0]): name for name, info in zip(lines[::2], lines[1::2], strict=True)}
    assert list(names) == list(range(1, 233))
    assert [names[i] for i in range(1, 33)] == tian_lut.read_text().split()
    args.output.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(labels, reference.affine, reference.header.copy())
    image.set_data_dtype(np.int16)
    image.set_qform(reference.affine, int(reference.header['qform_code']))
    image.set_sform(reference.affine, int(reference.header['sform_code']))
    nib.save(image, args.output / 'labels.nii.gz')
    with (args.output / 'lut.tsv').open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['roi_id', 'roi', 'component', 'source_roi_id'])
        for i, name in names.items():
            w.writerow([i, name, 'Tian S2' if i <= 32 else 'Schaefer 200', i if i <= 32 else i - 32])
    print(f"Verified all 232 labels and wrote {args.output}")


if __name__ == '__main__':
    main()
