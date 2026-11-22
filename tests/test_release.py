"""Check the public result families, reference assets and manuscript links."""
import ast
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import unittest

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_original_publication_assets(self):
        manifest = json.loads((ROOT / 'config/publication_manifest.json').read_text())
        for filename, expected in manifest['files'].items():
            self.assertEqual(sha256((ROOT / filename).read_bytes()).hexdigest(), expected, filename)
        self.assertEqual(len(list((ROOT / 'docs/figures').glob('*.pdf'))), 6)

    def test_statistical_families(self):
        enc = pd.read_csv(ROOT / 'results/encoding_all_assessment_tests.tsv', sep='\t')
        glm = pd.read_csv(ROOT / 'results/text_glm_all_tests.tsv', sep='\t')
        self.assertEqual((len(enc), len(glm)), (5104, 7704))
        self.assertEqual(set(enc.model), {'gemma3_1b'})
        self.assertEqual(set(enc.variant), {'pc128'})
        self.assertEqual(set(enc.top_pct), {25})
        self.assertEqual(set(glm.model), {'gemma3_1b'})
        self.assertEqual(set(glm.atlas), {'schaefer200_tian_s2'})
        self.assertEqual(set(glm.top_pct), {25})
        self.assertEqual(set(enc.n_subjects), {16})
        self.assertEqual(set(glm.n_subjects), {16})
        self.assertEqual(int((enc.q_bh_232_rois <= .05).sum()), 2)
        self.assertEqual(int((glm.q_bh <= .05).sum()), 33)
        for _, family in enc.groupby(['metric', 'assessment']):
            self.assertEqual(family.roi_id.nunique(), 232)
            np.testing.assert_allclose(false_discovery_control(family.p_exact), family.q_bh_232_rois, atol=5e-14, rtol=1e-12)
        for _, family in glm.groupby(['text_feature', 'contrast']):
            self.assertEqual(family.roi_id.nunique(), 214)
            np.testing.assert_allclose(false_discovery_control(family.p_permutation), family.q_bh, atol=5e-14, rtol=1e-12)
        performance = pd.read_csv(ROOT / 'results/encoding_roi_performance_all.tsv', sep='\t')
        self.assertEqual(performance.roi_id.nunique(), 232)

    def test_atlas(self):
        image = nib.load(ROOT / 'assets/atlas/labels.nii.gz')
        labels = np.asarray(image.dataobj)
        np.testing.assert_array_equal(np.unique(labels), np.arange(233))
        self.assertEqual(nib.aff2axcodes(image.affine), ('R', 'A', 'S'))
        with (ROOT / 'assets/atlas/lut.tsv').open() as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        self.assertEqual([int(r['roi_id']) for r in rows], list(range(1, 233)))
        self.assertEqual(sum(r['component'] == 'Tian S2' for r in rows), 32)
        self.assertEqual(sum(r['component'] == 'Schaefer 200' for r in rows), 200)

    def test_python_syntax(self):
        for p in (ROOT / 'code').rglob('*.py'):
            ast.parse(p.read_text(), filename=str(p.relative_to(ROOT)))

    def test_readme_local_links(self):
        for link in re.findall(r'\]\(([^)]+)\)', (ROOT / 'README.md').read_text()):
            if link.startswith(('https://', 'http://', 'mailto:', '#')):
                continue
            self.assertTrue((ROOT / link.split('#')[0]).exists(), link)


if __name__ == '__main__':
    unittest.main()
