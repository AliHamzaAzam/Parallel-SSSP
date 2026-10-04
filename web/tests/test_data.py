import importlib.util
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generate_data', ROOT / 'scripts/generate_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.raw = (ROOT / 'raw/report.txt').read_text()
    def test_generated_file_matches_source(self):
        self.assertEqual(module.extract(self.raw), json.loads((ROOT / 'data/results.json').read_text()))
    def test_exact_values_and_workloads(self):
        rows = module.extract(self.raw)['points']
        self.assertEqual([(r['mode'], r['updates'], r['speedup']) for r in rows], [('MPI', 10000, 1.19), ('MPI', 100000, 6.45), ('MPI + OpenMP', 100000, 18.03), ('OpenMP', 12500, 39.35)])
        self.assertTrue(all(r['page'] > 0 for r in rows))
    def test_missing_evidence_is_rejected(self):
        for value in ['1.19×', '18.03×', '39.35×']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.extract(self.raw.replace(value, 'unknown'))
    def test_duplicate_report_is_rejected(self):
        with self.assertRaises(ValueError):
            module.extract(self.raw + self.raw)
    def test_missing_mpi_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            module.extract(self.raw.replace('Pure MPI (2 ranks)', 'Pure MPI (unknown ranks)'))
    def test_misplaced_source_page_is_rejected(self):
        pages = self.raw.split('\f')
        pages[20], pages[21] = pages[21], pages[20]
        with self.assertRaises(ValueError):
            module.extract('\f'.join(pages))
    def test_layout_whitespace_is_accepted(self):
        self.assertEqual(module.extract(self.raw)['points'], module.extract('\f'.join(' '.join(page.split()) for page in self.raw.split('\f')))['points'])
    def test_no_invented_runtime_or_efficiency(self):
        self.assertTrue(all('runtime' not in p and 'efficiency' not in p for p in module.extract(self.raw)['points']))
if __name__ == '__main__':
    unittest.main()
