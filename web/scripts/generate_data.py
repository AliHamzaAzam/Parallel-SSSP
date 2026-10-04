"""Convert checked-in pdftotext output to a small, auditable evidence dataset."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
# Reviewed together against the original report. Update only after rechecking all claims.
PDF_SHA256 = 'e796d287ca6340d61b3e0355ba6940ba85fa9beda90c37850c13b53af7204b98'
RAW_SHA256 = 'dc77a9d6fab87e9629aabceb21ecc60faa79ad93fd15caf822c487e102c3075d'

def extract(raw):
    pages = [' '.join(page.split()) for page in raw.split('\f')]
    if len(pages) != 39 or pages[-1]:
        raise ValueError('Expected the 38-page report extracted with pdftotext -layout')
    statements = {
        'dataset': (17, 'The bio-human-gene2 dataset was used as the primary test case for evaluating the performance of various parallel implementations.'),
        'mpi': (36, 'Pure MPI: Speedup increased from 1.19× for 10,000 updates to 6.45× for 100,000 updates'),
        'hybrid': (35, '2 MPI ranks and 8 OpenMP threads per rank, resulting in a peak speedup of 18.03× for 100,000 updates.'),
        'openmp': (21, '12,500 updates, reaching an impressive 39.35× speedup with just 2 threads.'),
    }
    statements['mpi_configuration'] = (20, 'Pure MPI vs Hybrid: Pure MPI (2 ranks)')
    for key, (page, statement) in statements.items():
        if pages[page - 1].count(statement) != 1:
            raise ValueError(f'Missing or ambiguous report evidence: {key} on page {page}')
    return {
        'dataset': 'bio-human-gene2', 'source': 'Project_Report_Parallel_SSSP.pdf',
        'rawSha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        'pdfSha256': PDF_SHA256,
        'measurementKind': 'Reported speedup over serial; not independently reproduced',
        'points': [
            {'mode': 'MPI', 'updates': 10000, 'speedup': 1.19, 'configuration': '2 ranks (report p. 20)', 'page': 36},
            {'mode': 'MPI', 'updates': 100000, 'speedup': 6.45, 'configuration': '2 ranks (report p. 20)', 'page': 36},
            {'mode': 'MPI + OpenMP', 'updates': 100000, 'speedup': 18.03, 'configuration': '2 ranks, 8 threads per rank', 'page': 35},
            {'mode': 'OpenMP', 'updates': 12500, 'speedup': 39.35, 'configuration': '2 threads', 'page': 21},
        ],
        'missing': ['Raw benchmark timings', 'Repetitions and variance', 'Compiler version and flags', 'A consistent processor-count scaling series'],
    }

def main():
    pdf = (ROOT.parent / 'Project_Report_Parallel_SSSP.pdf').read_bytes()
    raw = (ROOT / 'raw/report.txt').read_bytes()
    for label, content, expected in [('PDF', pdf, PDF_SHA256), ('extracted text', raw, RAW_SHA256)]:
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'The {label} differs from the reviewed source. Re-extract and review the claims before updating source hashes.')
    data = extract(raw.decode('utf-8'))
    output = ROOT / 'data/results.json'
    temporary = output.with_suffix('.json.tmp')
    try:
        temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
