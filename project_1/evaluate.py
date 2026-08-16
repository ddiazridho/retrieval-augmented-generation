import json
from pathlib import Path
from typing import List
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'project_1'

PATH_GT = ROOT / 'ground_truth.json'

CONFIGS = [
    {
        'name': 'chunk_overlap',
        'path_chunks': DATA_DIR / 'chunk_overlap.json',
        'path_indices': DATA_DIR / 'top_k_indices_chunk_overlap.npy',
        'path_out': DATA_DIR / 'eval_results_chunk_overlap.json',
    },
    {
        'name': 'chunk_recursive',
        'path_chunks': DATA_DIR / 'chunk_recursive.json',
        'path_indices': DATA_DIR / 'top_k_indices_chunk_recursive.npy',
        'path_out': DATA_DIR / 'eval_results_chunk_recursive.json',
    },
    {
        'name': 'chunk_sentencebased',
        'path_chunks': DATA_DIR / 'chunk_sentencebased.json',
        'path_indices': DATA_DIR / 'top_k_indices_chunk_sentencebased.npy',
        'path_out': DATA_DIR / 'eval_results_chunk_sentencebased.json',
    },
]


def load_chunks_doc_map(path_chunks: Path) -> List[str]:
    with open(path_chunks, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    index_to_doc_md: List[str] = []
    for entry in chunks:
        doc_id = entry.get('doc_id')
        index_to_doc_md.append(f"{doc_id}.md")
    return index_to_doc_md


def map_indices_to_docs(indices: np.ndarray, index_to_doc_md: List[str]) -> List[List[str]]:
    results: List[List[str]] = []
    for row in indices:
        docs = [index_to_doc_md[int(idx)] for idx in row]
        results.append(docs)
    return results


def evaluate_variant(path_chunks: Path, path_indices: Path, path_out: Path) -> int:
    indices = np.load(path_indices)

    with open(PATH_GT, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    index_to_doc_md = load_chunks_doc_map(path_chunks)
    docs_pred = map_indices_to_docs(indices, index_to_doc_md)

    n = min(len(ground_truth), len(docs_pred))

    output = []
    for i in range(n):
        item = {
            'query': ground_truth[i]['query'],
            'docs_relevant': ground_truth[i]['docs_relevant'],
            'retrieve_docs_id': docs_pred[i],
        }
        output.append(item)

    with open(path_out, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return n


def main() -> None:
    total = 0
    for cfg in CONFIGS:
        n = evaluate_variant(cfg['path_chunks'], cfg['path_indices'], cfg['path_out'])
        print(f"Saved {n} records to {cfg['path_out']}")
        total += n
    print(f"Done. Variants evaluated: {len(CONFIGS)}. Total records written: {total}")


if __name__ == '__main__':
    main()
