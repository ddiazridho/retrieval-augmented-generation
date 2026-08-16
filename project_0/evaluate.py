import json
from pathlib import Path
from typing import List
import numpy as np


DATA_DIR = Path('project_0')
PATH_MANUAL = DATA_DIR / 'top_indices_manual.npy'
PATH_SKLEARN = DATA_DIR / 'top_indices_sklearn.npy'
PATH_CHUNKS = DATA_DIR / 'chunk.json'
PATH_GT = DATA_DIR / 'ground_truth.json'
PATH_OUT = DATA_DIR / 'eval_results.json'


def load_chunks_doc_map() -> List[str]:
    with open(PATH_CHUNKS, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    index_to_doc_md = []
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


def main() -> None:
    retrieve_manual = np.load(PATH_MANUAL)
    retrieve_sklearn = np.load(PATH_SKLEARN)

    with open(PATH_GT, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    index_to_doc_md = load_chunks_doc_map()

    docs_manual = map_indices_to_docs(retrieve_manual, index_to_doc_md)
    docs_sklearn = map_indices_to_docs(retrieve_sklearn, index_to_doc_md)

    n = min(len(ground_truth), len(docs_manual), len(docs_sklearn))

    output = []
    for i in range(n):
        item = {
            'query': ground_truth[i]['query'],
            'docs_relevant': ground_truth[i]['docs_relevant'],
            'retrieve_docs_id': docs_sklearn[i],
        }
        output.append(item)

    with open(PATH_OUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output)} records to {PATH_OUT}")


if __name__ == '__main__':
    main()
