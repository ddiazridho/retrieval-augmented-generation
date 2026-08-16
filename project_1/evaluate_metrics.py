import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / 'project_1'

EVAL_FILES = {
    'chunk_overlap': P1 / 'eval_results_chunk_overlap.json',
    'chunk_recursive': P1 / 'eval_results_chunk_recursive.json',
    'chunk_sentencebased': P1 / 'eval_results_chunk_sentencebased.json',
}

K = 3


def metrics_for_records(records: List[dict], k: int) -> Dict[str, float]:
    hits = 0.0
    precisions = 0.0
    mrr_total = 0.0

    for rec in records:
        relevant = set(rec.get('docs_relevant', []))
        retrieved = rec.get('retrieve_docs_id', [])[:k]

        # hit@k
        if any(doc in relevant for doc in retrieved):
            hits += 1.0

        # precision@k (counts duplicates if present)
        correct = sum(1 for doc in retrieved if doc in relevant)
        precisions += correct / float(k)

        # MRR@k
        rr = 0.0
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                rr = 1.0 / rank
                break
        mrr_total += rr

    n = max(1, len(records))
    return {
        f'hit@{k}': hits / n,
        f'precision@{k}': precisions / n,
        f'mrr@{k}': mrr_total / n,
        'count': n,
    }


def main() -> None:
    summary: Dict[str, Dict[str, float]] = {}

    for name, path in EVAL_FILES.items():
        if not path.exists():
            print(f"Warning: missing {path}, skipping {name}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        summary[name] = metrics_for_records(records, K)

    out = {
        'k': K,
        'variants': summary,
    }
    out_path = P1 / 'metrics_summary.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == '__main__':
    main()
