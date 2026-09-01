"""Evaluate Smart Doc retrieval and answer coverage against labelled JSONL cases.

The evaluator intentionally reuses the production extraction, summarisation,
indexing, retrieval, and answer-chain code. It does not require RAGAS or an
LLM judge, but the project itself still needs its normal model credentials to
build summaries and generate answers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FIELDS = {"source_id", "source_file", "page_number", "modality", "chunk_index"}

# Running ``python evaluation/evaluate_rag.py`` places ``evaluation/`` rather
# than the repository root on sys.path. Add the root so the production ``rag``
# and ingestion modules can be imported without requiring installation first.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class EvaluationError(ValueError):
    """Raised when an evaluation case cannot be run safely."""


def parse_top_k(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--top-k must be comma-separated positive integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("--top-k must contain positive integers")
    return values


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and validate one JSON object per non-empty line."""
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(f"{path}:{line_number} is not valid JSON: {error.msg}") from error
            validate_case(case, path, line_number)
            cases.append(case)

    if not cases:
        raise EvaluationError(f"{path} contains no evaluation cases")
    return cases


def validate_case(case: dict[str, Any], path: Path, line_number: int) -> None:
    required_strings = ("id", "document", "question")
    missing = [field for field in required_strings if not isinstance(case.get(field), str) or not case[field].strip()]
    if missing:
        raise EvaluationError(f"{path}:{line_number} is missing non-empty field(s): {', '.join(missing)}")

    expected_sources = case.get("expected_sources")
    if not isinstance(expected_sources, list) or not expected_sources:
        raise EvaluationError(f"{path}:{line_number} needs at least one expected_sources entry")
    for source in expected_sources:
        if not isinstance(source, dict) or not (SOURCE_FIELDS & set(source)):
            raise EvaluationError(
                f"{path}:{line_number} expected_sources entries need one of: {', '.join(sorted(SOURCE_FIELDS))}"
            )

    required_terms = case.get("required_terms", [])
    if not isinstance(required_terms, list) or not all(isinstance(term, str) for term in required_terms):
        raise EvaluationError(f"{path}:{line_number} required_terms must be a list of strings")


def resolve_document(document: str) -> Path:
    document_path = Path(document)
    return document_path if document_path.is_absolute() else PROJECT_ROOT / document_path


def _split_pdf_elements(chunks):
    texts, tables, images = [], [], []
    for chunk in chunks:
        if type(chunk).__name__ != "CompositeElement":
            continue
        texts.append(chunk)
        for element in getattr(chunk.metadata, "orig_elements", []):
            element_type = type(element).__name__
            if element_type == "Image":
                image_base64 = getattr(element.metadata, "image_base64", None)
                if image_base64:
                    images.append(image_base64)
            elif element_type == "Table":
                tables.append(element)
    return texts, tables, images


def build_retriever(document: Path, source_label: str):
    """Index one PDF in a unique in-memory collection for an evaluation run."""
    # Keep expensive and optional imports out of module import time so --help
    # and --validate-only work in a minimal Python environment.
    from extraction.extract_pdf import extract_pdf_elements
    from rag.retrieval import setup_retriever, store_documents
    from rag.vector_store import get_vectorstore
    from summarization.summarize_image import summarize_images
    from summarization.summarize_text_table import summarize_tables, summarize_texts

    chunks = extract_pdf_elements(str(document))
    texts, tables, images = _split_pdf_elements(chunks)

    text_summaries = summarize_texts([element.text for element in texts])
    table_summaries = summarize_tables([element.metadata.text_as_html for element in tables])
    image_summaries = summarize_images(images)

    collection_seed = sha256(str(document.resolve()).encode("utf-8")).hexdigest()[:10]
    collection_name = f"rag_eval_{collection_seed}_{uuid.uuid4().hex[:8]}"
    retriever = setup_retriever(get_vectorstore(collection_name=collection_name))
    store_documents(retriever, texts, text_summaries, "doc_id", source_file=source_label, modality="text")
    store_documents(retriever, tables, table_summaries, "doc_id", source_file=source_label, modality="table")
    store_documents(retriever, images, image_summaries, "doc_id", source_file=source_label, modality="image")
    return retriever


def source_matches(metadata: dict[str, Any], expected_source: dict[str, Any]) -> bool:
    """Return whether one retrieved document matches all supplied source labels."""
    return all(str(metadata.get(field)) == str(value) for field, value in expected_source.items())


def score_retrieval(retrieved_documents, expected_sources, top_k: tuple[int, ...]) -> dict[str, dict[str, float | bool]]:
    """Calculate per-case hit, recall, and precision for each requested k."""
    scores = {}
    for k in top_k:
        selected = retrieved_documents[:k]
        matched_expected = {
            expected_index
            for expected_index, expected_source in enumerate(expected_sources)
            if any(source_matches(document.metadata, expected_source) for document in selected)
        }
        relevant_retrieved = sum(
            any(source_matches(document.metadata, expected_source) for expected_source in expected_sources)
            for document in selected
        )
        scores[str(k)] = {
            "hit": bool(matched_expected),
            "recall": len(matched_expected) / len(expected_sources),
            "precision": relevant_retrieved / k,
        }
    return scores


def required_term_coverage(answer: str, required_terms: list[str]) -> float | None:
    """Return a transparent answer proxy; it is not a correctness score."""
    if not required_terms:
        return None
    normalised_answer = answer.casefold()
    return sum(term.casefold() in normalised_answer for term in required_terms) / len(required_terms)


def serialise_sources(retrieval_results) -> list[dict[str, Any]]:
    serialised = []
    for rank, (document, score) in enumerate(retrieval_results, start=1):
        serialised.append(
            {
                "rank": rank,
                # Chroma returns a raw distance here: lower values indicate a
                # closer match. Do not call it a relevance score or convert it
                # to 0-1, because its distance range depends on the index.
                "distance": float(score) if score is not None else None,
                "metadata": document.metadata,
                "summary_preview": document.page_content[:300],
            }
        )
    return serialised


def evaluate_cases(cases, top_k: tuple[int, ...], answer_k: int, include_answers: bool):
    """Run labelled cases, indexing each document only once per invocation."""
    from rag.rag_chain import get_rag_chain

    grouped_cases = defaultdict(list)
    for case in cases:
        grouped_cases[case["document"]].append(case)

    records = []
    for source_label, document_cases in grouped_cases.items():
        document = resolve_document(source_label)
        if not document.is_file():
            raise EvaluationError(f"PDF does not exist: {document}")

        print(f"Indexing {source_label} for {len(document_cases)} case(s)...", file=sys.stderr)
        ingest_started = time.perf_counter()
        retriever = build_retriever(document, source_label)
        ingestion_seconds = time.perf_counter() - ingest_started

        for case in document_cases:
            started = time.perf_counter()
            # Chroma's relevance-score helper warns when its score conversion
            # falls outside 0-1. Raw distances preserve the exact ranking used
            # for Hit@k/Recall@k/Precision@k without that misleading warning.
            retrieval_results = retriever.vectorstore.similarity_search_with_score(
                case["question"], k=max(top_k)
            )
            retrieval_seconds = time.perf_counter() - started
            retrieved_documents = [document for document, _ in retrieval_results]

            answer = None
            answer_seconds = None
            if include_answers:
                answer_started = time.perf_counter()
                answer = get_rag_chain(retriever, k=answer_k).invoke({"question": case["question"]})
                answer_seconds = time.perf_counter() - answer_started

            records.append(
                {
                    "id": case["id"],
                    "document": source_label,
                    "question": case["question"],
                    "expected_sources": case["expected_sources"],
                    "retrieval": score_retrieval(retrieved_documents, case["expected_sources"], top_k),
                    "retrieved_sources": serialise_sources(retrieval_results),
                    "reference_answer": case.get("reference_answer"),
                    "required_terms": case.get("required_terms", []),
                    "answer": answer,
                    "answer_required_term_coverage": (
                        required_term_coverage(answer, case.get("required_terms", [])) if answer is not None else None
                    ),
                    "latency_seconds": {
                        "document_ingestion": ingestion_seconds,
                        "retrieval": retrieval_seconds,
                        "answer": answer_seconds,
                    },
                }
            )
    return records


def aggregate_metrics(records, top_k: tuple[int, ...]) -> dict[str, Any]:
    aggregates = {"case_count": len(records), "retrieval": {}}
    for k in top_k:
        key = str(k)
        aggregates["retrieval"][f"@{k}"] = {
            "hit_rate": mean(record["retrieval"][key]["hit"] for record in records),
            "recall": mean(record["retrieval"][key]["recall"] for record in records),
            "precision": mean(record["retrieval"][key]["precision"] for record in records),
        }

    answer_coverages = [record["answer_required_term_coverage"] for record in records]
    answer_coverages = [coverage for coverage in answer_coverages if coverage is not None]
    if answer_coverages:
        aggregates["answer_required_term_coverage"] = mean(answer_coverages)
    return aggregates


def print_summary(aggregates: dict[str, Any]) -> None:
    print(f"Cases: {aggregates['case_count']}")
    for k, values in aggregates["retrieval"].items():
        print(
            f"Hit{k}: {values['hit_rate']:.1%} | "
            f"Recall{k}: {values['recall']:.1%} | "
            f"Precision{k}: {values['precision']:.1%}"
        )
    if "answer_required_term_coverage" in aggregates:
        print(f"Required-term coverage: {aggregates['answer_required_term_coverage']:.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="Path to a labelled JSONL case file")
    parser.add_argument("--top-k", type=parse_top_k, default=(1, 3, 5), help="Comma-separated retrieval depths")
    parser.add_argument("--answer-k", type=int, default=5, help="Number of sources provided to the answer chain")
    parser.add_argument("--skip-answers", action="store_true", help="Evaluate retrieval only")
    parser.add_argument("--output", type=Path, help="Write detailed JSON results to this path")
    parser.add_argument("--validate-only", action="store_true", help="Validate JSONL schema without indexing PDFs")
    args = parser.parse_args()

    if args.answer_k <= 0:
        parser.error("--answer-k must be positive")

    try:
        cases = load_cases(args.cases)
        if args.validate_only:
            print(f"Validated {len(cases)} case(s): {args.cases}")
            return 0

        records = evaluate_cases(cases, args.top_k, args.answer_k, not args.skip_answers)
        aggregates = aggregate_metrics(records, args.top_k)
    except EvaluationError as error:
        parser.error(str(error))

    print_summary(aggregates)
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"aggregate": aggregates, "cases": records}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Detailed results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
