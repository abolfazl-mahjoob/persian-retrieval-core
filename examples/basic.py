"""Run with: python examples/basic.py (after pip install -e .)."""
from persian_retrieval import (
    LexicalIndex,
    QueryCase,
    chunk_text,
    evaluate_rankings,
    normalize_persian,
    reciprocal_rank_fusion,
)

documents = {
    "shipping": "ارسال سفارش به تهران یک تا دو روز کاری زمان می‌برد.",
    "returns": "مرجوعی کالای سالم تا هفت روز پس از تحویل امکان‌پذیر است.",
    "hours": "ساعت کاری پشتیبانی از نه تا پنج است.",
}
index = LexicalIndex(documents)
query = "سفارش تهران کی ارسال میشه؟"
lexical = [hit.document_id for hit in index.search(query)]
# An external embedding/vector retrieval system could supply a second ranking.
vector = ["shipping", "hours"]
fused = reciprocal_rank_fusion([lexical, vector])
metrics = evaluate_rankings(
    [QueryCase(query, frozenset({"shipping"}))],
    [[doc_id for doc_id, _ in fused]],
    k=3,
)
print("normalized:", normalize_persian("يک كتاب ۱۲۳"))
print("lexical:", lexical)
print("fused:", fused)
print("recall@3:", metrics.recall_at_k, "mrr@3:", metrics.mrr_at_k)
print("chunks:", chunk_text("# ارسال\n" + documents["shipping"], "راهنما"))
