"""Functional PIM chunk stream; values are independent of bank placement."""
from poppim.scoring.reference import score_encoded


def score_chunks(q, k, head_to_kv=None, chunk_size=1024):
    for start in range(0, k.sign.shape[1], chunk_size):
        yield start, score_encoded(q, k.slice_tokens(start, start+chunk_size),
                                   head_to_kv, chunk_size)
