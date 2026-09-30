"""WikiText unsupervised LM dataset builder (HuggingFace; requires ``[text]`` extras).

WikiText (``wikitext-2-raw-v1``) is a corpus of Wikipedia text with no
supervised label.  Its HF rows are individual *lines* of one continuous corpus
rather than independent documents, so the lines are concatenated **without** a
per-line separator token and the stream is split into fixed-length blocks
(*concat-and-chunk*; see
:func:`neural_lofi.datasets.text.utils.tokenize_and_chunk`).  Samples are
returned as ``(block, block)`` (input == target).
"""

from __future__ import annotations

from torch.utils.data import Dataset

from ..config import DatasetConfig
from ..factory import register
from .utils import (
    DEFAULT_MAX_SEQ_LENGTH,
    TokenBlockDataset,
    get_tokenizer,
    tokenize_and_chunk,
)

#: HuggingFace dataset id and config name for the WikiText corpus.
WIKITEXT_REPO = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"


def _hf_split(split: str) -> str:
    """Map a :class:`DatasetConfig` split to a WikiText HF split."""
    if split == "train":
        return "train"
    if split == "val":
        return "validation"
    if split == "test":
        return "test"
    raise ValueError(f"Unsupported split for wikitext: {split!r}")


@register("wikitext")
def build_wikitext(
    config: DatasetConfig,
) -> Dataset:  # pragma: no cover - optional [text] dependency
    """Build the WikiText concat-and-chunk dataset.

    Loads from HuggingFace ``wikitext`` (config ``wikitext-2-raw-v1``), tokenizes
    with GPT-2 BPE, concatenates the lines, and splits the stream into
    non-overlapping blocks of ``config.max_seq_length`` (default 512).  Only the
    first ``config.n_samples`` blocks (corpus order) are materialized.  Requires
    the ``datasets`` and ``transformers`` packages (install via
    ``pip install -e '.[text]'``).
    """
    hf_split = _hf_split(config.split)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for text datasets. "
            "Install it with: pip install -e '.[text]'"
        ) from exc

    hf_ds = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split=hf_split)

    block_length = config.max_seq_length or DEFAULT_MAX_SEQ_LENGTH
    tokenizer = get_tokenizer()
    blocks = tokenize_and_chunk(
        (row["text"] for row in hf_ds),
        block_length,
        tokenizer,
        config.n_samples,
        separator_id=None,
    )
    return TokenBlockDataset(blocks)
