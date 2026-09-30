"""TinyStories unsupervised LM dataset (HuggingFace; requires ``[text]`` extras).

TinyStories (``roneneldan/TinyStories``) is a corpus of short synthetic stories
with no supervised label.  Each row is one full story, so stories are joined by
the GPT-2 end-of-text token and the resulting stream is split into fixed-length
blocks (*concat-and-chunk*; see
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

#: HuggingFace repository id for the TinyStories corpus.
TINYSTORIES_REPO = "roneneldan/TinyStories"


def _hf_split(split: str) -> str:
    """Map a :class:`DatasetConfig` split to a TinyStories HF split.

    TinyStories ships only ``train`` and ``validation`` splits, so both ``val``
    and ``test`` map onto ``validation``.
    """
    if split == "train":
        return "train"
    if split in {"val", "test"}:
        return "validation"
    raise ValueError(f"Unsupported split for tiny-stories: {split!r}")


@register("tiny-stories")
def build_tiny_stories(
    config: DatasetConfig,
) -> Dataset:  # pragma: no cover - optional [text] dependency
    """Build the TinyStories concat-and-chunk dataset.

    Loads from HuggingFace ``roneneldan/TinyStories``, tokenizes with GPT-2 BPE,
    joins stories with the end-of-text token, and splits the stream into
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

    hf_ds = load_dataset(TINYSTORIES_REPO, split=hf_split)

    block_length = config.max_seq_length or DEFAULT_MAX_SEQ_LENGTH
    tokenizer = get_tokenizer()
    blocks = tokenize_and_chunk(
        (row["text"] for row in hf_ds),
        block_length,
        tokenizer,
        config.n_samples,
        separator_id=tokenizer.eos_token_id,
    )
    return TokenBlockDataset(blocks)
