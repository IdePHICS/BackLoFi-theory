"""GPT-2 BPE tokenization utilities for text datasets (lazy ``transformers`` import)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import Dataset

DEFAULT_MAX_SEQ_LENGTH = 512

# GPT-2 BPE vocabulary size (including special tokens)
GPT2_VOCAB_SIZE = 50_257
# GPT-2 end-of-text token ID, reused as PAD
GPT2_PAD_TOKEN_ID = 50_256


def get_tokenizer():  # pragma: no cover - optional [text] dependency
    """Return the GPT-2 BPE tokenizer (lazy import).

    Sets ``pad_token`` to ``eos_token`` so the tokenizer can pad
    sequences.  Requires the ``transformers`` package.
    """
    try:
        from transformers import GPT2TokenizerFast
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for text datasets. "
            "Install it with: pip install -e '.[text]'"
        ) from exc

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_and_pad(
    text: str,
    max_seq_length: int,
    tokenizer,
) -> torch.Tensor:
    """Tokenize *text* and return a fixed-length ``LongTensor``.

    Parameters
    ----------
    text : str
        Raw input string.
    max_seq_length : int
        Target sequence length (truncate or pad).
    tokenizer
        A HuggingFace tokenizer instance (e.g. ``GPT2TokenizerFast``).

    Returns
    -------
    torch.Tensor
        ``LongTensor`` of shape ``(max_seq_length,)``.
    """
    encoded = tokenizer(
        text,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return encoded["input_ids"].squeeze(0)


def tokenize_and_chunk(
    documents: Iterable[str],
    block_length: int,
    tokenizer,
    n_blocks: int,
    *,
    separator_id: int | None = None,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Stream-tokenize *documents* and pack them into fixed-length token blocks.

    Implements the *concat-and-chunk* scheme used by the unsupervised
    language-modeling datasets: every document is tokenized (no padding, no
    truncation), the token streams are concatenated, and the result is split
    into **non-overlapping** blocks of ``block_length`` tokens.  Any trailing
    remainder shorter than ``block_length`` is dropped.

    Tokenization is streamed in batches so that only as much of the corpus as
    needed is processed: the iterator is consumed until ``n_blocks`` full blocks
    have been collected (or the corpus is exhausted).  Blocks therefore come from
    the **start** of *documents* in corpus order.

    Parameters
    ----------
    documents : Iterable[str]
        Raw document (or line) strings, e.g. ``(row["text"] for row in hf_ds)``.
    block_length : int
        Number of tokens per block.
    tokenizer
        A HuggingFace tokenizer supporting batch calls that return a list of
        token-id lists, e.g. ``tokenizer(batch, add_special_tokens=False)``.
    n_blocks : int
        Maximum number of blocks to collect (an upper bound on the rows of the
        returned tensor).
    separator_id : int | None
        Token id appended after every document (e.g. the GPT-2 end-of-text id)
        to mark document boundaries.  ``None`` concatenates documents with no
        separator — appropriate when the corpus rows are *lines* of one
        continuous text rather than independent documents.
    batch_size : int
        Number of documents tokenized per batch call.

    Returns
    -------
    torch.Tensor
        ``LongTensor`` of shape ``(M, block_length)`` with ``M <= n_blocks``.
    """
    blocks: list[list[int]] = []
    buffer: list[int] = []

    def _drain_buffer() -> bool:
        """Carve full blocks out of *buffer*; return True once *n_blocks* reached."""
        nonlocal buffer
        while len(buffer) >= block_length:
            blocks.append(buffer[:block_length])
            buffer = buffer[block_length:]
            if len(blocks) >= n_blocks:
                return True
        return False

    doc_iter = iter(documents)
    exhausted = False
    while not exhausted and len(blocks) < n_blocks:
        batch: list[str] = []
        for _ in range(batch_size):
            try:
                batch.append(next(doc_iter))
            except StopIteration:
                exhausted = True
                break
        if not batch:
            break

        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            buffer.extend(ids)
            if separator_id is not None:
                buffer.append(separator_id)
        if _drain_buffer():
            break

    if not blocks:
        return torch.empty((0, block_length), dtype=torch.long)
    return torch.tensor(blocks, dtype=torch.long)


class TokenBlockDataset(Dataset):
    """Holds a ``(M, block_length)`` block of token ids as an unsupervised set.

    Each sample is returned as ``(block, block)`` — the input token ids and an
    identical copy in the target slot.  The duplicated target keeps the standard
    ``(input, target)`` contract so the block flows through ``build_dataset``
    unchanged; downstream code may ignore the target (pure representation
    learning) or derive a next-token shift from it.
    """

    def __init__(self, blocks: torch.Tensor) -> None:
        self._blocks = blocks

    def __len__(self) -> int:
        return self._blocks.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        block = self._blocks[idx]
        return block, block
