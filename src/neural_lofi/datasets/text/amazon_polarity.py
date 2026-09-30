"""Amazon Polarity sentiment dataset builder (HuggingFace; requires ``[text]`` extras)."""


import torch
from torch.utils.data import Dataset

from ..config import DatasetConfig
from ..factory import register
from .utils import DEFAULT_MAX_SEQ_LENGTH, get_tokenizer, tokenize_and_pad


class _AmazonPolarityDataset(Dataset):
    """Wraps a HuggingFace ``amazon_polarity`` split as a PyTorch Dataset.

    Each sample returns ``(token_ids, label)`` where *token_ids* is a
    ``LongTensor`` of shape ``(max_seq_length,)`` produced by GPT-2 BPE
    tokenization of the concatenated title and content fields.  Self-supervised
    use (``target_mode="input"`` → ``(token_ids, token_ids)``) is handled
    centrally by :func:`neural_lofi.datasets.factory.build_dataset`.
    """

    def __init__(self, hf_dataset, tokenizer, max_seq_length: int) -> None:
        self._hf = hf_dataset
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self._hf[idx]
        text = row["title"] + " " + row["content"]
        token_ids = tokenize_and_pad(text, self._max_seq_length, self._tokenizer)
        return token_ids, row["label"]


@register("amazon-polarity")
def build_amazon_polarity(
    config: DatasetConfig,
) -> Dataset:  # pragma: no cover - optional [text] dependency
    """Build the Amazon Polarity dataset.

    Loads from HuggingFace ``fancyzhx/amazon_polarity``, tokenizes with
    GPT-2 BPE, and pads/truncates to ``config.max_seq_length`` (default
    512).  With ``config.target_mode == "input"`` the label is dropped and each
    sample becomes ``(token_ids, token_ids)`` for self-supervised use.  Requires
    the ``datasets`` and ``transformers`` packages (install via
    ``pip install -e '.[text]'``).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for text datasets. "
            "Install it with: pip install -e '.[text]'"
        ) from exc

    # --- split mapping ---
    if config.split == "train":
        hf_split = "train"
    elif config.split in {"val", "test"}:
        hf_split = "test"
    else:
        raise ValueError(f"Unsupported split for amazon-polarity: {config.split!r}")

    hf_ds = load_dataset("fancyzhx/amazon_polarity", split=hf_split)

    max_seq_length = config.max_seq_length or DEFAULT_MAX_SEQ_LENGTH
    tokenizer = get_tokenizer()

    return _AmazonPolarityDataset(hf_ds, tokenizer, max_seq_length)
