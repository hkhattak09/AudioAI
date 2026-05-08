"""CTC decoding utilities."""

import torch


BLANK_TOKEN = 0


def ctc_greedy_decode(
    logits: torch.Tensor,
    blank_id: int = BLANK_TOKEN,
) -> list[int]:
    """Greedy CTC decode from logits.

    Args:
        logits: [T, vocab] or [B, T, vocab] tensor of logits.
        blank_id: index of the blank token.

    Returns:
        If input is 2D, returns list of token ids.
        If input is 3D, returns list of lists of token ids.
    """
    if logits.dim() == 2:
        token_ids = logits.argmax(dim=-1).tolist()
        collapsed = []
        prev = None
        for tid in token_ids:
            if tid != blank_id and tid != prev:
                collapsed.append(tid)
            prev = tid
        return collapsed

    if logits.dim() == 3:
        results = []
        for b in range(logits.shape[0]):
            token_ids = logits[b].argmax(dim=-1).tolist()
            collapsed = []
            prev = None
            for tid in token_ids:
                if tid != blank_id and tid != prev:
                    collapsed.append(tid)
                prev = tid
            results.append(collapsed)
        return results

    raise ValueError(f"Expected logits of shape [T, vocab] or [B, T, vocab], got {logits.shape}")
