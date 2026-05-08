"""Character-level tokenizer for TTS."""

import string

import torch


class CharTokenizer:
    """Simple character-level tokenizer.

    Vocab:
      - <pad>
      - <eos>
      - lowercase a-z
      - space
      - common punctuation: .,!?-'
    """

    PAD = "<pad>"
    EOS = "<eos>"

    def __init__(self):
        vocab_chars = [self.PAD, self.EOS]
        vocab_chars.extend(list(string.ascii_lowercase))
        vocab_chars.extend([" ", ".", ",", "!", "?", "-", "'"])
        self._char_to_id = {c: i for i, c in enumerate(vocab_chars)}
        self._id_to_char = {i: c for c, i in self._char_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._char_to_id)

    @property
    def pad_id(self) -> int:
        return self._char_to_id[self.PAD]

    @property
    def eos_id(self) -> int:
        return self._char_to_id[self.EOS]

    def encode(self, text: str) -> torch.LongTensor:
        """Encode text to token ids."""
        text = text.lower()
        ids = []
        for ch in text:
            ids.append(self._char_to_id.get(ch, self.pad_id))
        ids.append(self.eos_id)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Decode token ids to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        chars = []
        for idx in ids:
            if idx == self.eos_id:
                break
            ch = self._id_to_char.get(idx, "")
            if ch == self.PAD:
                continue
            chars.append(ch)
        return "".join(chars)
