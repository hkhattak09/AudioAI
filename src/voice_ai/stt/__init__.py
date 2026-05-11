"""Speech-to-text modules."""

from voice_ai.stt.aligner import AlignmentError, AlignmentResult, forced_align_ctc

__all__ = ["AlignmentError", "AlignmentResult", "forced_align_ctc"]
