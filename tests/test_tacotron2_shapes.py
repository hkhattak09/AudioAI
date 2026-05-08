import torch

from voice_ai.tts.tacotron2 import Tacotron2, Tacotron2Config


def test_tacotron2_forward_shape():
    config = Tacotron2Config(
        n_symbols=32,
        symbols_embedding_dim=16,
        encoder_embedding_dim=16,
        attention_rnn_dim=16,
        decoder_rnn_dim=16,
        attention_dim=8,
        attention_location_n_filters=8,
        attention_location_kernel_size=15,
        prenet_dim=8,
        postnet_embedding_dim=16,
        n_mels=16,
    )
    model = Tacotron2(config)
    B = 2
    T_text = 12
    T_mel = 20
    text_tokens = torch.randint(0, config.n_symbols, (B, T_text))
    text_lengths = torch.tensor([T_text, T_text - 2])
    mel_targets = torch.randn(B, config.n_mels, T_mel)
    outputs = model(text_tokens, text_lengths, mel_targets)
    assert outputs["mel_before_postnet"].shape == (B, config.n_mels, T_mel)
    assert outputs["mel_after_postnet"].shape == (B, config.n_mels, T_mel)
    assert outputs["stop_logits"].shape == (B, T_mel)
    assert outputs["attention_weights"].shape[0] == B


def test_tacotron2_inference_shape():
    config = Tacotron2Config(
        n_symbols=32,
        symbols_embedding_dim=16,
        encoder_embedding_dim=16,
        attention_rnn_dim=16,
        decoder_rnn_dim=16,
        attention_dim=8,
        attention_location_n_filters=8,
        attention_location_kernel_size=15,
        prenet_dim=8,
        postnet_embedding_dim=16,
        n_mels=16,
        max_decoder_steps=10,
    )
    model = Tacotron2(config)
    text_tokens = torch.randint(0, config.n_symbols, (1, 5))
    outputs = model.inference(text_tokens)
    assert outputs["mel_after_postnet"].shape[1] == config.n_mels


def test_tacotron2_teacher_forcing_first_input_is_go_frame():
    """Assert that the first decoder input is the GO frame, not target frame 0."""
    config = Tacotron2Config(
        n_symbols=32,
        symbols_embedding_dim=16,
        encoder_embedding_dim=16,
        attention_rnn_dim=16,
        decoder_rnn_dim=16,
        attention_dim=8,
        attention_location_n_filters=8,
        attention_location_kernel_size=15,
        prenet_dim=8,
        postnet_embedding_dim=16,
        n_mels=16,
    )
    model = Tacotron2(config)
    B = 1
    T_text = 5
    T_mel = 10
    text_tokens = torch.randint(0, config.n_symbols, (B, T_text))
    text_lengths = torch.tensor([T_text])
    # Use a distinctive mel target so we can detect target leakage
    mel_targets = torch.ones(B, config.n_mels, T_mel) * 9.99

    first_prenet_input = []

    original_prenet_forward = model.decoder.prenet.forward

    def capturing_prenet(x):
        first_prenet_input.append(x.detach().clone())
        return original_prenet_forward(x)

    model.decoder.prenet.forward = capturing_prenet

    with torch.no_grad():
        model(text_tokens, text_lengths, mel_targets)

    model.decoder.prenet.forward = original_prenet_forward

    assert len(first_prenet_input) == T_mel
    # First input should be zeros (GO frame), not 9.99 (target frame 0)
    assert torch.allclose(first_prenet_input[0], torch.zeros_like(first_prenet_input[0]))
    # Second input should be target frame 0 (9.99)
    assert torch.allclose(first_prenet_input[1], torch.ones_like(first_prenet_input[1]) * 9.99)
