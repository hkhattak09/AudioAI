import torch

from voice_ai.pipeline.voice_agent import VoiceAgent


def test_pipeline_stub():
    agent = VoiceAgent()
    # Use a sine wave so VAD detects speech
    t = torch.arange(0, 16000) / 16000
    waveform = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    turn = agent.process_audio(waveform, sample_rate=16000)
    assert turn.transcript == "hello from fake stt"
    assert turn.llm_response == "Hello. This is a local stub response."
    assert turn.output_audio is not None
    assert turn.output_audio.shape[0] == 1


def test_voice_agent_from_config():
    config = {
        "tts": {"n_mels": 16, "n_frames": 20},
        "vocoder": {"output_samples": 500},
    }
    agent = VoiceAgent.from_config(config)
    t = torch.arange(0, 16000) / 16000
    waveform = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    turn = agent.process_audio(waveform, sample_rate=16000)
    assert turn.output_audio is not None
    assert turn.output_audio.shape == torch.Size([1, 1, 500])


def test_voice_agent_from_config_uses_local_stub_llm():
    config = {"llm": {"type": "local_stub"}}
    agent = VoiceAgent.from_config(config)
    assert agent.llm is not None
    t = torch.arange(0, 16000) / 16000
    waveform = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    turn = agent.process_audio(waveform, sample_rate=16000)
    assert turn.llm_response == "Hello. This is a local stub response."
