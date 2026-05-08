import torch

from voice_ai.training.losses import generator_adversarial_loss
from voice_ai.vocoder.hifigan import Generator, HiFiGAN, HiFiGANConfig, MultiPeriodDiscriminator, MultiScaleDiscriminator


def test_hifigan_generator_shape():
    config = HiFiGANConfig(
        resblock_kernel_sizes=(3, 7),
        resblock_dilation_sizes=((1, 3), (1, 3)),
        upsample_rates=(2, 2),
        upsample_initial_channel=16,
        upsample_kernel_sizes=(4, 4),
        model_in_channels=16,
        model_out_channels=1,
    )
    gen = Generator(config)
    mel = torch.randn(1, config.model_in_channels, 20)
    wav = gen(mel)
    assert wav.shape[0] == 1
    assert wav.shape[1] == 1
    assert wav.shape[2] > 0


def test_hifigan_discriminators():
    config = HiFiGANConfig(
        model_out_channels=1,
    )
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    y = torch.randn(1, 1, 100)
    y_hat = torch.randn(1, 1, 100)
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = mpd(y, y_hat)
    assert len(y_d_rs) == len(y_d_gs)
    y_d_rs2, y_d_gs2, fmap_rs2, fmap_gs2 = msd(y, y_hat)
    assert len(y_d_rs2) == len(y_d_gs2)


def test_hifigan_generator_receives_gradient():
    """Generator parameters should receive gradients through the generator path."""
    config = HiFiGANConfig(
        resblock_kernel_sizes=(3, 7),
        resblock_dilation_sizes=((1, 3), (1, 3)),
        upsample_rates=(2, 2),
        upsample_initial_channel=8,
        upsample_kernel_sizes=(4, 4),
        model_in_channels=8,
        model_out_channels=1,
    )
    model = HiFiGAN(config)
    mel = torch.randn(1, config.model_in_channels, 20)
    real_audio = torch.randn(1, 1, 80)

    fake_audio = model.generate(mel)
    gen_outputs = model.discriminate_for_generator(real_audio, fake_audio)

    # Compute a simple adversarial loss from generator path
    loss = generator_adversarial_loss(gen_outputs["mpd"][1]) + generator_adversarial_loss(gen_outputs["msd"][1])
    loss.backward()

    # Assert at least one generator parameter has a non-None gradient
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.generator.parameters())
    assert has_grad, "Generator did not receive gradients through the generator path"
