"""Tacotron2 model modules.

Reference architectures:
- https://github.com/NVIDIA/tacotron2
- https://github.com/priyammaz/Tacotron-From-Scratch
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Tacotron2Config:
    n_symbols: int = 32
    symbols_embedding_dim: int = 512
    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embedding_dim: int = 512
    attention_rnn_dim: int = 1024
    decoder_rnn_dim: int = 1024
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31
    prenet_dim: int = 256
    max_decoder_steps: int = 1000
    gate_threshold: float = 0.5
    p_attention_dropout: float = 0.1
    p_decoder_dropout: float = 0.1
    postnet_embedding_dim: int = 512
    postnet_kernel_size: int = 5
    postnet_n_convolutions: int = 5
    n_mels: int = 80


class LinearNorm(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = "linear"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = "linear",
    ):
        super().__init__()
        if padding is None:
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, config: Tacotron2Config):
        super().__init__()
        self.embedding = nn.Embedding(config.n_symbols, config.symbols_embedding_dim)
        std = math.sqrt(2.0 / (config.n_symbols + config.symbols_embedding_dim))
        nn.init.normal_(self.embedding.weight, 0, std)

        convs = []
        for i in range(config.encoder_n_convolutions):
            in_ch = config.symbols_embedding_dim if i == 0 else config.encoder_embedding_dim
            convs.append(
                nn.Sequential(
                    ConvNorm(in_ch, config.encoder_embedding_dim, kernel_size=config.encoder_kernel_size, w_init_gain="relu"),
                    nn.BatchNorm1d(config.encoder_embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                )
            )
        self.convs = nn.ModuleList(convs)
        self.lstm = nn.LSTM(
            config.encoder_embedding_dim,
            config.encoder_embedding_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        # x: [B, T_in]
        x = self.embedding(x)  # [B, T_in, embed]
        x = x.transpose(1, 2)  # [B, embed, T_in]
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)  # [B, T_in, embed]
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs  # [B, T_in, encoder_embedding_dim]


class Prenet(nn.Module):
    def __init__(self, in_dim: int, sizes: list[int]):
        super().__init__()
        layers = []
        for i, size in enumerate(sizes):
            in_size = in_dim if i == 0 else sizes[i - 1]
            layers.append(nn.Linear(in_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
        self.prenet = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prenet(x)


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters: int, attention_kernel_size: int, attention_dim: int):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters, kernel_size=attention_kernel_size, padding=padding, bias=False, w_init_gain="tanh")
        self.location_dense = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain="tanh")

    def forward(self, attention_weights_cat: torch.Tensor) -> torch.Tensor:
        # attention_weights_cat: [B, 2, T_in]
        processed = self.location_conv(attention_weights_cat)  # [B, attention_n_filters, T_in]
        processed = processed.transpose(1, 2)  # [B, T_in, attention_n_filters]
        return self.location_dense(processed)  # [B, T_in, attention_dim]


class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_rnn_dim: int, embedding_dim: int, attention_dim: int, location_n_filters: int, location_kernel_size: int):
        super().__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init_gain="tanh")
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, w_init_gain="tanh")
        self.location_layer = LocationLayer(location_n_filters, location_kernel_size, attention_dim)
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query: torch.Tensor, processed_memory: torch.Tensor, attention_weights_cat: torch.Tensor) -> torch.Tensor:
        # query: [B, 1, attention_rnn_dim]
        # processed_memory: [B, T_in, attention_dim]
        # attention_weights_cat: [B, 2, T_in]
        processed_query = self.query_layer(query)  # [B, 1, attention_dim]
        processed_location = self.location_layer(attention_weights_cat)  # [B, T_in, attention_dim]
        energies = self.v(torch.tanh(processed_query + processed_memory + processed_location))  # [B, T_in, 1]
        return energies.squeeze(-1)  # [B, T_in]

    def forward(
        self,
        attention_hidden_state: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        attention_weights_cat: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)
        if mask is not None:
            alignment.masked_fill_(mask, self.score_mask_value)
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)  # [B, 1, embedding_dim]
        return attention_context, attention_weights


class Decoder(nn.Module):
    def __init__(self, config: Tacotron2Config):
        super().__init__()
        self.n_mels = config.n_mels
        self.max_decoder_steps = config.max_decoder_steps
        self.gate_threshold = config.gate_threshold
        self.p_attention_dropout = config.p_attention_dropout
        self.p_decoder_dropout = config.p_decoder_dropout

        self.prenet = Prenet(config.n_mels, [config.prenet_dim, config.prenet_dim])
        self.attention_rnn = nn.LSTMCell(config.prenet_dim + config.encoder_embedding_dim, config.attention_rnn_dim)
        self.attention_layer = LocationSensitiveAttention(
            config.attention_rnn_dim,
            config.encoder_embedding_dim,
            config.attention_dim,
            config.attention_location_n_filters,
            config.attention_location_kernel_size,
        )
        self.decoder_rnn = nn.LSTMCell(config.attention_rnn_dim + config.encoder_embedding_dim, config.decoder_rnn_dim, 1)
        self.linear_projection = LinearNorm(config.decoder_rnn_dim + config.encoder_embedding_dim, config.n_mels)
        self.gate_layer = LinearNorm(config.decoder_rnn_dim + config.encoder_embedding_dim, 1, w_init_gain="sigmoid")

    def get_go_frame(self, memory: torch.Tensor) -> torch.Tensor:
        B = memory.shape[0]
        return memory.new_zeros(B, self.n_mels)

    def initialize_decoder_states(self, memory: torch.Tensor) -> tuple:
        B = memory.shape[0]
        MAX_TIME = memory.shape[1]
        device = memory.device
        attention_hidden = memory.new_zeros(B, self.attention_rnn.hidden_size)
        attention_cell = memory.new_zeros(B, self.attention_rnn.hidden_size)
        decoder_hidden = memory.new_zeros(B, self.decoder_rnn.hidden_size)
        decoder_cell = memory.new_zeros(B, self.decoder_rnn.hidden_size)
        attention_weights = memory.new_zeros(B, MAX_TIME)
        attention_weights_cum = memory.new_zeros(B, MAX_TIME)
        attention_context = memory.new_zeros(B, memory.shape[2])
        processed_memory = self.attention_layer.memory_layer(memory)
        return (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        )

    def parse_decoder_inputs(self, decoder_inputs: torch.Tensor) -> torch.Tensor:
        # decoder_inputs: [B, n_mels, T_out] -> [T_out, B, n_mels]
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs: list, gate_outputs: list, alignments: list) -> tuple:
        # mel_outputs: list of [B, n_mels]
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1)  # [B, T_out, n_mels]
        mel_outputs = mel_outputs.transpose(1, 2)  # [B, n_mels, T_out]
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)  # [B, T_out]
        alignments = torch.stack(alignments).transpose(0, 1)  # [B, T_out, T_in]
        return mel_outputs, gate_outputs, alignments

    def decode(
        self,
        decoder_input: torch.Tensor,
        attention_hidden: torch.Tensor,
        attention_cell: torch.Tensor,
        decoder_hidden: torch.Tensor,
        decoder_cell: torch.Tensor,
        attention_weights: torch.Tensor,
        attention_weights_cum: torch.Tensor,
        attention_context: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple:
        cell_input = torch.cat((decoder_input, attention_context), -1)
        attention_hidden, attention_cell = self.attention_rnn(cell_input, (attention_hidden, attention_cell))
        attention_hidden = F.dropout(attention_hidden, self.p_attention_dropout, self.training)
        attention_weights_cat = torch.cat((attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)), dim=1)
        attention_context, attention_weights = self.attention_layer(
            attention_hidden.unsqueeze(1),
            memory,
            processed_memory,
            attention_weights_cat,
            mask,
        )
        attention_context = attention_context.squeeze(1)
        attention_weights_cum = attention_weights_cum + attention_weights

        decoder_input_ = torch.cat((attention_hidden, attention_context), -1)
        decoder_hidden, decoder_cell = self.decoder_rnn(decoder_input_, (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(decoder_hidden, self.p_decoder_dropout, self.training)
        decoder_hidden_attention_context = torch.cat((decoder_hidden, attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return (
            decoder_output,
            gate_prediction,
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
        )

    def forward(
        self,
        memory: torch.Tensor,
        decoder_inputs: torch.Tensor,
        memory_lengths: torch.Tensor,
    ) -> tuple:
        # memory: [B, T_in, encoder_embedding_dim]
        # decoder_inputs: [B, n_mels, T_out] teacher forced mels
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        # decoder_inputs: [T_out, B, n_mels]
        go_frame = self.get_go_frame(memory).unsqueeze(0)  # [1, B, n_mels]
        decoder_inputs = torch.cat([go_frame, decoder_inputs[:-1]], dim=0)

        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self.initialize_decoder_states(memory)

        mask = ~get_mask_from_lengths(memory_lengths)
        mel_outputs = []
        gate_outputs = []
        alignments = []

        while len(mel_outputs) < decoder_inputs.size(0):
            decoder_input = self.prenet(decoder_inputs[len(mel_outputs)])
            (
                mel_output,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask,
            )
            mel_outputs.append(mel_output.squeeze(1))
            gate_outputs.append(gate_output.squeeze(1))
            alignments.append(attention_weights)

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory: torch.Tensor) -> tuple:
        decoder_input = self.get_go_frame(memory)
        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self.initialize_decoder_states(memory)

        mel_outputs = []
        gate_outputs = []
        alignments = []
        while True:
            decoder_input = self.prenet(decoder_input)
            (
                mel_output,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                None,
            )
            mel_outputs.append(mel_output.squeeze(1))
            gate_outputs.append(gate_output.squeeze(1))
            alignments.append(attention_weights)

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                break

            decoder_input = mel_output.squeeze(1)

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments


class Postnet(nn.Module):
    def __init__(self, config: Tacotron2Config):
        super().__init__()
        layers = []
        for i in range(config.postnet_n_convolutions):
            in_ch = config.n_mels if i == 0 else config.postnet_embedding_dim
            out_ch = config.n_mels if i == config.postnet_n_convolutions - 1 else config.postnet_embedding_dim
            w_init_gain = "linear" if i == config.postnet_n_convolutions - 1 else "tanh"
            layers.append(
                nn.Sequential(
                    ConvNorm(in_ch, out_ch, kernel_size=config.postnet_kernel_size, padding=(config.postnet_kernel_size - 1) // 2, w_init_gain=w_init_gain),
                    nn.BatchNorm1d(out_ch),
                )
            )
        self.convolutions = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_mels, T_out]
        for i, conv in enumerate(self.convolutions):
            if i < len(self.convolutions) - 1:
                x = torch.tanh(conv(x))
                x = F.dropout(x, 0.5, self.training)
            else:
                x = conv(x)
        return x


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_len = lengths.max().item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


class Tacotron2(nn.Module):
    def __init__(self, config: Tacotron2Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.postnet = Postnet(config)
        self.n_mels = config.n_mels

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_targets: torch.Tensor,
        mel_lengths: torch.Tensor | None = None,
    ) -> dict:
        # text_tokens: [B, T_text]
        # mel_targets: [B, n_mels, T_mel]
        encoder_outputs = self.encoder(text_tokens, text_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mel_targets, text_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs) + mel_outputs
        return {
            "mel_before_postnet": mel_outputs,
            "mel_after_postnet": mel_outputs_postnet,
            "stop_logits": gate_outputs,
            "attention_weights": alignments,
        }

    def inference(self, text_tokens: torch.Tensor) -> dict:
        was_training = self.training
        self.eval()
        try:
            encoder_outputs = self.encoder(text_tokens, torch.tensor([text_tokens.shape[1]], device=text_tokens.device))
            mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)
            mel_outputs_postnet = self.postnet(mel_outputs) + mel_outputs
            return {
                "mel_before_postnet": mel_outputs,
                "mel_after_postnet": mel_outputs_postnet,
                "stop_logits": gate_outputs,
                "attention_weights": alignments,
            }
        finally:
            self.train(was_training)
