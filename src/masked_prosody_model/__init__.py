from pathlib import Path
from dataclasses import dataclass

import yaml
import torch
from torch import nn
from transformers.utils.hub import cached_file
from rich.console import Console
import librosa
import numpy as np

console = Console()

from masked_prosody_model.modules import (
    PositionalEncoding,
    TransformerEncoder,
    ConformerLayer,
)

from speech_collator.measures import (
    PitchMeasure,
    EnergyMeasure,
    VoiceActivityMeasure,
)


@dataclass
class ModelArgs:
    n_layers: int = 8
    n_heads: int = 2
    kernel_size: int = 7
    filter_size: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1
    pitch_min: float = 50
    pitch_max: float = 300
    energy_min: float = 0
    energy_max: float = 0.2
    vad_min: float = 0
    vad_max: float = 1
    bins: int = 128


class MaskedProsodyModel(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        bins = args.bins

        self.pitch_embedding = nn.Embedding(bins + 2, args.filter_size)
        self.energy_embedding = nn.Embedding(bins + 2, args.filter_size)
        self.vad_embedding = nn.Embedding(2 + 2, args.filter_size)

        self.positional_encoding = PositionalEncoding(args.filter_size)

        self.transformer = TransformerEncoder(
            ConformerLayer(
                args.filter_size,
                args.n_heads,
                conv_in=args.filter_size,
                conv_filter_size=args.filter_size,
                conv_kernel=(args.kernel_size, 1),
                batch_first=True,
                dropout=args.dropout,
            ),
            num_layers=args.n_layers,
        )

        self.output_pitch = nn.Sequential(
            nn.Linear(args.filter_size, bins),
        )

        self.output_energy = nn.Sequential(
            nn.Linear(args.filter_size, bins),
        )

        self.output_vad = nn.Sequential(
            nn.Linear(args.filter_size, 1),
        )

        self.apply(self._init_weights)

        self.args = args

        self.pitch_measure = PitchMeasure()
        self.energy_measure = EnergyMeasure()
        self.vad_measure = VoiceActivityMeasure()

        self.bins = torch.linspace(0, 1, bins)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, return_layer=None):
        pitch = x[:, 0]
        energy = x[:, 1]
        vad = x[:, 2]
        pitch = self.pitch_embedding(pitch)
        energy = self.energy_embedding(energy)
        vad = self.vad_embedding(vad)
        x = pitch + energy + vad
        x = self.positional_encoding(x)
        if return_layer is not None:
            x, reprs = self.transformer(x, return_layer=return_layer)
        else:
            x = self.transformer(x)
        pitch = self.output_pitch(x)
        energy = self.output_energy(x)
        vad = self.output_vad(x)
        if return_layer is not None:
            return {
                "y": [
                    pitch,
                    energy,
                    vad,
                ],
                "representations": reprs,
            }
        else:
            return [
                pitch,
                energy,
                vad,
            ]

    def save_model(self, path, accelerator=None, onnx=False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if accelerator is not None:
            accelerator.save_model(self, path)
        else:
            torch.save(self.state_dict(), path / "pytorch_model.bin")
        if onnx:
            try:
                self.export_onnx(path / "model.onnx")
            except Exception as e:
                console.print(f"[red]Skipping ONNX export[/red]: {e}")
        with open(path / "model_config.yml", "w") as f:
            f.write(yaml.dump(self.args.__dict__, Dumper=yaml.Dumper))

    @staticmethod
    def from_pretrained(path_or_hubid):
        path = Path(path_or_hubid)
        if path.exists():
            config_file = path / "model_config.yml"
            model_file = path / "pytorch_model.bin"
        else:
            config_file = cached_file(path_or_hubid, "model_config.yml")
            model_file = cached_file(path_or_hubid, "pytorch_model.bin")
        args = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
        model_specific_args = [
            "bins",
            "max_length",
        ]
        fargs = {k: v for k, v in args.items() if k not in model_specific_args}
        margs = ModelArgs(**fargs)
        margs.bins = args["bins"]
        margs.max_length = args["max_length"]
        model = MaskedProsodyModel(margs)
        model.load_state_dict(torch.load(model_file))
        return model

    def process_audio(self, audio_path, layer=7):
        audio, sr = librosa.load(audio_path, sr=22050)
        audio = audio / np.abs(audio).max()
        # window into 6s chunks
        windows = []
        for i in range(0, len(audio), sr * 6):
            windows.append(audio[i : i + sr * 6])
        results = []
        for i, window in enumerate(windows):
            pitch = self.pitch_measure(window, np.array([1000]))["measure"]
            energy = self.energy_measure(window, np.array([1000]))["measure"]
            vad = self.vad_measure(window, np.array([1000]))["measure"]
            pitch[np.isnan(pitch)] = -1000
            energy[np.isnan(energy)] = -1000
            vad[np.isnan(vad)] = -1000
            pitch = np.clip(
                pitch,
                self.args.pitch_min,
                self.args.pitch_max,
            ) / (self.args.pitch_max - self.args.pitch_min)
            energy = np.clip(
                energy,
                self.args.energy_min,
                self.args.energy_max,
            ) / (self.args.energy_max - self.args.energy_min)
            vad = np.clip(
                vad,
                self.args.vad_min,
                self.args.vad_max,
            ) / (self.args.vad_max - self.args.vad_min)
            pitch = torch.tensor(pitch)
            energy = torch.tensor(energy)
            vad = torch.tensor(vad)
            pitch = torch.bucketize(pitch, self.bins).long().unsqueeze(0)
            energy = torch.bucketize(energy, self.bins).long().unsqueeze(0)
            vad = torch.bucketize(vad, torch.linspace(0, 1, 2)).long().unsqueeze(0)
            all_features = torch.stack([pitch, energy, vad]).transpose(0, 1)
            result = self(all_features, return_layer=layer)
            results.append(result)
        # bring all representations together
        representations = []
        for result in results:
            representations.append(result["representations"].squeeze(0))
        representations = torch.cat(representations, dim=0)
        return representations

    @property
    def dummy_input(self):
        torch.manual_seed(0)
        return torch.stack(
            [
                torch.randint(0, self.args.bins, (1, self.args.max_length)),
                torch.randint(0, self.args.bins, (1, self.args.max_length)),
                torch.randint(0, 2, (1, self.args.max_length)),
            ]
        ).transpose(0, 1)
