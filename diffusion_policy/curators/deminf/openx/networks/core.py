from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiEncoder(nn.Module):
    """
    Takes multiple inputs and returns them as a single representation.
    """

    def __init__(self, encoders: Dict[str, nn.Module], trunk: nn.Module):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.trunk = trunk

    def _encode(self, batch: Dict, train: bool = True):
        modalities = dict()
        for encoder_keys, encoder in self.encoders.items():
            args = []
            for encoder_key in encoder_keys.split(","):
                v = batch
                for k in encoder_key.split("->"):
                    v = v[k]
                args.append(v)
            args = tuple(args)
            if encoder is None:
                modalities[encoder_keys] = args[0] if len(args) == 1 else args
            else:
                modalities[encoder_keys] = encoder(*args) if train else encoder(*args)
        return modalities

    def forward(self, batch: Dict, train: bool = True):
        x = self._encode(batch, train=train)
        return self.trunk(x) if train else self.trunk(x)


class MultiDecoder(nn.Module):
    """
    Takes a single representation and returns them as multiple inputs.
    """

    def __init__(self, trunk: nn.Module, decoders: Dict[str, nn.Module]):
        super().__init__()
        self.trunk = trunk
        self.decoders = nn.ModuleDict(decoders)

    def _decode(self, z, batch: Dict, train: bool = True):
        output = dict()
        for decoder_keys, decoder in self.decoders.items():
            args = []
            for decoder_key in decoder_keys.split(","):
                v = batch
                for k in decoder_key.split("->"):
                    v = v[k]
                args.append(v)
            args = tuple(args)
            first_decoder_key = decoder_keys.split(",")[0]
            assert first_decoder_key not in output
            output[first_decoder_key] = z if decoder is None else decoder(z, *args)
        return output

    def forward(self, z, batch: Dict, train: bool = True):
        z = self.trunk(z)
        return self._decode(z, batch, train=train)


class Concatenate(nn.Module):
    def __init__(self, model: Optional[nn.Module] = None, flatten_time: bool = True):
        super().__init__()
        self.model = model
        self.flatten_time = flatten_time

    def forward(self, modalities: Dict[str, torch.Tensor], train: bool = False):
        if self.flatten_time:
            x = torch.cat(
                [v.reshape(v.shape[0], -1) for k, v in sorted(modalities.items())],
                dim=-1,
            )  # (B, D)
        else:
            x = torch.cat(
                [v.reshape(v.shape[:2] + (-1,)) for k, v in sorted(modalities.items())],
                dim=-1,
            )  # (B, T, D)

        if self.model is not None:
            x = self.model(x)
        return x


class Tokenize(nn.Module):
    """
    Tokenizers modalities into (B, T, S, D) or (B, T, D) and feeds it to `model`.
    """

    def __init__(
        self,
        embed_dim: int,
        flatten_time: bool = True,
        project_all: bool = False,
        model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.flatten_time = flatten_time
        self.project_all = project_all
        self.model = model
        self.projections = nn.ModuleDict()

    def forward(self, modalities: Dict[str, torch.Tensor], train: bool = False):
        tokens = []
        for k in sorted(modalities.keys()):
            x = modalities[k]
            shape = x.shape
            if len(shape) == 2:
                new_shape = (shape[0], 1, 1, shape[-1])
            elif len(shape) == 3:
                new_shape = (shape[0], shape[1], 1, shape[-1])
            elif len(shape) > 3:
                new_shape = (shape[0], shape[1], -1, shape[-1])
            else:
                new_shape = shape

            x = x.reshape(*new_shape)

            # Projection if needed
            if x.shape[-1] != self.embed_dim or self.project_all:
                if k not in self.projections:
                    self.projections[k] = nn.Linear(x.shape[-1], self.embed_dim)
                proj = self.projections[k]
                x = proj(x)
            tokens.append(x)

        b, t = tokens[0].shape[:2]
        if not all(tok.shape[1] == t for tok in tokens) and not self.flatten_time:
            raise ValueError(
                "flatten_time was not set to True in Tokenize, but not all modalities had the same time dimension."
            )

        if self.flatten_time:
            tokens = [x.reshape(b, -1, self.embed_dim) for x in tokens]

        x = torch.cat(tokens, dim=-2)

        if self.model is not None:
            x = self.model(x)
        return x
