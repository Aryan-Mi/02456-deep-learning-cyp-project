"""CheMeleon graph encoder with the project's shared prediction head.
Load the released chemeleon_mp.pt locally, because this module never downloads weights.
"""

from pathlib import Path

import numpy as np
import torch
from chemprop import data, featurizers
from chemprop import nn as chemprop_nn
from chemprop.data import collate_batch
from chemprop.utils import make_mol
from torch import nn
from torch.utils.data import DataLoader

from models.readouts import build_readout


class CheMeleonModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg = cfg
        checkpoint_path = Path(cfg.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CheMeleon checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.encoder = chemprop_nn.BondMessagePassing(**checkpoint["hyper_parameters"])  # pyright: ignore[reportAbstractUsage]
        self.encoder.load_state_dict(checkpoint["state_dict"])
        self.encoder.requires_grad_(not cfg.freeze_encoder)
        self.aggregation = chemprop_nn.MeanAggregation()
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.readout = build_readout(readout, self.encoder.output_dim, n_targets)
        self.to(device)
        if cfg.freeze_encoder:
            self.encoder.eval()

    @property
    def device(self):
        return next(self.readout.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def loader(self, frame, targets, batch_size, shuffle=False):
        labels = frame.reindex(columns=targets).to_numpy(dtype=np.float32)
        points = [
            data.MoleculeDatapoint(mol=make_mol(smiles), y=y, name=smiles)
            for smiles, y in zip(frame.SMILES, labels)
        ]
        dataset = data.MoleculeDataset(points, self.featurizer)
        dataset.cache = True

        def collate_molecules(samples):
            batch = collate_batch(samples)
            return {"graph": batch.bmg, "y": batch.Y}

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_molecules,
            num_workers=0,
        )

    def param_groups(self):
        groups = [{"params": list(self.readout.parameters()), "lr": self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({"params": list(self.encoder.parameters()), "lr": self.cfg.encoder_lr})
        return groups

    def forward(self, batch):
        graph = batch["graph"]
        graph.to(self.device)
        atom_features = self.encoder(graph)
        molecule_features = self.aggregation(atom_features, graph.batch)
        return self.readout(molecule_features)
