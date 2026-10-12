"""Phase 4 CTDA tracker adapter.

``CTDAAssociation`` implements the existing ``AssociationModel`` contract
(``associate(predicted_positions, measurements, gate)``) so it drops into
``MultiTargetTracker`` unchanged. It consumes positions only: the tracker has no
access to ground-truth ids or labels.

Pipeline: predicted track positions -> rows, current measurements -> columns,
RDF feature channels -> CTDA network -> threshold ``lambda_a`` (default 0.1) ->
Kuhn-Munkres one-to-one assignment.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.models.ctda.features import build_feature_tensor
from src.models.ctda.model import CTDANet, CTDAConfig
from src.models.ctda.objective import decode_association
from tracking_prototype import AssociationModel


Array = np.ndarray


class CTDAAssociation(AssociationModel):
    """Transformer-based association compatible with the prototype tracker."""

    def __init__(
        self,
        model: CTDANet,
        channels: Sequence[str] = ("x_rdf", "y_rdf", "range_rdf"),
        pad_size: int = 40,
        x_bias: float = 0.001,
        pad_value: float = -1.0,
        threshold: float = 0.1,
        one_to_one: bool = True,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.channels = tuple(channels)
        self.pad_size = int(pad_size)
        self.x_bias = float(x_bias)
        self.pad_value = float(pad_value)
        self.threshold = float(threshold)
        self.one_to_one = bool(one_to_one)
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, device: str = "cpu", threshold: Optional[float] = None) -> "CTDAAssociation":
        payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        configuration = payload["config"]
        model = CTDANet(CTDAConfig.from_dict(configuration))
        model.load_state_dict(payload["state_dict"])
        features = configuration.get("features", {})
        dataset = configuration.get("dataset", {})
        return cls(
            model,
            channels=tuple(features.get("channels", ["x_rdf", "y_rdf", "range_rdf"])),
            pad_size=int(dataset.get("pad_size", 40)),
            x_bias=float(features.get("x_bias", 0.001)),
            pad_value=float(features.get("pad_value", -1.0)),
            threshold=float(configuration.get("inference", {}).get("association_threshold", 0.1)) if threshold is None else threshold,
            one_to_one=bool(configuration.get("inference", {}).get("one_to_one", True)),
            device=device,
        )

    def associate(self, predicted_positions: Array, measurements: Array, gate: float) -> List[Tuple[int, int]]:
        predicted_positions = np.asarray(predicted_positions, dtype=float).reshape((-1, 2))
        measurements = np.asarray(measurements, dtype=float).reshape((-1, 2))
        tracks = int(min(len(predicted_positions), self.pad_size))
        detected = int(min(len(measurements), self.pad_size))
        if tracks == 0 or detected == 0:
            return []

        features = build_feature_tensor(
            predicted_positions[:tracks],
            measurements[:detected],
            self.pad_size,
            self.channels,
            self.x_bias,
            self.pad_value,
        )
        row_mask = np.zeros(self.pad_size, dtype=bool)
        col_mask = np.zeros(self.pad_size, dtype=bool)
        row_mask[:tracks] = True
        col_mask[:detected] = True

        with torch.no_grad():
            tensor = torch.from_numpy(features[None]).to(self.device)
            row_tensor = torch.from_numpy(row_mask[None]).to(self.device)
            col_tensor = torch.from_numpy(col_mask[None]).to(self.device)
            logits = self.model(tensor, row_tensor, col_tensor)
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()

        association = decode_association(probabilities, row_mask, col_mask, self.threshold, self.one_to_one)
        rows, columns = np.nonzero(association)
        return [(int(row), int(column)) for row, column in zip(rows, columns)]
