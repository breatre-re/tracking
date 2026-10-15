"""Plain dense CNN baseline; architecture chosen here, not specified by Zhang."""
import torch
from torch import nn


class AssociationCNN(nn.Module):
    def __init__(self, channels=(64, 128, 128, 64), kernel_size=5, dropout=.3):
        super().__init__()
        self.config = {'channels': list(channels), 'kernel_size': kernel_size, 'dropout': dropout,
                       'receptive_field': 1 + len(channels) * (kernel_size - 1),
                       'attention': False, 'pooling': False, 'normalization': False}
        layers = []
        previous = 3
        for width in channels:
            layers.extend([nn.Conv2d(previous, width, kernel_size, padding=kernel_size // 2),
                           nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Conv2d(previous, 1, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features).squeeze(1)  # 40x40 logits; no row softmax.


def runtime_check(device):
    import time
    from ctda_model import focal_loss_elements
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    model = AssociationCNN().to(device)
    batch = 256 if device.type == 'cuda' else 2
    x = torch.randn(batch, 3, 40, 40, device=device)
    y = torch.eye(40, device=device).expand(batch, -1, -1)
    valid = torch.ones_like(y, dtype=torch.bool)
    valid[:, -5:, :] = False
    elapsed = []
    for i in range(5 if device.type == 'cuda' else 1):
        model.zero_grad(set_to_none=True)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits = model(x)
        assert logits.shape == y.shape
        logits.retain_grad()
        loss = focal_loss_elements(logits, y)[valid].mean()
        loss.backward()
        if device.type == 'cuda': torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - start)
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert not logits.grad[~valid].any()
    return {'output_shape': list(logits.shape), 'parameters': sum(p.numel() for p in model.parameters()),
            'all_gradients_finite': True, 'padding_logit_gradient_zero': True,
            'forward_backward_seconds': sum(elapsed[1:] or elapsed) / len(elapsed[1:] or elapsed),
            'configuration': model.config}
