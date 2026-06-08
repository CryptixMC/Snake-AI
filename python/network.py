import torch

# Architecture: 20 -> 64 -> 4
# Flat weight layout per individual:
#   W1: 20*64=1280  b1: 64  W2: 64*4=256  b2: 4
#   Total: 1604
_IN  = 20
_H   = 64
_OUT = 4
_W1  = _IN * _H
_B1  = _H
_W2  = _H * _OUT
_B2  = _OUT
PARAM_COUNT = _W1 + _B1 + _W2 + _B2  # 1604


def forward_batch(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    N = params.shape[0]
    i0, i1, i2, i3 = 0, _W1, _W1 + _B1, _W1 + _B1 + _W2

    w1 = params[:, i0:i1].view(N, _IN, _H)
    b1 = params[:, i1:i2]
    w2 = params[:, i2:i3].view(N, _H, _OUT)
    b2 = params[:, i3:]

    h = torch.relu(torch.bmm(x.unsqueeze(1), w1).squeeze(1) + b1)
    return torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2


def forward_with_activations(params: torch.Tensor, x: torch.Tensor) -> dict:
    import numpy as np

    w1 = params[:_W1].view(_IN, _H)
    b1 = params[_W1:_W1 + _B1]
    w2 = params[_W1 + _B1:_W1 + _B1 + _W2].view(_H, _OUT)
    b2 = params[_W1 + _B1 + _W2:]

    h   = torch.relu(x @ w1 + b1)
    out = h @ w2 + b2

    return {
        "inputs": x.detach().cpu().numpy().astype(np.float32),
        "hidden": h.detach().cpu().numpy().astype(np.float32),
        "output": out.detach().cpu().numpy().astype(np.float32),
        "w1": w1.detach().cpu().numpy().astype(np.float32),
        "w2": w2.detach().cpu().numpy().astype(np.float32),
        "action": int(out.argmax()),
    }


def random_population(n: int, device: torch.device) -> torch.Tensor:
    params = torch.zeros(n, PARAM_COUNT, device=device)
    torch.nn.init.xavier_uniform_(params[:, :_W1].view(n, _IN, _H))
    torch.nn.init.xavier_uniform_(params[:, _W1 + _B1:_W1 + _B1 + _W2].view(n, _H, _OUT))
    return params
