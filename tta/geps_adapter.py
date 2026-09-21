import torch
import torch.nn as nn
import torch.nn.functional as F


class GEPSAdapter:
    def __init__(self, model, adapt_lr: float = 1e-3, n_steps: int = 10):
        self.model = model
        self.adapt_lr = adapt_lr
        self.n_steps = n_steps
        self.adapt_code = None
        self.optimizer = None

    def reset(self):
        for param in self.model.parameters(): # freeze
            param.requires_grad_(False)
            param.grad = None

        mean_code = self.model.codes.data.mean(dim=0)  # (code_c,)
        self.adapt_code = nn.Parameter(mean_code.clone())
        self.optimizer = torch.optim.Adam([self.adapt_code], lr=self.adapt_lr)

    # adaptation
    def update(self, x_block: torch.Tensor, y_true: torch.Tensor):
        for _ in range(self.n_steps): # x_block : (1, T, H, W, C), y_true  : (1, T, H, W, C)
            pred = self.model(x_block, adapt_code=self.adapt_code)
            loss = F.mse_loss(pred[..., :2], y_true[..., :2])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, x_block: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x_block, adapt_code=self.adapt_code)

    @property
    def current_code(self) -> torch.Tensor:
        return self.adapt_code.detach()
