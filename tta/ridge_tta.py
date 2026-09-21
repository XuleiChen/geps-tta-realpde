import torch
import torch.nn as nn


class RidgeLastLayerAdapter:
    def __init__(self, model, lam=1e-2, lam_bias=None):
        self.model = model
        self.last_layer = model.final_conv[1]

        self.feature_extractor = model.final_conv[0]  # ResnetBlock, outputs Phi
        self.lam = lam
        self.lam_bias = lam if lam_bias is None else lam_bias

        self._captured_phi = None
        self._hook_handle = None
        self.register_prior() # snapshot the pretrained weights

    def register_prior(self):
        with torch.no_grad():
            self.w0 = self.last_layer.weight.detach().clone()   # [out_dim, dim, 1, 1, 1]
            self.b0 = self.last_layer.bias.detach().clone() if self.last_layer.bias is not None else None

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.last_layer.parameters():
            p.requires_grad_(True)

    def unfreeze_all(self):
        for p in self.model.parameters():
            p.requires_grad_(True)

    def enable_capture(self):
        def _hook(module, inp, out):
            self._captured_phi = out
        self._hook_handle = self.feature_extractor.register_forward_hook(_hook)

    def disable_capture(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @property
    def phi(self):
        return self._captured_phi

    @staticmethod
    def _flatten_phi(phi):
        # phi: [B, dim, T, H, W] -> [N, dim], N = B*T*H*W
        b, d, t, h, w = phi.shape
        return phi.permute(0, 2, 3, 4, 1).reshape(-1, d)

    @staticmethod
    def _flatten_target(y):
        # y: [B, T, H, W, out_dim] -> [N, out_dim]
        b, t, h, w, c = y.shape
        return y.reshape(-1, c)

    def ridge_update(self, y_true, phi=None):
        phi = phi if phi is not None else self.phi
        device, dtype = phi.device, phi.dtype

        Phi = self._flatten_phi(phi).to(dtype)                         # [N, d]
        Y = self._flatten_target(y_true).to(device=device, dtype=dtype)  # [N, out_dim]

        n, d = Phi.shape
        out_dim = Y.shape[1]

        # augment with a constant column so the bias is solved for jointly
        ones = torch.ones(n, 1, device=device, dtype=dtype)
        Phi_aug = torch.cat([Phi, ones], dim=1)                         # [N, d+1]

        # w0 [out_dim, d]->[d, out_dim]; b0->[1, out_dim]
        w0 = self.w0.view(out_dim, d).t().to(device=device, dtype=dtype)
        if self.b0 is not None:
            b0 = self.b0.view(1, out_dim).to(device=device, dtype=dtype)
        else:
            b0 = torch.zeros(1, out_dim, device=device, dtype=dtype)
        w0_aug = torch.cat([w0, b0], dim=0)                              # [d+1, out_dim]

        reg = torch.full((d + 1,), self.lam, device=device, dtype=dtype)
        reg[-1] = self.lam_bias
        R = torch.diag(reg)                                              # [d+1, d+1]

        A = Phi_aug.t() @ Phi_aug + R                                    # [d+1, d+1]
        rhs = Phi_aug.t() @ Y + R @ w0_aug                               # [d+1, out_dim]

        sol = torch.linalg.solve(A, rhs)                                 # [d+1, out_dim]

        new_w = sol[:-1, :].t().contiguous().view(out_dim, d, 1, 1, 1)
        new_b = sol[-1, :].contiguous()

        with torch.no_grad():
            self.last_layer.weight.copy_(new_w)
            if self.last_layer.bias is not None:
                self.last_layer.bias.copy_(new_b)
        self._captured_phi = None

    def reset_to_prior(self):
        with torch.no_grad():
            self.last_layer.weight.copy_(self.w0)
            if self.last_layer.bias is not None and self.b0 is not None:
                self.last_layer.bias.copy_(self.b0)
