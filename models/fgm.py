import torch
from torch import nn

class FGM:
    """Fast Gradient Method cho adversarial training trên embedding layer."""
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    # def attack(self, emb_name='embedding'):
    #     """Thêm nhiễu đối kháng vào embedding."""
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad and emb_name in name and param.grad is not None:
    #             self.backup[name] = param.data.clone()
    #             norm = torch.norm(param.grad)
    #             if norm != 0:
    #                 r_at = self.epsilon * param.grad / norm
    #                 param.data.add_(r_at)
    def attack(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                # print(f"grad norm: {norm:.6f} | param norm: {torch.norm(param.data):.6f} | ratio: {norm/torch.norm(param.data):.6f}")
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        """Khôi phục embedding về giá trị ban đầu."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}