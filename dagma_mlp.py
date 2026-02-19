"""
一个精简版的 DAGMA-MLP 结构学习实现，用于和 LimiX 产生的先验结合。

注意：这里不是论文的完整复现，而是遵循你给出的目标函数形式的
PyTorch 参考实现，方便你后续根据需要继续扩展。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from limix_interface import LimixConstraints


def acyclicity_constraint(A: torch.Tensor) -> torch.Tensor:
    """
    DAGMA 中常用的无环约束 h(A) ≈ trace(exp(A ⊙ A)) - d。
    """
    d = A.size(0)
    expm = torch.matrix_exp(A * A)
    return torch.trace(expm) - d


def nll_gaussian(X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
    """
    简单的高斯负对数似然（等价于均方误差）。
    """
    return 0.5 * torch.mean((X - X_hat) ** 2)


class DagmaDecoderMLP(nn.Module):
    """
    用于建模 X_j = f_j(Parents_j) 的 MLP。

    实现上我们采用共享的两层 MLP，然后通过邻接矩阵 A
    在输入维度上进行线性混合。
    """

    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.d = d
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # 先按 A 做线性混合：X_tilde = X @ A^T
        X_tilde = X @ A.t()
        h = self.activation(self.fc1(X_tilde))
        X_hat = self.fc2(h)
        return X_hat


@dataclass
class DagmaHyperParams:
    lambda_l1: float = 0.01
    lambda_h: float = 10.0
    lambda_black: float = 50.0  # 增加黑名单惩罚权重，确保硬约束被严格遵守
    lambda_white: float = 5.0
    tau_white: float = 0.3
    lambda_pref: float = 1.0
    lambda_group: float = 1.0
    lr: float = 1e-3
    steps: int = 10000


class DagmaMLP:
    """
    带有 LimiX 先验正则项的 DAGMA-MLP。
    """

    def __init__(self, d: int, limix: LimixConstraints, hparams: DagmaHyperParams):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d = d
        self.hparams = hparams
        self.model = DagmaDecoderMLP(d).to(self.device)

        # 邻接矩阵 A 作为需要学习的参数
        self.A = nn.Parameter(torch.zeros(d, d, device=self.device))

        # 处理 LimiX 约束，转成 tensor / 索引
        self.black_edges_idx: List[Tuple[int, int]] = []
        self.white_edges_idx: List[Tuple[int, int]] = []
        name_to_idx = {name: i for i, name in enumerate(limix.var_names)}

        for u, v in limix.blacklist:
            if u in name_to_idx and v in name_to_idx:
                self.black_edges_idx.append((name_to_idx[u], name_to_idx[v]))
        for u, v in limix.whitelist:
            if u in name_to_idx and v in name_to_idx:
                self.white_edges_idx.append((name_to_idx[u], name_to_idx[v]))

        self.edge_pref = torch.tensor(
            limix.edge_pref, dtype=torch.float32, device=self.device
        )
        self.groups = limix.groups  # 已经是 (i, j) 索引列表

        self.optimizer = optim.Adam(
            list(self.model.parameters()) + [self.A], lr=hparams.lr
        )

    def loss(self, X: torch.Tensor) -> torch.Tensor:
        X_hat = self.model(X, self.A)
        nll = nll_gaussian(X, X_hat)

        # L1 稀疏
        l1 = torch.sum(torch.abs(self.A))

        # 无环约束
        h_val = acyclicity_constraint(self.A)

        # 黑名单惩罚：黑名单边的 |A_ij|
        black_penalty = torch.zeros(1, device=self.device)
        if self.black_edges_idx:
            idx_i = torch.tensor([i for i, _ in self.black_edges_idx], device=self.device)
            idx_j = torch.tensor([j for _, j in self.black_edges_idx], device=self.device)
            black_penalty = torch.sum(torch.abs(self.A[idx_i, idx_j]))

        # 白名单奖励：max(0, tau - A_ij)
        white_penalty = torch.zeros(1, device=self.device)
        if self.white_edges_idx:
            idx_i = torch.tensor([i for i, _ in self.white_edges_idx], device=self.device)
            idx_j = torch.tensor([j for _, j in self.white_edges_idx], device=self.device)
            white_penalty = torch.sum(
                torch.relu(self.hparams.tau_white - self.A[idx_i, idx_j])
            )

        # 边偏好权重
        pref_penalty = torch.sum(
            torch.abs(self.edge_pref * self.A)
        )

        # 组/层级稀疏：对每一组边的 L2 范数求和
        group_penalty = torch.zeros(1, device=self.device)
        for group in self.groups:
            if not group:
                continue
            vals = torch.stack([self.A[i, j] for (i, j) in group])
            group_penalty = group_penalty + torch.norm(vals, p=2)

        hp = self.hparams
        total = (
            nll
            + hp.lambda_l1 * l1
            + hp.lambda_h * h_val
            + hp.lambda_black * black_penalty
            + hp.lambda_white * white_penalty
            + hp.lambda_pref * pref_penalty
            + hp.lambda_group * group_penalty
        )
        return total

    def fit(self, X_np: np.ndarray) -> np.ndarray:
        """
        训练 DAGMA-MLP，返回学习到的邻接矩阵 A（numpy 数组）。
        """
        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        for step in range(self.hparams.steps):
            self.optimizer.zero_grad()
            loss_val = self.loss(X)
            loss_val.backward()
            self.optimizer.step()

            # 可选：每隔一段打印一次损失，方便调试
            if (step + 1) % 500 == 0:
                print(f"[DAGMA] step {step+1}/{self.hparams.steps}, loss={loss_val.item():.4f}")

        with torch.no_grad():
            A_learned = self.A.clamp(min=0.0).cpu().numpy()
        
        # 强制应用黑名单约束：将黑名单中的边置零（硬约束必须严格遵守）
        if self.black_edges_idx:
            for i, j in self.black_edges_idx:
                A_learned[i, j] = 0.0
            print(f"[DAGMA] ✅ 已强制将 {len(self.black_edges_idx)} 条黑名单边置零（硬约束）")
        
        # 强制应用白名单约束：确保白名单中的边存在（硬约束必须严格遵守）
        if self.white_edges_idx:
            white_count = 0
            for i, j in self.white_edges_idx:
                if A_learned[i, j] < self.hparams.tau_white:
                    # 如果白名单边的权重小于阈值，强制设置为阈值
                    A_learned[i, j] = self.hparams.tau_white
                    white_count += 1
            if white_count > 0:
                print(f"[DAGMA] ✅ 已强制确保 {white_count} 条白名单边存在（硬约束，权重 >= {self.hparams.tau_white}）")
        
        return A_learned


__all__ = ["DagmaHyperParams", "DagmaMLP"]


