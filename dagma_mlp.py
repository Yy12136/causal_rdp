"""
联合优化模块：CRWM (Causal Reward Weight Matrix)

优化目标（对每个 (s_{t-1}, s_t) 对）：
  M_inst(s_{t-1}) = M_inv ⊙ (1 + tanh(M_trans(s_{t-1}; θ)))
  L_MSE  = ||s_t - M_inst(s_{t-1}) @ s_{t-1}||²   (仅在 active_mask = 1 的维度)
  L_soft = γ ||M_conf ⊙ (M_inv - M_prior)||_F²
  L_alm  = L_MSE + L_soft + λ·h(M_inv) + (ρ/2)·h(M_inv)²   [ALM 内层]
  h(M)   = -log det(αI - M⊙M) + d log α   [DAGMA log-det 无环约束]

ALM 外层：λ ← λ + ρ·h(M_inv)；ρ 每轮增长。
最终只输出 M_inv_star 作为 CRWM；M_trans 训练辅助，不进入最终因果图。

符号约定：
  d         : 全局变量维度
  M_inv     : (d, d)，不变因果矩阵，clamp(≥0) 后输出
  M_trans   : s_{t-1} (d,) → delta (d, d)，tanh 约束输出在 (-1, 1)
  M_prior   : (d, d)，LimiX 先验因果矩阵
  M_conf    : (d, d)，M_prior 的置信度，[0, 1]
  active_mask: (B, d)，binary，1 = 该维度在当前 episode 中激活（有真实数据）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from limix_interface import LimixConstraints


# ---------------------------------------------------------------------------
# 无环约束（DAGMA log-det）
# ---------------------------------------------------------------------------

def acyclicity_constraint(M: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """DAGMA log-det 无环约束 h(M) = -log det(alpha I - M ⊙ M) + d log alpha，仅作用于 M_inv。"""
    d = M.size(0)
    alpha_t = torch.tensor(alpha, device=M.device, dtype=M.dtype)
    B = alpha_t * torch.eye(d, device=M.device, dtype=M.dtype) - M * M
    sign, logabsdet = torch.linalg.slogdet(B)
    h = -logabsdet + d * torch.log(alpha_t)
    if sign <= 0:
        h = torch.tensor(1e6, device=M.device, dtype=M.dtype)
    domain_penalty = torch.relu(1e-6 - torch.linalg.eigvalsh(B).min()) * 1e3
    return torch.nan_to_num(h + domain_penalty, nan=1e6, posinf=1e6, neginf=1e6)


# ---------------------------------------------------------------------------
# 瞬态调制网络 M_trans
# ---------------------------------------------------------------------------

class TransientNetwork(nn.Module):
    """
    M_trans: s_{t-1} (d,) → delta (d, d)，输出经 tanh 约束在 (-1, 1)。
    保证 1 + delta ∈ (0, 2)，不改变 M_inv 的符号。
    注意：输出维度为 d×d，d 较大时参数量大，可通过 hidden_dim 控制。
    """

    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, d * d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d) → delta: (B, d, d)
        B = x.size(0)
        return torch.tanh(self.net(x)).view(B, self.d, self.d)


# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------

@dataclass
class CRWMHyperParams:
    # Loss 权重
    gamma: float = 1.0          # L_soft 权重
    lambda_black: float = 50.0  # 黑名单硬约束惩罚
    lambda_white: float = 5.0   # 白名单硬约束惩罚
    tau_white: float = 0.3      # 白名单边最小权重

    # ALM 超参数（自适应增广拉格朗日）
    lambda_init: float = 0.1    # 初始 ALM 乘子 λ
    rho_init: float = 1.0       # 初始增广惩罚系数 ρ
    rho_growth: float = 2.0     # 每轮外层 ρ 增长因子
    rho_max: float = 1e4        # ρ 上限

    # DAGMA 无环约束
    alpha: float = 1.0          # log-det 约束中的 alpha I 系数

    # 训练控制
    outer_iters: int = 10       # ALM 外层迭代次数
    inner_steps: int = 1000     # 每次内层优化步数
    lr: float = 1e-3
    batch_size: int = 256
    hidden_dim: int = 32        # M_trans 隐藏层维度（越大越耗显存）


# ---------------------------------------------------------------------------
# CRWM 联合优化器
# ---------------------------------------------------------------------------

class CRWMOptimizer:
    """
    联合优化器：同时学习不变因果矩阵 M_inv 和瞬态调制网络 M_trans。
    训练结束后只输出 M_inv_star（CRWM），M_trans 丢弃。
    """

    def __init__(
        self,
        d: int,
        limix: LimixConstraints,
        hparams: CRWMHyperParams | None = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d = d
        self.hparams = hparams or CRWMHyperParams()

        # 可学习参数
        self.M_inv = nn.Parameter(torch.zeros(d, d, device=self.device))
        self.M_trans = TransientNetwork(d, hidden=self.hparams.hidden_dim).to(self.device)

        # LimiX 先验（固定，不参与梯度）
        self.M_prior = torch.tensor(
            limix.M_prior, dtype=torch.float32, device=self.device
        )
        self.M_conf = torch.tensor(
            limix.M_conf, dtype=torch.float32, device=self.device
        )

        # 硬约束边集合（索引形式）
        name_to_idx = {name: i for i, name in enumerate(limix.var_names)}
        # 约定 M[target, source]：u -> v 对应矩阵索引 [v_idx, u_idx]
        self.black_edges: List[Tuple[int, int]] = [
            (name_to_idx[v], name_to_idx[u])
            for u, v in limix.blacklist
            if u in name_to_idx and v in name_to_idx
        ]
        self.white_edges: List[Tuple[int, int]] = [
            (name_to_idx[v], name_to_idx[u])
            for u, v in limix.whitelist
            if u in name_to_idx and v in name_to_idx
        ]

        # ALM 状态（外层循环自适应更新）
        self.lam = float(self.hparams.lambda_init)
        self.rho = float(self.hparams.rho_init)

    def _make_optimizer(self) -> optim.Adam:
        return optim.Adam(
            [self.M_inv] + list(self.M_trans.parameters()),
            lr=self.hparams.lr,
        )

    def _inner_loss(
        self,
        s_prev: torch.Tensor,       # (B, d)
        s_curr: torch.Tensor,       # (B, d)
        active_mask: torch.Tensor,  # (B, d)，binary
    ) -> torch.Tensor:
        hp = self.hparams

        # --- M_inst = M_inv ⊙ (1 + tanh(M_trans(s_prev))) ---
        delta = self.M_trans(s_prev)                         # (B, d, d), ∈ (-1, 1)
        M_inst = self.M_inv.unsqueeze(0) * (1.0 + delta)    # (B, d, d)

        # --- L_MSE：标准矩阵向量积，只在 active 维度上计算误差 ---
        # s_hat_i = Σ_j M_inst_{i,j}(s_{t-1}) · s_{t-1,j}
        s_hat = torch.einsum("bij,bj->bi", M_inst, s_prev)  # (B, d)
        residual = (s_curr - s_hat) * active_mask
        n_active = active_mask.sum().clamp(min=1.0)
        L_MSE = (residual ** 2).sum() / n_active

        # --- L_soft = γ ||M_conf ⊙ (M_inv - M_prior)||_F² ---
        L_soft = hp.gamma * (self.M_conf * (self.M_inv - self.M_prior)).pow(2).sum()

        # --- 硬约束惩罚（黑名单）---
        black_penalty = torch.zeros(1, device=self.device)
        if self.black_edges:
            bi = torch.tensor([i for i, _ in self.black_edges], device=self.device)
            bj = torch.tensor([j for _, j in self.black_edges], device=self.device)
            black_penalty = self.M_inv[bi, bj].abs().sum()

        # --- 无环约束 h(M_inv)（只作用在 M_inv，不含 M_trans）---
        h = acyclicity_constraint(self.M_inv, alpha=hp.alpha)

        # --- ALM 总损失（内层，λ 和 ρ 固定）---
        L_total = (
            L_MSE
            + L_soft
            + hp.lambda_black * black_penalty
            + self.lam * h
            + (self.rho / 2.0) * h ** 2
        )
        return L_total

    def fit(
        self,
        s_prev: np.ndarray,      # (N, d)，Dmicro s_{t-1}
        s_curr: np.ndarray,      # (N, d)，Dmicro s_t
        active_mask: np.ndarray, # (N, d)，binary
    ) -> np.ndarray:
        """
        训练 CRWM，返回学习到的 M_inv_star（numpy 数组，d×d）。
        使用 ALM 外层自适应更新 λ；M_trans 只在内层优化，最终不输出。
        """
        sp_all = torch.tensor(s_prev, dtype=torch.float32)
        sc_all = torch.tensor(s_curr, dtype=torch.float32)
        mk_all = torch.tensor(active_mask, dtype=torch.float32)
        N = sp_all.size(0)
        batch_size = min(self.hparams.batch_size, N)
        hp = self.hparams

        print(f"[CRWM] 开始训练: N={N}, d={self.d}, device={self.device}")
        print(f"[CRWM] ALM: outer={hp.outer_iters}, inner={hp.inner_steps}, "
              f"λ_init={hp.lambda_init}, ρ_init={hp.rho_init}")

        for outer in range(hp.outer_iters):
            optimizer = self._make_optimizer()

            for step in range(hp.inner_steps):
                idx = torch.randint(0, N, (batch_size,))
                loss = self._inner_loss(
                    sp_all[idx].to(self.device),
                    sc_all[idx].to(self.device),
                    mk_all[idx].to(self.device),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step + 1) % 200 == 0:
                    with torch.no_grad():
                        h_val = acyclicity_constraint(self.M_inv, alpha=hp.alpha).item()
                    print(
                        f"[CRWM] outer={outer+1}/{hp.outer_iters} "
                        f"inner={step+1}/{hp.inner_steps} "
                        f"loss={loss.item():.4f} h={h_val:.6f} "
                        f"λ={self.lam:.4f} ρ={self.rho:.2f}"
                    )

            # ALM 外层：λ ← λ + ρ·h(M_inv)；增长 ρ
            with torch.no_grad():
                h_scalar = acyclicity_constraint(self.M_inv, alpha=hp.alpha).item()
            self.lam = self.lam + self.rho * h_scalar
            self.rho = min(self.rho * hp.rho_growth, hp.rho_max)
            print(
                f"[CRWM] 外层第 {outer+1} 轮结束: "
                f"h={h_scalar:.6f}, λ←{self.lam:.4f}, ρ←{self.rho:.2f}"
            )

        # 提取 M_inv_star，clamp ≥ 0，强制应用硬约束
        with torch.no_grad():
            M_inv_np = self.M_inv.clamp(min=0.0).cpu().numpy()

        for i, j in self.black_edges:
            M_inv_np[i, j] = 0.0
        if self.black_edges:
            print(f"[CRWM] ✅ 已强制将 {len(self.black_edges)} 条黑名单边置零")

        return M_inv_np


# ---------------------------------------------------------------------------
# 向后兼容别名（旧导入不报错）
# ---------------------------------------------------------------------------

DagmaHyperParams = CRWMHyperParams
DagmaMLP = CRWMOptimizer


__all__ = [
    "acyclicity_constraint",
    "TransientNetwork",
    "CRWMHyperParams",
    "CRWMOptimizer",
    "DagmaHyperParams",
    "DagmaMLP",
]
