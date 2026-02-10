# guide.py — PathGuide for LiDAR->TSDF costmap guidance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter  # 保留以防后续需要

from tsdf_cost_map import TsdfCostMap
from costmap_cfg import CostMapConfig


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


def bilinear_sample(cost_grid: torch.Tensor, norm_grid: torch.Tensor) -> torch.Tensor:
    """
    Differentiable bilinear sampler based on advanced indexing.

    Args:
        cost_grid: [N, 1, H, W]
        norm_grid: [N, T, 2] OR [N, 1, T, 2]
            last dim [x_norm, y_norm] in [-1, 1].
            NOTE: x_norm is along W (cols), y_norm along H (rows).

    Returns:
        sampled: [N, T] float tensor, differentiable w.r.t cost_grid & norm_grid.
    """
    if norm_grid.ndim == 4:
        # [N, 1, T, 2] -> [N, T, 2]
        norm_grid = norm_grid.squeeze(1)

    N, T, _ = norm_grid.shape
    _, _, H, W = cost_grid.shape
    device = cost_grid.device

    # extract normalized x,y
    x_norm = norm_grid[..., 0]  # [-1,1]
    y_norm = norm_grid[..., 1]  # [-1,1]

    # to pixel coordinates [0, W-1], [0, H-1]
    x = (x_norm + 1.0) * 0.5 * (W - 1)
    y = (y_norm + 1.0) * 0.5 * (H - 1)

    # corner indices
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    # clamp to valid range
    x0c = x0.clamp(0, W - 1)
    x1c = x1.clamp(0, W - 1)
    y0c = y0.clamp(0, H - 1)
    y1c = y1.clamp(0, H - 1)

    # float versions for weights
    x0f = x0c.to(dtype=torch.float32)
    x1f = x1c.to(dtype=torch.float32)
    y0f = y0c.to(dtype=torch.float32)
    y1f = y1c.to(dtype=torch.float32)

    # bilinear weights
    wa = (x1f - x) * (y1f - y)  # top-left
    wb = (x1f - x) * (y - y0f)  # bottom-left
    wc = (x - x0f) * (y1f - y)  # top-right
    wd = (x - x0f) * (y - y0f)  # bottom-right

    # squeeze cost grid to [N, H, W]
    cg = cost_grid.squeeze(1)

    # batch indices for advanced indexing
    batch_idx = torch.arange(N, device=device).view(N, 1).expand(N, T)

    vals_a = cg[batch_idx, y0c, x0c]  # [N, T]
    vals_b = cg[batch_idx, y1c, x0c]
    vals_c = cg[batch_idx, y0c, x1c]
    vals_d = cg[batch_idx, y1c, x1c]

    sampled = wa * vals_a + wb * vals_b + wc * vals_c + wd * vals_d
    return sampled  # [N, T]


class PathGuide:
    """
    PathGuide: 基于 LiDAR → TSDF costmap 的可微引导模块。

    用法约定：
      - 外部传入的是归一化的轨迹增量 trajs ∈ [-1,1]，shape [B, T, 2]
      - get_gradient() 内部会做一次 detach + requires_grad_(True)
      - goal_cost / collision_cost 均使用 autograd.grad 返回 w.r.t 该 leaf tensor 的梯度
    """

    def __init__(self, device, ACTION_STATS, guide_cfgs=None):
        self.device = device
        self.guide_cfgs = guide_cfgs
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.l1_loss = nn.L1Loss(reduction="mean")

        # 机器人几何参数
        self.robot_width = 0.6  # meters
        self.spatial_resolution = 0.1
        self.max_distance = 10.0
        self.bev_dist = self.max_distance / self.spatial_resolution

        # action 归一化区间
        self.delta_min = from_numpy(ACTION_STATS["min"]).to(self.device)
        self.delta_max = from_numpy(ACTION_STATS["max"]).to(self.device)

        # tsdf config and object
        self.tsdf_cfg = CostMapConfig()
        self.tsdf_cost_map = TsdfCostMap(
            self.tsdf_cfg.general,
            self.tsdf_cfg.tsdf_cost_map,
        )

        # cost_map is set via set_cost_map()
        self.cost_map = None

    # ------------------------------------------------------------------
    # 配置 TSDF cost map
    # ------------------------------------------------------------------
    def set_cost_map(self, cost_array: np.ndarray, start_x: float = 0.0, start_y: float = 0.0) -> None:
        """
        设置 TSDF 代价地图。

        Args:
            cost_array: numpy 2D array, shape [H, W]
            start_x, start_y: world frame 中地图左下角坐标
        """
        if cost_array is None:
            self.cost_map = None
            setattr(self.tsdf_cost_map, "is_map_ready", False)
            return

        if cost_array.ndim != 2:
            raise ValueError("cost_array must be 2D numpy array")

        # set metadata for Pos2Ind (H,W) = (num_x, num_y)
        self.tsdf_cost_map.start_x = float(start_x)
        self.tsdf_cost_map.start_y = float(start_y)
        self.tsdf_cost_map.num_x = int(cost_array.shape[0])
        self.tsdf_cost_map.num_y = int(cost_array.shape[1])
        self.tsdf_cost_map.is_map_ready = True

        # store cost_map as torch tensor on device (H,W)
        self.cost_map = torch.as_tensor(
            cost_array.astype(np.float32),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # 辅助函数：从归一化 delta 还原到 world 位置
    # ------------------------------------------------------------------
    def _norm_delta_to_ori_trajs(self, trajs: torch.Tensor) -> torch.Tensor:
        """
        trajs: [B, T, 2] in [-1,1] (normalized deltas)
        returns: [B, T, 2] world deltas accumulated as positions
        """
        delta_tmp = (trajs + 1.0) / 2.0  # [0,1]
        delta_ori = delta_tmp * (self.delta_max - self.delta_min) + self.delta_min
        trajs_ori = delta_ori.cumsum(dim=1)
        return trajs_ori

    # ------------------------------------------------------------------
    # 目标代价：让轨迹终点靠近 goal
    # ------------------------------------------------------------------
    def goal_cost(
        self,
        trajs_in: torch.Tensor,
        goal: torch.Tensor,
        scale_factor: float = None,
    ) -> torch.Tensor:
        """
        计算 goal 导向的梯度（不调用 backward，只返回 grad）。

        Args:
            trajs_in: [B, T, 2], requires_grad=True
            goal: [2] or [B,2] world 坐标
        Returns:
            grad: [B, T, 2]
        """
        trajs_ori = self._norm_delta_to_ori_trajs(trajs_in)
        if scale_factor is not None:
            trajs_ori = trajs_ori * scale_factor

        trajs_end_positions = trajs_ori[:, -1, :]  # [B,2]

        if goal.dim() == 1:
            # [2] -> [B,2]
            goal = goal.unsqueeze(0).expand_as(trajs_end_positions)

        distances = torch.norm(goal - trajs_end_positions, dim=1)  # [B]
        gloss = 0.05 * distances.sum()

        grad = torch.autograd.grad(
            outputs=gloss,
            inputs=trajs_in,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]

        if grad is None:
            grad = torch.zeros_like(trajs_in)

        return grad

    # ------------------------------------------------------------------
    # 时间尺度加权（越接近终点权重越大）
    # ------------------------------------------------------------------
    def generate_scale(self, n: int) -> torch.Tensor:
        """
        生成沿时间步的线性权重 [0..1]，用于强调晚期步骤的梯度。
        """
        scale = torch.linspace(0.0, 1.0, steps=n, device=self.device)
        return scale

    # ------------------------------------------------------------------
    # 机器人体积膨胀：生成左/中/右三条轨迹
    # ------------------------------------------------------------------
    def add_robot_dim(self, world_ps: torch.Tensor) -> torch.Tensor:
        """
        world_ps: [B, T, C], xy in [:,:,0:2]
        returns: [3*B, T-1, C'] (three tracks per original sample). If C>2, pad remaining dims.
        """
        B, T, C = world_ps.shape
        if T < 2:
            return world_ps.new_empty((0, 0, C))

        # 切线方向
        tangent = world_ps[:, 1:, 0:2] - world_ps[:, :-1, 0:2]  # [B, T-1, 2]
        eps = 1e-6
        tnorm = torch.norm(tangent, dim=2, keepdim=True).clamp(min=eps)
        tangent = tangent / tnorm

        # 法线方向（左/右）
        normals = tangent[:, :, [1, 0]] * torch.tensor(
            [-1.0, 1.0], dtype=torch.float32, device=world_ps.device
        )

        centers = world_ps[:, :-1, 0:2]  # [B, T-1, 2]
        left = centers + normals * (self.robot_width / 2.0)
        right = centers - normals * (self.robot_width / 2.0)

        # stack left, center, right across new batch
        stacked = torch.cat(
            [left.unsqueeze(1), centers.unsqueeze(1), right.unsqueeze(1)],
            dim=1,
        )  # [B,3,T-1,2]
        stacked = (
            stacked.permute(1, 0, 2, 3)
            .contiguous()
            .view(3 * B, T - 1, 2)
        )  # [3B, T-1, 2]

        # if original had extra dims (z etc), attach them from centers
        if C > 2:
            extra = (
                world_ps[:, :-1, 2:]
                .unsqueeze(1)
                .expand(B, 3, T - 1, C - 2)
                .permute(1, 0, 2, 3)
                .contiguous()
                .view(3 * B, T - 1, C - 2)
            )
            full = torch.cat([stacked, extra], dim=2)
            return full

        return stacked

    # ------------------------------------------------------------------
    # 碰撞代价：TSDF cost map 引导
    # ------------------------------------------------------------------
    def collision_cost(
        self,
        trajs_in: torch.Tensor,
        scale_factor: float = None,
    ):
        """
        Args:
            trajs_in: [B, T, 2] normalized deltas (-1..1), requires_grad=True
        Returns:
            scaled_grads: [B, T, 2]
            cost_list:   [B] or None
        """
        if self.cost_map is None or not getattr(self.tsdf_cost_map, "is_map_ready", False):
            # 没有地图就返回 0 梯度
            return torch.zeros_like(trajs_in), None

        device = self.device
        if trajs_in.device != device:
            raise RuntimeError(f"trajs_in.device={trajs_in.device}, but PathGuide.device={device}")
        if not (trajs_in.dim() == 3 and trajs_in.size(-1) >= 2):
            raise RuntimeError(f"trajs_in shape expected [B,T,>=2], got {tuple(trajs_in.shape)}")

        B, T, C = trajs_in.shape

        # convert to original-space positions and optionally scale
        trajs_ori = self._norm_delta_to_ori_trajs(trajs_in)  # [B, T, 2]
        if scale_factor is not None:
            trajs_ori = trajs_ori * scale_factor

        # inflate for robot width -> [N=3*B, T-1, C]
        trajs_inflated = self.add_robot_dim(trajs_ori)
        if trajs_inflated.numel() == 0:
            return torch.zeros_like(trajs_in), None

        # get normalized indices (torch) and H_np (unused)
        norm_inds, _ = self.tsdf_cost_map.Pos2Ind(trajs_inflated)  # [N, T-1, 2]
        if not isinstance(norm_inds, torch.Tensor):
            raise RuntimeError("Pos2Ind must return torch.Tensor in torch branch")
        if not norm_inds.requires_grad:
            raise RuntimeError("norm_inds.requires_grad is False — Pos2Ind must keep grad path from trajs_inflated.")

        # prepare cost grid [N, 1, H, W]
        if self.cost_map is None:
            return torch.zeros_like(trajs_in), None
        cost_grid_single = self.cost_map.T.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        N = norm_inds.shape[0]
        cost_grid = cost_grid_single.expand(N, -1, -1, -1).contiguous()  # [N,1,H,W]

        if not (cost_grid.device == trajs_in.device == norm_inds.device):
            raise RuntimeError("device mismatch among cost_grid, trajs_in, norm_inds.")

        # call bilinear_sample (works with norm grid shape [N, T, 2] or [N,1,T,2])
        try:
            sampled = bilinear_sample(cost_grid, norm_inds)  # [N, T-1]
        except Exception as e:
            print("[PathGuide] bilinear_sample failed:", e)
            return torch.zeros_like(trajs_in), None

        # sample loss and reduce along time
        loss = 0.003 * sampled.sum(dim=1)  # [N]

        # compute gradients wrt trajs_in using autograd.grad
        try:
            grad_out = torch.ones_like(loss, device=device)
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=trajs_in,
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
        except Exception as e:
            print("[PathGuide] autograd.grad raised exception:", e)
            grads = None

        if grads is None:
            grads = torch.zeros_like(trajs_in)

        # collapse loss from 3*B -> B by averaging the three tracks (if shape matches), else fallback
        if loss.numel() == 3 * B:
            cost_list = loss.view(3, B).mean(dim=0)
        else:
            # 尝试取中间那条作为代表；如果也不够就直接返回全部/部分
            cost_list = loss[1::3] if loss.numel() >= 2 * B else loss[:B]

        # 时间权重：越靠后的时间步权重越大
        generate_scale = self.generate_scale(T).to(device)  # [T]
        scaled_grads = generate_scale.view(1, T, 1) * grads  # [B,T,2]

        return scaled_grads, cost_list

    # ------------------------------------------------------------------
    # 对外主接口：返回 w.r.t trajs 的梯度
    # ------------------------------------------------------------------
    def get_gradient(
        self,
        trajs: torch.Tensor,
        alpha: float = 0.3,
        t: int = None,
        goal_pos=None,
        ACTION_STATS=None,
        scale_factor: float = None,
    ):
        """
        Args:
            trajs: [B, T, 2] normalized deltas (-1..1)，来自上层 diffusion / policy
        Returns:
            grad: [B, T, 2]
            cost_list: [B] or None
        """
        # 在这里统一 detach & requires_grad，确保是 leaf tensor
        trajs_in = trajs.detach().to(self.device).requires_grad_(True)

        if goal_pos is not None:
            goal_t = torch.as_tensor(
                goal_pos,
                dtype=torch.float32,
                device=self.device,
            )
            grad_goal = self.goal_cost(trajs_in, goal_t, scale_factor=scale_factor)
            return grad_goal, None
        else:
            collision_grad, cost_list = self.collision_cost(
                trajs_in,
                scale_factor=scale_factor,
            )
            return collision_grad, cost_list
