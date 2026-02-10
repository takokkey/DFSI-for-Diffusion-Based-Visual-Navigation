# tsdf_cost_map.py
import math
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import torch
from costmap_cfg import GeneralCostMapConfig, TsdfCostMapConfig

class TsdfCostMap:
    def __init__(self, cfg_general: GeneralCostMapConfig, cfg_tsdf: TsdfCostMapConfig):
        self._cfg_general = cfg_general
        self._cfg_tsdf = cfg_tsdf
        self.is_map_ready = False

        # keep both pcd and raw arrays for compat
        self.obs_pcd = o3d.geometry.PointCloud()
        self.free_pcd = o3d.geometry.PointCloud()
        self.obs_points = np.zeros((0, 3), dtype=np.float32)
        self.free_points = np.zeros((0, 3), dtype=np.float32)

        # metadata defaults (will be set by UpdateMapParams or external set)
        self.start_x = 0.0
        self.start_y = 0.0
        self.num_x = 1
        self.num_y = 1

    def Pos2Ind(self, points):
        """
        world coordinates -> normalized indices (for grid sampling) + grid coords H (float)
        - If input is torch.Tensor: perform pure torch ops so that resulting norm_H keeps grad_fn
        - If numpy or None: produce safe outputs; returns (torch.tensor(NH), H_np)
        Returns:
            norm_H_tensor: torch.FloatTensor shape [..., 2], values in [-1,1]
            H_np: numpy array with grid indices in float (for indexing/debug) or empty array
        """
        res = float(self._cfg_general.resolution)

        # handle empty / None
        if points is None:
            empty_t = torch.empty((0, 2), dtype=torch.float32)
            return empty_t.requires_grad_(True), np.empty((0, 2), dtype=np.float32)

        # torch branch: keep graph
        if isinstance(points, torch.Tensor):
            if points.numel() == 0:
                empty_t = torch.empty((0, 2), dtype=torch.float32, device=points.device)
                return empty_t.requires_grad_(True), np.empty((0, 2), dtype=np.float32)

            pts = points[..., :2].to(dtype=torch.float32)  # [...,2]
            device = pts.device

            # start as tensor same device
            start_xy_t = torch.tensor([getattr(self, "start_x", 0.0), getattr(self, "start_y", 0.0)],
                                      dtype=torch.float32, device=device)

            # world -> grid index (float)  H_t (keeps grad from pts)
            H_t = (pts - start_xy_t) / res  # [...,2]

            # denom (N-1) as tensor (avoid zero)
            nx = max(int(getattr(self, "num_x", 1)) - 1, 1)
            ny = max(int(getattr(self, "num_y", 1)) - 1, 1)
            denom = torch.tensor([float(nx), float(ny)], dtype=torch.float32, device=device)

            # normalized indices to [-1,1] for grid_sample: 2*(index/(N-1) - 0.5)
            norm_H_t = 2.0 * (H_t / denom - 0.5)  # [...,2] -> has grad_fn if pts requires_grad

            H_np = H_t.detach().cpu().numpy()
            return norm_H_t, H_np

        # numpy branch
        pts_np = np.asarray(points)
        if pts_np.size == 0:
            return torch.empty((0, 2), dtype=torch.float32), np.empty((0, 2), dtype=np.float32)

        start_xy = np.array([getattr(self, "start_x", 0.0), getattr(self, "start_y", 0.0)], dtype=np.float32)
        H = (pts_np[..., :2] - start_xy) / res
        nx = max(int(getattr(self, "num_x", 1)) - 1, 1)
        ny = max(int(getattr(self, "num_y", 1)) - 1, 1)
        denom = np.array([float(nx), float(ny)], dtype=np.float32)
        NH = 2.0 * (H / denom - 0.5)
        return torch.from_numpy(NH.astype(np.float32)), H

    def UpdatePoints(self, obs_pts, free_pts, downsample=True):
        """
        Update internal point lists. Accepts numpy arrays (N,3) or empty.
        Returns stored numpy arrays.
        """
        def proc(points, factor=1.0):
            if points is None:
                return np.zeros((0, 3), dtype=np.float32)
            pts = np.asarray(points, dtype=np.float32)
            if pts.size == 0:
                return np.zeros((0, 3), dtype=np.float32)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            if downsample:
                try:
                    pcd = pcd.voxel_down_sample(self._cfg_general.resolution * factor)
                except Exception:
                    pass
            out = np.asarray(pcd.points, dtype=np.float32)
            if out.size == 0:
                return np.zeros((0, 3), dtype=np.float32)
            return out

        self.obs_points = proc(obs_pts, factor=1.0)
        self.free_points = proc(free_pts, factor=0.85)
        # update pcd objects for compatibility
        try:
            self.obs_pcd.points = o3d.utility.Vector3dVector(self.obs_points)
            self.free_pcd.points = o3d.utility.Vector3dVector(self.free_points)
        except Exception:
            pass
        # debug print
        # print("number of obs points: %d, free points: %d" % (self.obs_points.shape[0], self.free_points.shape[0]))
        return self.obs_points, self.free_points

    def TerrainAnalysis(self, input_points):
        """
        Split raw points into obs and free based on height thresholds.
        Works with empty input.
        """
        if input_points is None:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        pts = np.asarray(input_points, dtype=np.float32)
        if pts.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        obs_list = []
        free_list = []
        for p in pts:
            p_height = p[2] + self._cfg_tsdf.offset_z
            if (p_height > self._cfg_tsdf.ground_height * 1.0) and (p_height < self._cfg_tsdf.robot_height * self._cfg_tsdf.robot_height_factor):
                obs_list.append(p)
            elif p_height < self._cfg_tsdf.ground_height:
                free_list.append(p)
        obs = np.array(obs_list, dtype=np.float32) if len(obs_list) else np.zeros((0, 3), dtype=np.float32)
        free = np.array(free_list, dtype=np.float32) if len(free_list) else np.zeros((0, 3), dtype=np.float32)
        return obs, free

    def LoadPointCloud(self, pcd):
        """
        pcd: open3d.geometry.PointCloud (or None)
        """
        pts = np.asarray(pcd.points) if (pcd is not None and hasattr(pcd, "points")) else np.zeros((0, 3))
        obs_p, free_p = self.TerrainAnalysis(pts)
        # optional outlier filter
        if self._cfg_tsdf.filter_outliers:
            obs_p = self.FilterCloud(obs_p)
            free_p = self.FilterCloud(free_p)
        self.UpdatePoints(obs_p, free_p, downsample=True)
        self.UpdateMapParams()

    def UpdateMapParams(self):
        """
        compute grid extents from obs_points (if any)
        """
        if self.obs_points.shape[0] == 0:
            self.is_map_ready = False
            return
        max_xyz = np.amax(self.obs_points, axis=0) + self._cfg_general.clear_dist
        min_xyz = np.amin(self.obs_points, axis=0) - self._cfg_general.clear_dist
        self.num_x = int(np.ceil((max_xyz[0] - min_xyz[0]) / self._cfg_general.resolution / 10) * 10)
        self.num_y = int(np.ceil((max_xyz[1] - min_xyz[1]) / self._cfg_general.resolution / 10) * 10)
        self.start_x = (max_xyz[0] + min_xyz[0]) / 2.0 - self.num_x / 2.0 * self._cfg_general.resolution
        self.start_y = (max_xyz[1] + min_xyz[1]) / 2.0 - self.num_y / 2.0 * self._cfg_general.resolution
        # debug
        # print("tsdf map initialized, with size: %d, %d" % (self.num_x, self.num_y))
        self.is_map_ready = True

    def zero_single_isolated_ones(self, arr):
        neighbors = ndimage.convolve(arr, weights=np.ones((3, 3)), mode="constant", cval=0)
        isolated_ones = (arr == 1) & (neighbors == 1)
        arr[isolated_ones] = 0
        return arr

    def CreateTSDFMap(self):
        if not self.is_map_ready:
            raise ValueError("create tsdf map fails, no points received.")
        free_map = np.ones([self.num_x, self.num_y], dtype=np.float32)
        obs_map = np.zeros([self.num_x, self.num_y], dtype=np.float32)

        free_I = self.IndexArrayOfPs(self.free_points)
        obs_I = self.IndexArrayOfPs(self.obs_points)

        for i in obs_I:
            obs_map[i[0], i[1]] = 1.0
        obs_map = self.zero_single_isolated_ones(obs_map)
        obs_map = gaussian_filter(obs_map, sigma=self._cfg_tsdf.sigma_expand)

        for i in free_I:
            xi, yi = int(i[0]), int(i[1])
            if 0 <= xi < self.num_x and 0 <= yi < self.num_y:
                free_map[xi, yi] = 0.0
        free_map = gaussian_filter(free_map, sigma=self._cfg_tsdf.sigma_expand)
        free_map[free_map < self._cfg_tsdf.free_space_threshold] = 0
        free_map[obs_map > self._cfg_tsdf.obstacle_threshold] = 1.0

        tsdf_array = ndimage.distance_transform_edt(free_map)
        tsdf_array[tsdf_array > 0.0] = np.log(tsdf_array[tsdf_array > 0.0] + math.e)
        tsdf_array = gaussian_filter(tsdf_array, sigma=self._cfg_general.sigma_smooth)
        viz_points = np.concatenate((self.obs_points, self.free_points), axis=0) if (self.obs_points.size or self.free_points.size) else np.zeros((0,3))
        ground_array = np.zeros([self.num_x, self.num_y], dtype=np.float32)
        return [tsdf_array, viz_points, ground_array], [float(self.start_x), float(self.start_y)]

    def IndexArrayOfPs(self, points):
        """
        points: numpy array (N,3) or empty -> returns integer indices array (M,2) within grid bounds
        """
        if points is None:
            return np.zeros((0, 2), dtype=int)
        pts = np.asarray(points)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=int)
        idx = np.round((pts[:, :2] - np.array([self.start_x, self.start_y])) / self._cfg_general.resolution).astype(int)
        mask = (idx[:, 0] >= 0) & (idx[:, 0] < self.num_x) & (idx[:, 1] >= 0) & (idx[:, 1] < self.num_y)
        return idx[mask].astype(int)

    def FilterCloud(self, points, outlier_filter=True):
        if points is None:
            return np.zeros((0, 3), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if any([self._cfg_general.x_max, self._cfg_general.x_min, self._cfg_general.y_max, self._cfg_general.y_min]):
            points_x_idx_upper = (pts[:, 0] < self._cfg_general.x_max) if self._cfg_general.x_max is not None else np.ones(pts.shape[0], dtype=bool)
            points_x_idx_lower = (pts[:, 0] > self._cfg_general.x_min) if self._cfg_general.x_min is not None else np.ones(pts.shape[0], dtype=bool)
            points_y_idx_upper = (pts[:, 1] < self._cfg_general.y_max) if self._cfg_general.y_max is not None else np.ones(pts.shape[0], dtype=bool)
            points_y_idx_lower = (pts[:, 1] > self._cfg_general.y_min) if self._cfg_general.y_min is not None else np.ones(pts.shape[0], dtype=bool)
            pts = pts[np.vstack((points_x_idx_lower, points_x_idx_upper, points_y_idx_upper, points_y_idx_lower)).all(axis=0)]
        if outlier_filter:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                cl, _ = pcd.remove_statistical_outlier(nb_neighbors=self._cfg_tsdf.nb_neighbors, std_ratio=self._cfg_tsdf.std_ratio)
                pts = np.asarray(cl.points, dtype=np.float32)
            except Exception:
                pass
        return pts

    def VizCloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])
