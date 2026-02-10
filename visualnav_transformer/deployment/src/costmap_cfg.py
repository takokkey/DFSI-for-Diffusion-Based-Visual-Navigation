# costmap_cfg.py
# Lightweight config/data classes for TSDF costmap (compatible with NaviD style)

import os
from typing import Optional
import yaml

class Loader(yaml.SafeLoader):
    pass

# Basic dataclass-like containers
class ReconstructionCfg:
    data_dir: str = "${USER_PATH_TO_DATA}"
    env: str = "town01"
    depth_suffix = "_cam0"
    sem_suffix = "_cam1"
    high_res_depth: bool = False
    voxel_size: float = 0.1
    start_idx: int = 0
    max_images: Optional[int] = 1000
    depth_scale: float = 1000.0
    semantics: bool = False
    point_cloud_batch_size: int = 200

class SemCostMapConfig:
    ground_height: Optional[float] = -0.5
    robot_height: float = 0.70
    robot_height_factor: float = 3.0
    nb_neighbors: int = 100
    std_ratio: float = 2.0
    downsample: bool = False
    nb_neigh: int = 15
    change_decimal: int = 3
    conv_crit: float = 0.45
    nb_tasks: Optional[int] = 10
    sigma_smooth: float = 2.5
    max_iterations: int = 1
    obstacle_threshold: float = 0.8
    negative_reward: float = 0.5
    round_decimal_traversable: int = 2
    compute_height_map: bool = False

class TsdfCostMapConfig:
    offset_z: float = 0.3
    ground_height: float = 0.3
    robot_height: float = 0.7
    robot_height_factor: float = 1.2
    nb_neighbors: int = 50
    std_ratio: float = 0.2
    filter_outliers: bool = True
    sigma_expand: float = 3.0
    obstacle_threshold: float = 0.01
    free_space_threshold: float = 0.8

class GeneralCostMapConfig:
    root_path: str = "town01"
    ply_file: str = "cloud.ply"
    resolution: float = 0.04
    clear_dist: float = 1.0
    sigma_smooth: float = 3.0
    x_min: Optional[float] = -4.0
    y_min: Optional[float] = -3
    x_max: Optional[float] = 4.0
    y_max: Optional[float] = 3.0

class CostMapConfig:
    semantics: bool = True
    geometry: bool = True
    map_name: str = "cost_map_sem"
    general: GeneralCostMapConfig = GeneralCostMapConfig()
    sem_cost_map: SemCostMapConfig = SemCostMapConfig()
    tsdf_cost_map: TsdfCostMapConfig = TsdfCostMapConfig()
    visualize: bool = True
    x_start: float = None
    y_start: float = None

# YAML loader constructors kept minimal (if you plan to load YAML into these classes)
def construct_GeneralCostMapConfig(loader, node):
    return GeneralCostMapConfig(**loader.construct_mapping(node))

Loader.add_constructor("tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.GeneralCostMapConfig", construct_GeneralCostMapConfig)

def construct_TsdfCostMapConfig(loader, node):
    return TsdfCostMapConfig(**loader.construct_mapping(node))

Loader.add_constructor("tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.TsdfCostMapConfig", construct_TsdfCostMapConfig)
