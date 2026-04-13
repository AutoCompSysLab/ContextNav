"""Loads model_config.yaml and applies environment variable overrides."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "model_config.yaml"
_cached_config: Dict[str, Any] = {}

# Mapping: env var name -> (yaml key path, type cast)
_ENV_OVERRIDES: Dict[str, tuple] = {
    # Ports
    "GROUNDING_DINO_PORT": (("ports", "grounding_dino"), int),
    "CLIP2ITM_PORT": (("ports", "clip2itm"), int),
    "SAM_PORT": (("ports", "sam"), int),
    "YOLOV7_PORT": (("ports", "yolov7"), int),
    # VLM
    "VLM_MODEL": (("vlm", "model"), str),
    "OLLAMA_BASE_URL": (("vlm", "base_url"), str),
    "OLLAMA_API_KEY": (("vlm", "api_key"), str),
    "VLM_USE_BBOX": (("vlm", "use_bbox"), lambda v: bool(int(v))),
    "VERIFY_WITH_VLM": (("vlm", "verify_with_vlm"), lambda v: bool(int(v))),
    # LLM
    "LLM_MODEL": (("llm", "model"), str),
    # IG weights
    "IG_W_VIS": (("ig_weights", "vis"), float),
    "IG_W_SIG": (("ig_weights", "sig"), float),
    # Detection
    "MAXDEPTH_NORM_TH": (("detection", "maxdepth_norm_th"), float),
    "MAXDEPTH_SKIP_RATIO": (("detection", "maxdepth_skip_ratio"), float),
    "VLM_DECISION_THRESHOLD": (("detection", "vlm_decision_threshold"), float),
    # Navigation
    "BLACKLIST_RADIUS_M": (("navigation", "blacklist_radius_m"), float),
    "RELATION_MAX_DIST_M": (("navigation", "relation_max_distance_m"), float),
    # Spatial
    "REL_VERIFY_MAX_INST_PER_CLASS": (("spatial", "rel_verify_max_inst_per_class"), int),
    # Visualization
    "DEPTH_HFOV_DEG": (("visualization", "depth_hfov_deg"), float),
    "RGB_HFOV_DEG": (("visualization", "rgb_hfov_deg"), float),
    # YNU
    "YNU_DEFER_WINDOW": (("ynu", "defer_window"), int),
    # Value map
    "RECORD_VALUE_MAP": (("value_map", "record"), lambda v: v == "1"),
    "PLAY_VALUE_MAP": (("value_map", "play"), lambda v: v == "1"),
    "MAP_FUSION_TYPE": (("value_map", "fusion_type"), str),
    # Object cloud
    "INSTANCE_EPS_M": (("object_cloud", "instance_eps_m"), float),
    "INSTANCE_MIN_PTS": (("object_cloud", "instance_min_pts"), int),
    "DET_CACHE_CAP": (("object_cloud", "det_cache_cap"), int),
    # Cloud geometry
    "D2R_ROLL_DEG": (("cloud_geometry", "d2r_roll_deg"), float),
    "D2R_PITCH_DEG": (("cloud_geometry", "d2r_pitch_deg"), float),
    "D2R_YAW_DEG": (("cloud_geometry", "d2r_yaw_deg"), float),
    "D2R_TX": (("cloud_geometry", "d2r_tx"), float),
    "D2R_TY": (("cloud_geometry", "d2r_ty"), float),
    "D2R_TZ": (("cloud_geometry", "d2r_tz"), float),
    # Scoring
    "CAND_W_ATTR": (("scoring", "cand_w_attr"), float),
    "CAND_W_CTX": (("scoring", "cand_w_ctx"), float),
    "CAND_W_CTXCOUNT": (("scoring", "cand_w_ctxcount"), float),
    "CAND_W_REL": (("scoring", "cand_w_rel"), float),
    # Instance
    "DET_VOXEL_M": (("instance", "det_voxel_m"), float),
    "DET_OVERLAP_MIN_RATIO": (("instance", "det_overlap_min_ratio"), float),
    # Wall map
    "WALLMAP_MIN_RANGE_M": (("wall_map", "min_range_m"), float),
    "WALLMAP_MAX_RANGE_M": (("wall_map", "max_range_m"), float),
    "WALLMAP_STAT_NB": (("wall_map", "stat_nb"), int),
    "WALLMAP_STAT_STD": (("wall_map", "stat_std"), float),
    "WALLMAP_RAD_NB": (("wall_map", "rad_nb"), int),
    "WALLMAP_RAD_RADIUS": (("wall_map", "rad_radius"), float),
    "WALLMAP_RANSAC_DIST": (("wall_map", "ransac_dist"), float),
    "WALLMAP_RANSAC_ITERS": (("wall_map", "ransac_iters"), int),
    "WALLMAP_MAX_PLANES": (("wall_map", "max_planes"), int),
    "WALLMAP_VERTICAL_NZ_MAX": (("wall_map", "vertical_nz_max"), float),
    "WALLMAP_MIN_INLIERS": (("wall_map", "min_inliers"), int),
    "WALLMAP_MIN_BLOB_PX": (("wall_map", "min_blob_px"), int),
    "WALLMAP_VOXEL_M": (("wall_map", "voxel_m"), float),
    "WALLMAP_SAMPLE_CAP_N": (("wall_map", "sample_cap_n"), int),
    # Wall vis
    "WALL_VIS_EXPLORED_MASK": (("wall_vis", "explored_mask"), lambda v: bool(int(v))),
    # Benchmark overrides
    "CONTEXTNAV_BENCHMARK": (("benchmark",), str),
    "FALLBACK_STEP": (("_override_fallback_step",), int),
    "ENABLE_VLM_EXIST_CHECK": (("_override_enable_vlm_exist_check",), lambda v: bool(int(v))),
    "ENABLE_DEPTH_FOV_CORRECTION": (("_override_enable_depth_fov_correction",), lambda v: bool(int(v))),
}


def _set_nested(d: dict, keys: tuple, value: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _get_nested(d: dict, keys: tuple, default: Any = None) -> Any:
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load model config from YAML, apply benchmark profile, then env var overrides.

    Results are cached after the first call (same process).
    """
    global _cached_config
    if _cached_config and config_path is None:
        return _cached_config
    path = Path(config_path) if config_path else _CONFIG_PATH
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply env var overrides
    for env_key, (yaml_path, cast_fn) in _ENV_OVERRIDES.items():
        val = os.environ.get(env_key)
        if val is not None:
            _set_nested(cfg, yaml_path, cast_fn(val))

    # Resolve benchmark profile
    benchmark = cfg.get("benchmark", "psl")
    profiles = cfg.get("benchmark_profiles", {})
    profile = profiles.get(benchmark, profiles.get("psl", {}))

    # Benchmark profile values (can be overridden by explicit env vars)
    for key in ("fallback_step", "enable_vlm_exist_check", "enable_depth_fov_correction"):
        override_key = f"_override_{key}"
        if override_key in cfg:
            cfg.setdefault("benchmark_resolved", {})[key] = cfg.pop(override_key)
        else:
            cfg.setdefault("benchmark_resolved", {})[key] = profile.get(key)

    if config_path is None:
        _cached_config = cfg
    return cfg
