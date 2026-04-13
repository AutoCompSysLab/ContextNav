import os

# The following imports require habitat to be installed, and despite not being used by
# this script itself, will register several classes and make them discoverable by Hydra.
# This run.py script is expected to only be used when habitat is installed, thus they
# are hidden here instead of in an __init__.py file. This avoids import errors when used
# in an environment without habitat, such as when doing real-world deployment. noqa is
# used to suppress the unused import and unsorted import warnings by ruff.
import frontier_exploration  # noqa
import hydra  # noqa
from habitat import get_config  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import DatasetConfig, register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig, OmegaConf

import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
import vlfm.utils.vlfm_trainer  # noqa: F401

# Benchmark-specific module registrations
_benchmark = os.environ.get("CONTEXTNAV_BENCHMARK", "psl")

if _benchmark == "psl":
    import vlfm.benchmarks.psl.sensors.psl_sensor  # noqa: F401
    import vlfm.benchmarks.psl.dataset.psl_episode  # noqa: F401
elif _benchmark == "coin":
    import vlfm.benchmarks.coin.instance_image_nav  # noqa: F401
    import vlfm.benchmarks.coin.uncertainty_task  # noqa: F401
    from vlfm.measurements import distractor_success  # noqa: F401


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")


register_hydra_plugin(HabitatConfigPlugin)

# Default config is selected based on the benchmark environment variable.
# Override via Hydra CLI: --config-name=experiments/uncertainty_objectnav_hm3d
_default_configs = {
    "psl": "experiments/vlfm_attrnav_hm3d",
    "coin": "experiments/uncertainty_objectnav_hm3d",
}


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name=_default_configs.get(os.environ.get("CONTEXTNAV_BENCHMARK", "psl"), "experiments/vlfm_attrnav_hm3d"),
)
def main(cfg: DictConfig) -> None:
    assert os.path.isdir("data"), "Missing 'data/' directory!"
    if not os.path.isfile("data/dummy_policy.pth"):
        print("Dummy policy weights not found! Please run the following command first:")
        print("python -m vlfm.utils.generate_dummy_policy")
        exit(1)

    cfg = patch_config(cfg)

    benchmark = os.environ.get("CONTEXTNAV_BENCHMARK", "psl")
    if benchmark == "psl":
        OmegaConf.set_readonly(cfg, False)
        cfg.habitat.dataset = DatasetConfig(**cfg.DATASET)
        cfg.habitat.dataset["content_scenes"] = ["*"]
        cfg.habitat.task.type = "AttrObjNavTask-v1"

    with read_write(cfg):
        try:
            cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
        except KeyError:
            pass
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    main()
