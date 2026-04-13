from dataclasses import dataclass
from typing import Any

import numpy as np
from habitat import registry
from habitat.config.default_structured_configs import (
    MeasurementConfig,
)
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Simulator
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@registry.register_measure
class DistractorSuccess(Measure):
    cls_uuid: str = "distractor_success"

    def __init__(self, sim: Simulator, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        self._current_episode = None
        self._success_distance = kwargs["task"]._config.measurements.success.success_distance
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return DistractorSuccess.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._current_episode = kwargs["episode"]
        assert len(self._current_episode.goals) == 1, "uncertainty lang task is an instance obj nav"
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        if "action" not in kwargs:
            self._metric = 0
            return

        action = kwargs["action"]["action"]
        if action != 0:
            self._metric = 0
            return

        # otherwise, let's check the position of the distractor
        distractors = self._current_episode.goals[0].distractors
        current_position = self._sim.get_agent_state().position

        for distractor in distractors:
            geo_dist = self._sim.geodesic_distance(current_position, distractor)
            if np.isnan(geo_dist) or np.isinf(geo_dist):
                continue

            # Small offset: some objects (pictures, mirrors) are flush against walls
            offset = 0.1
            if geo_dist < self._success_distance + offset:
                self._metric = 1
                return

        self._metric = 0


@dataclass
class DistractorSuccessMeasurementConfig(MeasurementConfig):
    type: str = DistractorSuccess.__name__


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.distractor_success",
    group="habitat/task/measurements",
    name="distractor_success",
    node=DistractorSuccessMeasurementConfig,
)
