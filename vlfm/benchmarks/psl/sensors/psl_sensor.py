import os
from typing import Any

from gym import spaces

from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.nav.nav import NavigationEpisode


@registry.register_sensor
class ObjectGoalSensor(Sensor):
    cls_uuid: str = "objectgoal"

    def __init__(self, *args: Any, config, dataset: Dataset, **kwargs: Any):
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Discrete(1)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def get_observation(
        self,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        return episode.goal_attributes
