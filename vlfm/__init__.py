from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from habitat.config.default_structured_configs import LabSensorConfig

cs = ConfigStore.instance()


@dataclass
class InstructionSensorConfig(LabSensorConfig):
    type: str = "InstructionSensor"


cs.store(
    package="habitat.task.lab_sensors.instruction",
    group="habitat/task/lab_sensors",
    name="instruction",
    node=InstructionSensorConfig,
)


@dataclass
class ImageGoalSensorSensorConfig(LabSensorConfig):
    type: str = "ImageGoalSensor"
    image_cache_encoder: str = ""


cs.store(
    package="habitat.task.lab_sensors.instance_imagegoal_sensor",
    group="habitat/task/lab_sensors",
    name="instance_imagegoal_sensor",
    node=ImageGoalSensorSensorConfig,
)