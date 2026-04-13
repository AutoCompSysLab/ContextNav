import gzip
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from omegaconf import OmegaConf


def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)


def _get_benchmark() -> str:
    return os.environ.get("CONTEXTNAV_BENCHMARK", "psl")


@baseline_registry.register_trainer(name="vlfm")
class VLFMTrainer(PPOTrainer):
    envs: VectorEnv

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """Evaluates a single checkpoint."""
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        if self.config.habitat_baselines.eval.should_load_ckpt:
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        else:
            ckpt_dict = {"config": None}

        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])
        benchmark = _get_benchmark()

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split
            config.habitat.task.measurements.frontier_exploration_map.draw_goal_aabbs = False

        # COIN: load dataset metadata for object-id-to-name mapping
        integer_to_string_mapping = None
        if benchmark == "coin":
            ds_json_data_path = config.habitat.dataset.data_path.format(
                split=config.habitat.dataset.split
            )
            with gzip.open(ds_json_data_path, "rt") as f:
                ds_json_data = json.load(f)
            integer_to_string_mapping = ds_json_data["category_to_task_category_id"]

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(render_view.uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        # PSL: override max steps and success distance from TASK config
        if benchmark == "psl":
            config.habitat.environment.max_episode_steps = config.TASK.MAX_STEP
            config.habitat.task.measurements.success.success_distance = config.TASK.SUCCESS.SUCCESS_DISTANCE

        self._init_envs(config, is_eval=True)
        self._agent = self._create_agent(None)

        # COIN: set object id-to-name mapping on the policy
        if integer_to_string_mapping is not None:
            self._agent.actor_critic.set_obj_id_to_name(
                integer_to_string_mapping=integer_to_string_mapping
            )

        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)

        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        observations = self.envs.reset()
        goal_attributes = self._extract_goal(observations, benchmark)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")
        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *self._agent.hidden_state_shape,
            ),
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = {}
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames: List[List[np.ndarray]] = [
            [] for _ in range(self.config.habitat_baselines.num_environments)
        ]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, "
                    f"dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert number_of_eval_episodes > 0, (
            "You must specify a number of evaluation episodes with test_episode_count"
        )

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self._agent.eval()

        from vlfm.utils.episode_stats_logger import log_episode_stats
        from vlfm.utils.habitat_visualizer import HabitatVis

        num_successes = 0
        num_total = 0
        hab_vis = HabitatVis()

        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and self.envs.num_envs > 0
        ):
            current_episodes_info = self.envs.current_episodes()

            with inference_mode():
                action_data = self._agent.actor_critic.act(
                    batch,
                    goal_attributes,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                    action_id = action_data.actions.cpu()[0].item()
                    filepath = os.path.join(
                        os.environ["VLFM_RECORD_ACTIONS_DIR"],
                        "actions.txt",
                    )
                    if not os.path.exists(filepath):
                        open(filepath, "w").close()
                    with open(filepath, "a") as f:
                        f.write(f"{action_id}\n")

                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    for i, should_insert in enumerate(action_data.should_inserts):
                        if should_insert.item():
                            test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore

            if is_continuous_action_space(self._env_spec.action_space):
                step_data = [
                    np.clip(
                        a.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = self.envs.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

            # Build episode name for video (PSL uses a richer name)
            episode_name = self._make_episode_name(
                current_episodes_info[0], goal_attributes, benchmark
            )

            goal_attributes = self._extract_goal(observations, benchmark)
            policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                elif int(next_episodes_info[i].episode_id) == 123123123:
                    envs_to_pause.append(i)

                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)

                # Episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    try:
                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
                            infos[i],
                        )
                    except Exception:
                        failure_cause = "Unknown"

                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_total += 1
                    print(
                        f"Success rate: {num_successes / num_total * 100:.2f}% "
                        f"({num_successes} out of {num_total})"
                    )

                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        rgb_frames[i] = hab_vis.flush_frames(failure_cause)
                        should_save_video = (
                            episode_stats["success"] == 1
                        )
                        if should_save_video:
                            generate_video(
                                video_option=self.config.habitat_baselines.eval.video_option,
                                video_dir=self.config.habitat_baselines.video_dir,
                                images=rgb_frames[i],
                                episode_id=episode_name,
                                checkpoint_idx=checkpoint_index,
                                metrics=extract_scalars_from_info(infos[i]),
                                fps=self.config.habitat_baselines.video_fps,
                                tb_writer=writer,
                                keys_to_include_in_name=(
                                    self.config.habitat_baselines.eval_keys_to_include_in_name
                                ),
                            )
                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()

        if "ZSOS_DONE_PATH" in os.environ:
            done_path = os.environ["ZSOS_DONE_PATH"]
            with open(done_path, "w") as f:
                f.write("")

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)
        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()

    # ---- Benchmark-specific helpers ----

    @staticmethod
    def _extract_goal(observations: list, benchmark: str) -> Any:
        """Extract goal attributes from observations and remove them from the obs dict."""
        if benchmark == "psl":
            from vlfm.benchmarks.psl.sensors.psl_sensor import ObjectGoalSensor
            goal_attributes = observations[0][ObjectGoalSensor.cls_uuid]
            observations[0].pop("objectgoal")
        else:
            goal_attributes = observations[0]["instruction"]
            observations[0].pop("instruction")
        return goal_attributes

    @staticmethod
    def _make_episode_name(episode_info: Any, goal_attributes: Any, benchmark: str) -> str:
        """Build a descriptive episode name for video filenames."""
        if benchmark == "psl" and isinstance(goal_attributes, dict) and "image" in goal_attributes:
            return (
                episode_info.scene_id.split("/")[-2]
                + "-"
                + episode_info.episode_id
                + "-"
                + goal_attributes["image"].split(".")[0].split("/")[-1]
            )
        return episode_info.episode_id
