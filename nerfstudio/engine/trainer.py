# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train model.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Tuple, Type, cast

import numpy as np
import torch
import viser
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs.experiment_config import ExperimentConfig
try:  # pragma: no cover - matplotlib is optional at runtime
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore
    from matplotlib.patches import Patch  # type: ignore
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # register 3D projection
except Exception:  # pragma: no cover - matplotlib optional
    plt = None  # type: ignore
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled, check_main_thread, check_viewer_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.viewer import Viewer as ViewerState
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
TORCH_DEVICE = str


@dataclass
class TrainerConfig(ExperimentConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: Trainer)
    """target class to instantiate"""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 500
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 500
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    use_grad_scaler: bool = False
    """Use gradient scaler even if the automatic mixed precision is disabled."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    load_checkpoint: Optional[Path] = None
    """Path to checkpoint file."""
    log_gradients: bool = False
    """Optionally log gradients during training"""
    gradient_accumulation_steps: Dict[str, int] = field(default_factory=lambda: {})
    """Number of steps to accumulate gradients over. Contains a mapping of {param_group:num}"""
    start_paused: bool = False
    """Whether to start the training in a paused state."""


class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        self.train_lock = Lock()
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device: TORCH_DEVICE = config.machine.device_type
        if self.device == "cuda":
            self.device += f":{local_rank}"
        self.mixed_precision: bool = self.config.mixed_precision
        self.use_grad_scaler: bool = self.mixed_precision or self.config.use_grad_scaler
        self.training_state: Literal["training", "paused", "completed"] = (
            "paused" if self.config.start_paused else "training"
        )
        self.gradient_accumulation_steps: DefaultDict = defaultdict(lambda: 1)
        self.gradient_accumulation_steps.update(self.config.gradient_accumulation_steps)

        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step: int = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler)

        self.base_dir: Path = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir: Path = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")

        self.viewer_state = None

        # used to keep track of the current step
        self.step = 0

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerLegacyState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Legacy viewer at: {self.viewer_state.viewer_url}"]
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()

        self._load_checkpoint()

        if self.local_rank == 0:
            self._export_camera_distributions()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()
        return Optimizers(optimizer_config, param_groups)

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"
        if hasattr(self.pipeline.datamanager, "train_dataparser_outputs"):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(  # type: ignore
                self.base_dir / "dataparser_transforms.json"
            )

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations - self._start_step
            step = 0
            self.stop_training = False
            for step in range(self._start_step, self._start_step + num_iterations):
                self.step = step
                if self.stop_training:
                    break
                while self.training_state == "paused":
                    if self.stop_training:
                        self._after_train()
                        return
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    with self.train_lock:
                        self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

        # save checkpoint at the end of training, and write out any remaining events
        self._after_train()

    def shutdown(self) -> None:
        """Stop the trainer and stop all associated threads/processes (such as the viewer)."""
        self.stop_training = True  # tell the training loop to stop
        if self.viewer_state is not None:
            # stop the viewer
            # this condition excludes the case where `viser_server` is either `None` or an
            # instance of `viewer_legacy`'s `ViserServer` instead of the upstream one.
            if isinstance(self.viewer_state.viser_server, viser.ViserServer):
                self.viewer_state.viser_server.stop()

    def _after_train(self) -> None:
        """Function to run after training is complete"""
        self.training_state = "completed"  # used to update the webui state
        # save checkpoint at the end of training
        self.save_checkpoint(self.step)
        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()
        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=self.step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    @check_main_thread
    def _check_viewer_warnings(self) -> None:
        """Helper to print out any warnings regarding the way the viewer/loggers are enabled"""
        if (
            (self.config.is_viewer_legacy_enabled() or self.config.is_viewer_enabled())
            and not self.config.is_tensorboard_enabled()
            and not self.config.is_wandb_enabled()
            and not self.config.is_comet_enabled()
        ):
            string: str = (
                "[NOTE] Not running eval iterations since only viewer is enabled.\n"
                "Use [yellow]--vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard}[/yellow] to run with eval."
            )
            CONSOLE.print(f"{string}")

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state and self.pipeline.datamanager.train_dataset
        self.viewer_state.init_scene(
            train_dataset=self.pipeline.datamanager.train_dataset,
            train_state=self.training_state,
            eval_dataset=self.pipeline.datamanager.eval_dataset,
        )

    @check_viewer_enabled
    def _update_viewer_state(self, step: int) -> None:
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        try:
            self.viewer_state.update_scene(step, num_rays_per_batch)
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")

    @check_viewer_enabled
    def _train_complete_viewer(self) -> None:
        """Let the viewer know that the training is complete"""
        assert self.viewer_state is not None
        self.training_state = "completed"
        try:
            self.viewer_state.training_complete()
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")
        CONSOLE.print("Use ctrl+c to quit", justify="center")
        while True:
            time.sleep(0.01)

    @check_viewer_enabled
    def _update_viewer_rays_per_sec(self, train_t: TimeWriter, vis_t: TimeWriter, step: int) -> None:
        """Performs update on rays/sec calculation for training

        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=self.world_size * train_num_rays_per_batch / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
        )

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_dir = Path(load_dir)
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                checkpoint_steps = []
                for entry in load_dir.iterdir():
                    if not entry.is_file() or entry.suffix != ".ckpt":
                        continue
                    stem = entry.stem
                    if not stem.startswith("step-"):
                        continue
                    step_str = stem[len("step-") :]
                    if not step_str.isdigit():
                        continue
                    checkpoint_steps.append(int(step_str))

                if not checkpoint_steps:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(
                        f"No checkpoints matching 'step-*.ckpt' found in {load_dir}", justify="center"
                    )
                    sys.exit(1)

                load_step = max(checkpoint_steps)
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")

    def _export_camera_distributions(self) -> None:
        """Save camera distribution visualisations next to checkpoints."""

        datamanager = getattr(self.pipeline, "datamanager", None)
        if datamanager is None:
            return

        def _pose_info_from_outputs(outputs) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            cameras = getattr(outputs, "cameras", None)
            if cameras is None:
                return None
            camera_to_worlds = getattr(cameras, "camera_to_worlds", None)
            if camera_to_worlds is None:
                return None
            tensor = torch.as_tensor(camera_to_worlds)
            if tensor.numel() == 0:
                return None
            origins = tensor[..., :3, 3]
            rotations = tensor[..., :3, :3]
            forwards = -rotations[..., :, 2]
            origins = origins.reshape(-1, 3)
            forwards = forwards.reshape(-1, 3)
            return origins, forwards

        splits: List[Tuple[str, torch.Tensor, torch.Tensor]] = []

        train_outputs = getattr(datamanager, "train_dataparser_outputs", None)
        train_info = _pose_info_from_outputs(train_outputs) if train_outputs is not None else None
        if train_info is not None:
            splits.append(("train", *train_info))

        eval_outputs = getattr(datamanager, "eval_dataparser_outputs", None)
        eval_info = _pose_info_from_outputs(eval_outputs) if eval_outputs is not None else None
        if eval_info is not None:
            splits.append(("eval", *eval_info))

        if not splits:
            return

        model = getattr(self.pipeline, "model", None)
        normal = getattr(model, "water_plane_normal", None)
        offset = getattr(model, "water_plane_offset", None)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def _export_matplotlib() -> None:
            if plt is None:
                CONSOLE.log(
                    "Matplotlib unavailable; skipping camera distribution plot.",
                    style="yellow",
                )
                return

            png_path = self.checkpoint_dir / "camera_distribution.png"
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")

            cmap = plt.get_cmap("tab10")

            plane_info = ""
            plane_plotted = False

            marker_positions_np: Optional[np.ndarray] = None
            marker_labels_plot: Optional[List[str]] = None

            # Collect camera positions only (for plane extent calculation)
            camera_positions_list: List[np.ndarray] = [pos.detach().cpu().numpy() for _, pos, _ in splits if pos.numel() > 0]

            # Collect all points (cameras + markers) for final bounds
            points_for_bounds: List[np.ndarray] = camera_positions_list.copy()

            if train_outputs is not None:
                train_metadata = getattr(train_outputs, "metadata", {})
                water_surface_meta = train_metadata.get("water_surface") if isinstance(train_metadata, dict) else None
                if water_surface_meta is not None:
                    markers_model = water_surface_meta.get("markers_model")
                    if markers_model:
                        marker_labels_plot = water_surface_meta.get("marker_labels") or list(markers_model.keys())
                        marker_positions_np = np.array(
                            [markers_model[label] for label in marker_labels_plot],
                            dtype=float,
                        )
                        points_for_bounds.append(marker_positions_np)

            # Calculate extent from cameras only (for plane size)
            if camera_positions_list:
                camera_concat = np.concatenate(camera_positions_list, axis=0)
                camera_min = camera_concat.min(axis=0)
                camera_max = camera_concat.max(axis=0)
                camera_extent = camera_max - camera_min
                camera_max_extent = float(camera_extent.max()) if camera_extent.size > 0 else 1.0
            else:
                camera_max_extent = 1.0

            # Calculate bounds from all points (cameras + markers) for axis limits
            if points_for_bounds:
                concatenated = np.concatenate(points_for_bounds, axis=0)
                min_bounds = torch.from_numpy(concatenated.min(axis=0))
                max_bounds = torch.from_numpy(concatenated.max(axis=0))
            else:
                min_bounds = torch.zeros(3)
                max_bounds = torch.ones(3)

            extent = max_bounds - min_bounds
            max_extent = float(extent.max().item()) if extent.numel() > 0 else 1.0
            arrow_length = max(max_extent * 0.08, 1e-3)

            for idx, (name, positions, directions) in enumerate(splits):
                color = cmap(idx % cmap.N)
                pos_np = positions.detach().cpu().numpy()
                dir_np = directions.detach().cpu().numpy()

                if pos_np.size == 0:
                    continue

                norms = np.linalg.norm(dir_np, axis=1, keepdims=True)
                dir_unit = np.divide(dir_np, np.maximum(norms, 1e-8))

                ax.scatter(
                    pos_np[:, 0],
                    pos_np[:, 1],
                    pos_np[:, 2],
                    label=f"{name} cameras",
                    color=color,
                    s=20,
                    depthshade=False,
                )

                ax.quiver(
                    pos_np[:, 0],
                    pos_np[:, 1],
                    pos_np[:, 2],
                    dir_unit[:, 0],
                    dir_unit[:, 1],
                    dir_unit[:, 2],
                    length=arrow_length,
                    normalize=True,
                    color=color,
                    linewidths=1.0,
                )

            if normal is not None and offset is not None:
                try:
                    normal_tensor = torch.as_tensor(normal).detach().cpu().view(-1).float()
                    offset_val = float(torch.as_tensor(offset).detach().cpu().view(-1)[0])
                    normal_np = normal_tensor.numpy()
                    norm = np.linalg.norm(normal_np)
                    if norm > 1e-6:
                        normal_unit = normal_np / norm
                        point_on_plane = -offset_val * normal_unit

                        basis = np.array([1.0, 0.0, 0.0])
                        if abs(np.dot(basis, normal_unit)) > 0.95:
                            basis = np.array([0.0, 1.0, 0.0])
                        tangent_u = np.cross(normal_unit, basis)
                        tangent_u_norm = np.linalg.norm(tangent_u)
                        if tangent_u_norm < 1e-6:
                            basis = np.array([0.0, 0.0, 1.0])
                            tangent_u = np.cross(normal_unit, basis)
                            tangent_u_norm = np.linalg.norm(tangent_u)
                        if tangent_u_norm > 1e-6:
                            tangent_u /= tangent_u_norm
                            tangent_v = np.cross(normal_unit, tangent_u)

                            # Use camera extent only for plane size (not markers)
                            plane_extent = max(camera_max_extent * 1.2, 1.0)  # 20% larger than camera extent
                            grid = np.linspace(-plane_extent, plane_extent, 30)
                            uu, vv = np.meshgrid(grid, grid)
                            plane_points = (
                                point_on_plane[None, None, :]
                                + tangent_u[None, None, :] * uu[..., None]
                                + tangent_v[None, None, :] * vv[..., None]
                            )
                            # Don't add plane points to bounds - keep plane constrained to camera extent

                            ax.plot_surface(
                                plane_points[..., 0],
                                plane_points[..., 1],
                                plane_points[..., 2],
                                color=(0.0, 0.6, 1.0, 0.35),
                                linewidth=0,
                                antialiased=False,
                                shade=False,
                                alpha=0.4,
                            )
                            plane_plotted = True

                            if abs(normal_unit[2]) > 1e-6:
                                water_z = -offset_val / normal_unit[2]
                                plane_info = (
                                    f"Water plane n=[{normal_unit[0]:.2f}, {normal_unit[1]:.2f}, {normal_unit[2]:.2f}], "
                                    f"z₀≈{water_z:.3f}"
                                )
                            else:
                                plane_info = (
                                    f"Water plane n=[{normal_unit[0]:.2f}, {normal_unit[1]:.2f}, {normal_unit[2]:.2f}] (vertical)"
                                )
                except Exception as exc:  # pragma: no cover - visualization best effort
                    CONSOLE.print(f"[yellow]Warning: Could not visualise water surface: {exc}[/yellow]")

            marker_handles: List[Line2D] = []
            if marker_positions_np is not None and marker_labels_plot is not None:
                ax.scatter(
                    marker_positions_np[:, 0],
                    marker_positions_np[:, 1],
                    marker_positions_np[:, 2],
                    color="red",
                    s=40,
                    depthshade=False,
                    marker="o",
                )
                for label, position in zip(marker_labels_plot, marker_positions_np):
                    ax.text(
                        position[0],
                        position[1],
                        position[2],
                        f" {label}",
                        color="red",
                        fontsize=10,
                    )
                marker_handles.append(
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=6, linestyle="None", label="water markers")
                )

            if points_for_bounds:
                concatenated_all = np.concatenate(points_for_bounds, axis=0)
                min_bounds = torch.from_numpy(concatenated_all.min(axis=0))
                max_bounds = torch.from_numpy(concatenated_all.max(axis=0))

            x_margin = max_extent * 0.1
            ax.set_xlim(float(min_bounds[0] - x_margin), float(max_bounds[0] + x_margin))
            ax.set_ylim(float(min_bounds[1] - x_margin), float(max_bounds[1] + x_margin))
            ax.set_zlim(float(min_bounds[2] - x_margin), float(max_bounds[2] + x_margin))

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            handles, labels_existing = ax.get_legend_handles_labels()
            if plane_plotted:
                handles.append(Patch(facecolor=(0.0, 0.6, 1.0, 0.35), edgecolor="none", label="water surface"))
                labels_existing.append("water surface")
            handles.extend(marker_handles)
            labels_existing.extend([handle.get_label() for handle in marker_handles])
            if handles:
                ax.legend(handles, labels_existing, loc="upper right")
            ax.set_title("Camera distribution in NeRF space")

            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.close(fig)

            if plane_info:
                CONSOLE.log(plane_info)
            CONSOLE.log(f"Saved camera distribution plot to: {png_path}")

        try:
            _export_matplotlib()
        except Exception as exc:  # pragma: no cover - visualization best effort
            CONSOLE.print(f"[yellow]Warning: Camera distribution plot failed: {exc}[/yellow]")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete every other checkpoint in the checkpoint folder
            for f in self.checkpoint_dir.glob("*.ckpt"):
                if f != ckpt_path:
                    f.unlink()

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        needs_zero = [
            group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        needs_step = [
            group
            for group in self.optimizers.parameters.keys()
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)
