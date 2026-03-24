"""Pydantic schema for motor controller model configuration."""

import hashlib
from pathlib import Path
from typing import List, Literal

import structlog
import yaml
from pydantic import BaseModel, Field

_log = structlog.get_logger("m1_config")


class SimulationConfig(BaseModel):
    """Simulation and environment parameters."""

    rng_seed: int = Field(default=1234, description="Random seed for reproducibility")
    print_time: bool = Field(default=False, description="Print simulation progress")
    total_num_virtual_procs: int = Field(
        default=16, description="Number of virtual processes for NEST"
    )
    step: float = Field(default=1.0, description="Simulation time step (ms)")


class TaskConfig(BaseModel):
    """Task and experiment setup parameters."""

    gradient_batch_size: int = Field(
        default=1, description="Batch size for gradient computation"
    )
    n_iter: int = Field(default=200, description="Number of training iterations")
    input_shift_ms: float = Field(
        default=50.0,
        description="Temporal delay to shift M1 target forward (ms)",
    )
    learning_start_ms: float = Field(
        default=600.0,
        description="Absolute start time (ms) for learning window within each sequence. "
        "NEST zeros the error/target/readout signals before this time. "
        "E.g. with sequence=1150ms and learning_start_ms=600ms, learning is active "
        "from t=600 to t=1150 within each sequence.",
    )


class TrajectorySpec(BaseModel):
    """One trajectory to train on, specified as start/end angles."""

    init_angle_deg: float
    target_angle_deg: float


class TrainingSignalConfig(BaseModel):
    """Parameters for end-to-end training signal generation."""

    trajectories: List[TrajectorySpec] = Field(
        default_factory=lambda: [
            TrajectorySpec(init_angle_deg=90, target_angle_deg=140),
            TrajectorySpec(init_angle_deg=90, target_angle_deg=20),
            # TrajectorySpec(init_angle_deg=0, target_angle_deg=90),
            # TrajectorySpec(init_angle_deg=90, target_angle_deg=90),
            # TrajectorySpec(init_angle_deg=90, target_angle_deg=20),
            # TrajectorySpec(init_angle_deg=20, target_angle_deg=20),
            # TrajectorySpec(init_angle_deg=20, target_angle_deg=80),
            # TrajectorySpec(init_angle_deg=80, target_angle_deg=140),
            # TrajectorySpec(init_angle_deg=90, target_angle_deg=90),
        ]
    )
    n_input_neurons: int = Field(
        default=200, description="Neurons per channel (pos/neg) for planner input"
    )
    planner_kp: float = Field(
        default=100.0, description="Gain for planner tracking neurons"
    )
    planner_base_rate: float = Field(
        default=5, description="Base rate for planner tracking neurons (Hz)"
    )
    m1_kp: float = Field(
        default=2000.0, description="Gain for M1 target (matches mocked M1)"
    )
    m1_base_rate: float = Field(default=0.0, description="Base rate for M1 target (Hz)")
    inertia: float = Field(
        default=0.00189, description="Moment of inertia for 1-DOF robot (kg·m²)"
    )
    time_prep_ms: float = Field(
        default=50.0, description="Preparation phase duration (ms)"
    )
    time_move_ms: float = Field(
        default=500.0, description="Movement phase duration (ms)"
    )
    time_post_ms: float = Field(
        default=0.0, description="Post-movement phase duration (ms)"
    )


class TrainingTimings(BaseModel):
    """Computed timing parameters for a training run.

    Derived from TaskConfig + TrainingSignalConfig + SimulationConfig.
    """

    step_ms: float
    sequence_ms: float
    input_shift_ms: float
    learning_window: float
    n_samples: int
    n_iter: int

    @classmethod
    def from_config(cls, config: "MotorControllerConfig") -> "TrainingTimings":
        task = config.task
        step_ms = config.simulation.step
        training = config.training
        sequence_ms = (
            training.time_prep_ms + training.time_move_ms + training.time_post_ms
        )
        learning_window = sequence_ms - task.learning_start_ms
        if learning_window <= 0:
            _log.warning(
                "learning_start_ms is at or past the end of sequence; "
                "learning window will be empty or invalid",
                learning_start_ms=task.learning_start_ms,
                sequence_ms=sequence_ms,
            )
        return cls(
            step_ms=step_ms,
            sequence_ms=sequence_ms,
            input_shift_ms=task.input_shift_ms,
            learning_window=learning_window,
            n_samples=len(training.trajectories),
            n_iter=task.n_iter,
        )

    @property
    def n_timesteps_per_sequence(self) -> int:
        return int(round(self.sequence_ms / self.step_ms))

    @property
    def task_ms(self) -> float:
        return (
            self.n_timesteps_per_sequence * self.n_samples * self.n_iter * self.step_ms
        )


class RBFConfig(BaseModel):
    """RBF/rb_neuron encoding parameters.

    All parameters for the input layer encoding. When rb_neuron is used (default),
    these configure both the RBF encoding and the rb_neuron model behavior.
    """

    num_centers: int = Field(
        default=20, description="Number of RBF centers for input encoding"
    )
    desired_min_rate: float = Field(
        default=0.0, description="Lower bound of desired-rate linspace (Hz)"
    )
    desired_max_rate: float = Field(
        default=60000.0,
        description="Upper bound of desired-rate linspace (Hz)",
    )

    # rb_neuron model parameters
    kp: float = Field(
        default=1000.0,
        description="Input gain (1000 ensures correct conversion from spikes/ms to spikes/s)",
    )
    base_rate: float = Field(default=0.0, description="Base firing rate in Hz")
    buffer_size: float = Field(
        default=10.0, description="Size of the sliding window in ms"
    )
    sdev_hz: float = Field(
        default=3600.0,
        description="Gaussian width for rb_neurons (Hz). Controls selectivity: "
        "smaller values make each center respond more narrowly to its preferred input rate.",
    )
    max_peak_rate_hz: float = Field(
        default=800.0,
        description="Maximum peak firing rate for rb_neuron (Hz)",
    )


class RecurrentNeuronConfig(BaseModel):
    """Recurrent neuron parameters."""

    C_m: float = Field(default=250.0, description="Membrane capacitance (pF)")
    c_reg: float = Field(default=300.0, description="Regularization constant")
    E_L: float = Field(default=0.0, description="Resting membrane potential (mV)")
    f_target: float = Field(default=10.0, description="Target firing rate (Hz)")
    gamma: float = Field(default=0.3, description="Learning rate scaling factor")
    I_e: float = Field(default=0.0, description="External current (pA)")
    regular_spike_arrival: bool = Field(
        default=False, description="Use regular spike arrival"
    )
    surrogate_gradient_function: str = Field(
        default="piecewise_linear", description="Surrogate gradient function"
    )
    t_ref: float = Field(default=2.0, description="Refractory period (ms)")
    tau_m: float = Field(default=20.0, description="Membrane time constant (ms)")
    V_m: float = Field(default=0.0, description="Initial membrane potential (mV)")
    V_th: float = Field(default=20.0, description="Spike threshold (mV)")


class OutputNeuronConfig(BaseModel):
    """Output neuron parameters."""

    C_m: float = Field(default=250.0, description="Membrane capacitance (pF)")
    E_L: float = Field(default=0.0, description="Resting membrane potential (mV)")
    I_e: float = Field(default=0.0, description="External current (pA)")
    loss: str = Field(
        default="mean_squared_error", description="Loss function for output neurons"
    )
    regular_spike_arrival: bool = Field(
        default=False, description="Use regular spike arrival"
    )
    tau_m: float = Field(default=20.0, description="Membrane time constant (ms)")
    V_m: float = Field(default=0.0, description="Initial membrane potential (mV)")


class NeuronsConfig(BaseModel):
    """Neuron parameters."""

    n_rec: int = Field(default=300, description="Number of recurrent neurons")
    n_out: int = Field(default=2, description="Number of output neurons")
    exc_ratio: float = Field(
        default=0.8,
        description="Fraction of excitatory neurons in recurrent population",
    )
    rec: RecurrentNeuronConfig = Field(default_factory=RecurrentNeuronConfig)
    out: OutputNeuronConfig = Field(default_factory=OutputNeuronConfig)


class OptimizerConfig(BaseModel):
    """Optimizer parameters."""

    type: Literal["gradient_descent"] = Field(
        default="gradient_descent", description="Optimizer type"
    )
    eta: float = Field(default=0.01, description="Learning rate for optimizer")
    Wmin: float = Field(description="Minimum synaptic weight (pA)")
    Wmax: float = Field(description="Maximum synaptic weight (pA)")


class ExcSynapseConfig(BaseModel):
    """Excitatory synapse parameters."""

    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(Wmin=0.0, Wmax=1000.0)
    )


class InhSynapseConfig(BaseModel):
    """Inhibitory synapse parameters."""

    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(Wmin=-1000.0, Wmax=0.0)
    )
    weight: float = Field(
        default=-400.0, description="Initial inhibitory synaptic weight (pA)"
    )


class SynapsesConfig(BaseModel):
    """Synapse parameters."""

    w_input: float = Field(default=20.0, description="Default synaptic weight (pA)")
    w_rec: float = Field(default=20.0, description="Recurrent synaptic weight (pA)")
    g: float = Field(default=4.0, description="Inhibitory/excitatory weight ratio")
    conn_bernoulli_p: float = Field(
        default=0.1, description="Connection probability for recurrent connections"
    )
    average_gradient: bool = Field(
        default=False, description="Average gradient across batch"
    )
    static_delay: float = Field(
        default=1.0, description="Delay for static synapses (ms)"
    )
    feedback_delay: float = Field(
        default=1.0, description="Delay for feedback synapses (ms)"
    )
    rate_target_delay: float = Field(
        default=1.0, description="Delay for rate target synapses (ms)"
    )
    receptor_type: int = Field(
        default=2, description="Receptor type for rate target synapses"
    )
    exc: ExcSynapseConfig = Field(default_factory=ExcSynapseConfig)
    inh: InhSynapseConfig = Field(default_factory=InhSynapseConfig)


class MultimeterRecConfig(BaseModel):
    """Multimeter recording configuration for recurrent neurons."""

    interval: float = Field(default=1.0, description="Recording interval (ms)")
    record_from: List[str] = Field(
        default=["V_m", "surrogate_gradient", "learning_signal"],
        description="Variables to record from recurrent neurons",
    )


class MultimeterOutConfig(BaseModel):
    """Multimeter recording configuration for output neurons."""

    interval: float = Field(default=1.0, description="Recording interval (ms)")
    record_from: List[str] = Field(
        default=[
            "V_m",
            "readout_signal",
            "readout_signal_unnorm",
            "target_signal",
            "error_signal",
        ],
        description="Variables to record from output neurons",
    )


class RecordingConfig(BaseModel):
    """Recording parameters."""

    n_record: int = Field(default=2, description="Number of neurons to record")
    n_record_w: int = Field(default=5, description="Number of weights to record")
    mm_rec: MultimeterRecConfig = Field(default_factory=MultimeterRecConfig)
    mm_out: MultimeterOutConfig = Field(default_factory=MultimeterOutConfig)


class PlottingConfig(BaseModel):
    """Plotting parameters."""

    do_plotting: bool = Field(default=True, description="Enable or disable plotting")


class MotorControllerConfig(BaseModel):
    """Complete configuration for the motor controller model."""

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    training: TrainingSignalConfig = Field(default_factory=TrainingSignalConfig)
    rbf: RBFConfig = Field(default_factory=RBFConfig)
    neurons: NeuronsConfig = Field(default_factory=NeuronsConfig)
    synapses: SynapsesConfig = Field(default_factory=SynapsesConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    plotting: PlottingConfig = Field(default_factory=PlottingConfig)

    git_commit: str = Field(
        default="unknown", description="Git commit hash of the training code"
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "MotorControllerConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated MotorControllerConfig instance.
        """
        path = Path(path)
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path where the YAML file will be saved.
        """
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary (for backward compatibility).

        Excludes None values to maintain compatibility with existing code that checks
        for key existence rather than None values.

        Returns:
            Dictionary representation of the configuration.
        """
        return self.model_dump(exclude_none=True)

    def hash(self) -> str:
        """
        Compute a SHA256 hash of the configuration for versioning or caching.

        Returns:
            str: The SHA256 hexadecimal digest representing the current configuration state.
        """
        """Compute a SHA256 hash of the configuration for versioning or caching."""
        config_bytes = yaml.dump(self.model_dump(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(config_bytes).hexdigest()
