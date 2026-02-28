"""Pydantic schema for motor controller model configuration."""

import hashlib
from pathlib import Path
from typing import List, Literal
import yaml

from pydantic import BaseModel, Field, field_validator


class SimulationConfig(BaseModel):
    """Simulation and environment parameters."""

    rng_seed: int = Field(default=1234, description="Random seed for reproducibility")
    print_time: bool = Field(default=False, description="Print simulation progress")
    total_num_virtual_procs: int = Field(
        default=4, description="Number of virtual processes for NEST"
    )
    step: float = Field(default=1.0, description="Simulation time step (ms)")


class TaskConfig(BaseModel):
    """Task and experiment setup parameters."""

    samples_per_trajectory_in_dataset: int = Field(
        default=10, description="Describes the structure of your dataset file (legacy)"
    )
    trajectory_ids_to_use: List[int] = Field(
        default=[0, 1], description="Specific list of trajectory IDs to use (legacy)"
    )
    n_samples_per_trajectory_to_use: int = Field(
        default=1,
        description="How many samples to take from the start of each trajectory (legacy)",
    )
    gradient_batch_size: int = Field(
        default=1, description="Batch size for gradient computation"
    )
    n_iter: int = Field(default=100, description="Number of training iterations")
    sequence: float = Field(default=1500.0, description="Sequence length (ms)")
    silent_period: float = Field(
        default=0.0, description="Silent period before receiving the trajectory (ms)"
    )
    input_shift_ms: float = Field(
        default=100.0,
        description="Temporal delay to shift M1 target spikes forward (ms)",
    )
    learning_start: float = Field(
        default=650.0,
        description="Start time (ms) inside one sequence where learning becomes active",
    )
    learning_end: float = Field(
        default=1500.0,
        description="End time (ms) inside one sequence where learning stops",
    )


class RBFConfig(BaseModel):
    """RBF/rb_neuron encoding parameters.

    All parameters for the input layer encoding. When rb_neuron is used (default),
    these configure both the RBF encoding and the rb_neuron model behavior.
    """

    # Core RBF encoding parameters
    num_centers: int = Field(
        default=20, description="Number of RBF centers for input encoding"
    )
    width: float = Field(
        default=0.06, description="RBF width in rad (standard deviation)"
    )

    # Trajectory mode parameters
    scale_rate: float = Field(
        default=500.0, description="RBF output scaling factor for trajectory mode (Hz)"
    )
    shift_min_rate: float = Field(
        default=0.0, description="Minimum rate shift for RBF output (Hz)"
    )

    # Spike-input mode parameters
    desired_upper_hz: float = Field(
        default=60000.0,
        description="Desired upper firing rate for spike-input mode (Hz). Used to compute rb_neuron sdev.",
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

    # Optional computed parameters (computed from above if not provided)
    sdev_hz: float | None = Field(
        default=3600.0,
        description="Standard deviation for rb_neurons (Hz). If None, computed based on mode: "
        "spike-input mode uses desired_upper_hz * width, "
        "trajectory mode uses scale_rate * width",
    )
    max_peak_rate_hz: float | None = Field(
        default=800.0,
        description="Maximum peak firing rate for rb_neuron (Hz). If None, computed based on mode: "
        "spike-input mode uses desired_upper_hz, "
        "trajectory mode uses scale_rate / step",
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
