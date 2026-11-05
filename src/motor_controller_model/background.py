import nest


def connect_background_poisson(
    target_population,
    simulation_time_ms: float,
    config,
):
    """Connect background Poisson drive to a population."""

    cfg = config.background

    if not cfg.enabled:
        return None

    if cfg.rate_hz is None:
        raise ValueError(
            "background.rate_hz must be specified when background.enabled=True"
        )

    stop_ms = cfg.stop_ms
    if stop_ms is None:
        stop_ms = simulation_time_ms

    generator = nest.Create(
        "poisson_generator",
        1,
        {
            "rate": cfg.rate_hz,
            "start": cfg.start_ms,
            "stop": stop_ms,
        },
    )

    nest.Connect(
        generator,
        target_population,
        "all_to_all",
        {
            "synapse_model": "static_synapse",
            "weight": cfg.weight,
            "delay": cfg.delay,
        },
    )

    return generator
