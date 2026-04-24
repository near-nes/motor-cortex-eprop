def test_imports():
    try:
        import motor_controller_model
        import motor_controller_model.plot_results
        import motor_controller_model.nestml_neurons.compile_nestml_neurons
        import motor_controller_model.config_schema
        import motor_controller_model.m1_factory
        import motor_controller_model.m1_network
        import motor_controller_model.m1_training
        import motor_controller_model.run_m1
        import motor_controller_model.signals
        import motor_controller_model.utils
    except Exception as e:
        assert False, f"Import failed: {e}"
