def test_imports():
    try:
        import motor_controller_model
        import motor_controller_model.eprop_reaching_task
        import motor_controller_model.plot_results
        import motor_controller_model.dataset_motor_training.load_dataset
        import motor_controller_model.nestml_neurons.compile_nestml_neurons
    except Exception as e:
        assert False, f"Import failed: {e}"
