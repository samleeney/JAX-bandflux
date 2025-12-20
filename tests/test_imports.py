def test_top_level_data_namespace():
    import jax_supernovae as js

    assert hasattr(js, "data"), "jax_supernovae should expose the data module"
    assert hasattr(js.data, "load_and_process_data"), "data module should provide load_and_process_data"
    assert hasattr(js, "load_and_process_data"), "load_and_process_data should be re-exported at top level"
