from salstm_lwaro.optimizer import LWAROOptimizer


def test_lwaro_optimizer_finds_simple_minimum():
    optimizer = LWAROOptimizer(
        bounds={"x": [-5, 5], "y": [-5, 5]},
        population_size=6,
        iterations=5,
        seed=7,
    )

    result = optimizer.optimize(lambda params: (params["x"] - 1) ** 2 + (params["y"] + 2) ** 2)

    assert result.best_score < 25
    assert set(result.best_params) == {"x", "y"}
    assert len(result.history) == 6

