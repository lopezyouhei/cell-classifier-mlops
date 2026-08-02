import numpy as np


def test_class_weights_are_inverse_frequency():
    counts = np.array([100, 10, 1000])
    n = len(counts)
    w = counts.sum() / (n * np.maximum(counts, 1))
    w = w / w.mean()

    assert w.argmax() == counts.argmin()
    assert w.argmin() == counts.argmax()


def test_weights_use_train_split_only():
    import inspect

    from src import data

    source = inspect.getsource(data.main)
    assert "train.labels" in source
    assert "test.labels" not in source
