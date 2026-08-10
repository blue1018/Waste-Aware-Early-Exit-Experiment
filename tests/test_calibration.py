import torch

from waste_early_exit.calibration import calibrate_bundle, fit_temperature
from waste_early_exit.training import PredictionBundle


def test_temperature_fit_is_positive_and_does_not_increase_nll() -> None:
    logits = torch.tensor(
        [[5.0, 0.0], [4.0, 0.0], [0.0, 5.0], [0.0, 4.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 1, 0])

    fit = fit_temperature(logits, labels)

    assert fit.temperature > 0
    assert fit.after_nll <= fit.before_nll + 1e-9


def test_calibrate_bundle_keeps_labels_and_returns_each_exit() -> None:
    bundle = PredictionBundle(
        logits={
            "exit1": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            "exit2": torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
            "final": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        },
        labels=torch.tensor([0, 1]),
        sample_ids=["a", "b"],
    )

    calibrated = calibrate_bundle(bundle)

    assert set(calibrated.temperatures) == {"exit1", "exit2", "final"}
    assert set(calibrated.probabilities) == {"exit1", "exit2", "final"}
    assert torch.equal(calibrated.labels, bundle.labels)
    assert calibrated.sample_ids == ["a", "b"]
    assert all(torch.allclose(values.sum(dim=1), torch.ones(2), atol=1e-6) for values in calibrated.probabilities.values())

