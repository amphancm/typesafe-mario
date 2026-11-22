from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from typesafe_mario.rl.actions import NUM_ACTIONS
from typesafe_mario.rl.bc import load_demonstrations, train_bc
from typesafe_mario.rl.observations import OBSERVATION_DIM


def test_load_demonstrations_and_train_bc_from_real_artifact(tmp_path: Path) -> None:
    artifact = Path("artifacts/run-20261121T021124Z.jsonl")
    dataset = load_demonstrations([artifact], min_confidence=0.5)

    assert len(dataset) > 100
    assert dataset.observations.shape == (len(dataset), OBSERVATION_DIM)
    assert dataset.observations.dtype == np.float32
    assert dataset.actions.shape == (len(dataset),)
    assert dataset.action_probs.shape == (len(dataset), NUM_ACTIONS)
    assert np.all(np.isfinite(dataset.observations))

    model_path = tmp_path / "bc_test.zip"
    result = train_bc(dataset, model_path=model_path, epochs=5, batch_size=64, seed=7)

    assert result.model_path.exists()
    assert result.samples == len(dataset)
    assert 0.0 <= result.final_train_accuracy <= 1.0
    assert result.val_accuracy is not None

    loaded = PPO.load(str(model_path), device="cpu")
    action, _ = loaded.predict(dataset.observations[0], deterministic=True)
    assert 0 <= int(action) < NUM_ACTIONS
