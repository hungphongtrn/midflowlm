import json


def test_cache_dataset_init_builds_sample_map_without_loading_shards(
    tmp_path, monkeypatch
):
    from src.training.data import CacheDataset

    cache_dir = tmp_path / "train"
    cache_dir.mkdir()

    metadata = {
        "model_name": "test-model",
        "model_revision": None,
        "start_layer": 8,
        "end_layer": 11,
        "span_depth": 4,
        "seq_len": 32,
        "store_logits": False,
        "num_samples": 3,
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata))

    for shard_idx in range(3):
        (cache_dir / f"shard_{shard_idx:04d}_of_0003.pt").write_bytes(b"not used")

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("dataset init should not load shard payloads")

    monkeypatch.setattr("src.training.data.load_shard", fail_if_loaded)

    dataset = CacheDataset(tmp_path, split="train")

    assert len(dataset) == 3
    assert dataset.sample_map == [(0, 0), (1, 0), (2, 0)]
