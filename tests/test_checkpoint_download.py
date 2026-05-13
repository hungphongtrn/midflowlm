from unittest.mock import patch

import pytest

from scripts.train_sft import _maybe_download_checkpoint, _resolve_hf_token


def test_resolve_hf_token_prefers_cli_arg():
    assert _resolve_hf_token("cli-token") == "cli-token"


def test_resolve_hf_token_uses_hf_token_env():
    with patch.dict("os.environ", {"HF_TOKEN": "env-token"}, clear=True):
        assert _resolve_hf_token(None) == "env-token"


def test_resolve_hf_token_uses_hf_hub_env_when_hf_token_missing():
    with patch.dict("os.environ", {"HUGGING_FACE_HUB_TOKEN": "hub-env-token"}, clear=True):
        assert _resolve_hf_token(None) == "hub-env-token"


def test_resolve_hf_token_falls_back_to_hf_cli_login_token():
    with patch.dict("os.environ", {}, clear=True):
        with patch("huggingface_hub._login.get_token", return_value="login-token"):
            assert _resolve_hf_token(None) == "login-token"


def test_resolve_hf_token_returns_none_when_all_sources_missing():
    with patch.dict("os.environ", {}, clear=True):
        with patch("huggingface_hub._login.get_token", side_effect=RuntimeError("missing")):
            assert _resolve_hf_token(None) is None


def test_maybe_download_checkpoint_returns_local_path_when_exists():
    cfg = {"path": "models/checkpoint.pth"}
    with patch("scripts.train_sft.os.path.exists", return_value=True):
        assert _maybe_download_checkpoint(cfg, token=None) == "models/checkpoint.pth"


def test_maybe_download_checkpoint_downloads_when_missing_and_remote_config_present():
    cfg = {
        "path": "models/checkpoint.pth",
        "remote_repo": "org/repo",
        "remote_filename": "checkpoint.pth",
    }
    with patch("scripts.train_sft.os.path.exists", return_value=False):
        with patch("scripts.train_sft.hf_hub_download", return_value="models/checkpoint.pth") as mock_download:
            result = _maybe_download_checkpoint(cfg, token=None)

    assert result == "models/checkpoint.pth"
    mock_download.assert_called_once_with(
        repo_id="org/repo",
        filename="checkpoint.pth",
        local_dir="models",
        local_dir_use_symlinks=False,
        token=None,
    )


def test_maybe_download_checkpoint_raises_when_missing_without_remote_config():
    cfg = {"path": "models/checkpoint.pth"}
    with patch("scripts.train_sft.os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            _maybe_download_checkpoint(cfg, token=None)


def test_maybe_download_checkpoint_passes_token_to_download():
    cfg = {
        "path": "models/checkpoint.pth",
        "remote_repo": "org/repo",
        "remote_filename": "checkpoint.pth",
    }
    with patch("scripts.train_sft.os.path.exists", return_value=False):
        with patch("scripts.train_sft.hf_hub_download", return_value="models/checkpoint.pth") as mock_download:
            _maybe_download_checkpoint(cfg, token="secret")

    assert mock_download.call_args.kwargs["token"] == "secret"


def test_maybe_download_checkpoint_with_nested_remote_filename_uses_parent_local_dir():
    cfg = {
        "path": "models/p3_d3_mix_c/checkpoint.pth",
        "remote_repo": "org/repo",
        "remote_filename": "p3_d3_mix_c/checkpoint.pth",
    }
    with patch("scripts.train_sft.os.path.exists", return_value=False):
        with patch(
            "scripts.train_sft.hf_hub_download",
            return_value="models/p3_d3_mix_c/checkpoint.pth",
        ) as mock_download:
            result = _maybe_download_checkpoint(cfg, token=None)

    assert result == "models/p3_d3_mix_c/checkpoint.pth"
    mock_download.assert_called_once_with(
        repo_id="org/repo",
        filename="p3_d3_mix_c/checkpoint.pth",
        local_dir="models",
        local_dir_use_symlinks=False,
        token=None,
    )


def test_sft_flow_midblock_model_exposes_hf_save_ignore_keys_attribute():
    from src.model.sft_flow_midblock import SFTFlowMidblockModel

    assert hasattr(SFTFlowMidblockModel, "_keys_to_ignore_on_save")
