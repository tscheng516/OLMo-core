from typing import Any, Dict, Type

import pytest
import torch
from torch.distributed.tensor import Shard, init_device_mesh

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.transformer.block import (
    HybridNormTransformerBlock,
    PeriNormTransformerBlock,
    ReorderedNormTransformerBlock,
    TransformerBlock,
)
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.utils import get_default_device, seed_all


def _build_block(
    block_cls: Type[TransformerBlock],
    *,
    d_model: int,
    init_device: str,
    attn_kwargs: Dict[str, Any],
) -> TransformerBlock:
    attn_cfg = AttentionConfig(**attn_kwargs)
    ff_cfg = FeedForwardConfig(hidden_size=4 * d_model)
    ln_cfg = LayerNormConfig()
    return block_cls(
        d_model=d_model,
        block_idx=0,
        n_layers=1,
        sequence_mixer=attn_cfg,
        feed_forward=ff_cfg,
        layer_norm=ln_cfg,
        init_device=init_device,
    )


def _run_tensor_parallel_block(
    checkpoint_dir: str,
    inputs_path: str,
    outputs_path: str,
    block_cls: Type[TransformerBlock],
    d_model: int,
    attn_kwargs: Dict[str, Any],
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    block = _build_block(
        block_cls, d_model=d_model, init_device=device.type, attn_kwargs=attn_kwargs
    )

    # Shard sequence dim in/out like the transformer model does.
    block.apply_tp(mesh["tp"], input_layout=Shard(1))
    load_model_and_optim_state(checkpoint_dir, block)

    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk = x.size(1) // world_size
    x_local = x[:, rank * chunk : (rank + 1) * chunk, :]
    y_local = block(x_local)

    # Backward to exercise graph in TP mode.
    y_local.sum().backward()

    y_ref = torch.load(outputs_path, map_location=device)
    y_ref_local = y_ref[:, rank * chunk : (rank + 1) * chunk, :]
    torch.testing.assert_close(y_ref_local, y_local.to_local())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "attn_kwargs",
    [
        pytest.param(dict(n_heads=8), id="default"),
        pytest.param(dict(n_heads=8, rope=None, bias=False), id="no-bias"),
    ],
)
@pytest.mark.parametrize(
    "block_cls", [TransformerBlock, ReorderedNormTransformerBlock, PeriNormTransformerBlock]
)
def test_tensor_parallel_transformer_block(
    backend: str, block_cls: Type[TransformerBlock], attn_kwargs: Dict[str, Any], tmp_path
):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")

    seed_all(0)
    d_model = 128
    attn_kwargs = {**attn_kwargs, "name": AttentionType.default, "use_flash": False}

    block = _build_block(
        block_cls, d_model=d_model, init_device=device.type, attn_kwargs=attn_kwargs
    )

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, d_model, device=device)
    y = block(x)

    outputs_path = tmp_path / "block_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "block_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, block)

    run_distributed_test(
        _run_tensor_parallel_block,
        backend=backend,
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, block_cls, d_model, attn_kwargs),
    )


def test_hybrid_norm_transformer_block_structure():
    """Verify HybridNormTransformerBlock has post-attention norm, qk_norm, v_norm, no FFN norm."""
    d_model = 64
    attn_kwargs = {"name": AttentionType.default, "n_heads": 4, "use_flash": False}
    block = _build_block(
        HybridNormTransformerBlock, d_model=d_model, init_device="cpu", attn_kwargs=attn_kwargs
    )

    # Key modules must be present.
    assert hasattr(block, "attention"), "Block should have attention"
    assert hasattr(block, "feed_forward"), "Block should have feed_forward"
    assert hasattr(block, "attention_norm"), "Block should have attention_norm (post-attn)"
    assert hasattr(
        block, "attention_residual_stream"
    ), "Block should have attention_residual_stream"
    assert hasattr(
        block, "feed_forward_residual_stream"
    ), "Block should have feed_forward_residual_stream"
    # Feed-forward norm must NOT be present.
    assert not hasattr(block, "feed_forward_norm"), "Block should NOT have feed_forward_norm"
    # qk_norm and v_norm must be set on the attention module.
    assert block.attention.q_norm is not None, "Attention should have q_norm"
    assert block.attention.k_norm is not None, "Attention should have k_norm"
    assert block.attention.v_norm is not None, "Attention should have v_norm"


def test_hybrid_norm_transformer_block_forward():
    """HybridNormTransformerBlock forward pass should run without errors."""
    d_model = 64
    attn_kwargs = {"name": AttentionType.default, "n_heads": 4, "use_flash": False}
    block = _build_block(
        HybridNormTransformerBlock, d_model=d_model, init_device="cpu", attn_kwargs=attn_kwargs
    )

    bs, seq_len = 2, 16
    x = torch.randn(bs, seq_len, d_model)
    y = block(x)
    assert y.shape == x.shape


def test_hybrid_norm_transformer_block_via_config():
    """Selecting TransformerBlockType.hybrid_norm via config dispatch builds the correct block."""
    from olmo_core.nn.transformer.config import (
        TransformerBlockConfig,
        TransformerBlockType,
    )

    d_model = 64
    attn_cfg = AttentionConfig(name=AttentionType.default, n_heads=4, use_flash=False)
    ff_cfg = FeedForwardConfig(hidden_size=4 * d_model)
    ln_cfg = LayerNormConfig()

    block_cfg = TransformerBlockConfig(
        sequence_mixer=attn_cfg,
        feed_forward=ff_cfg,
        layer_norm=ln_cfg,
        name=TransformerBlockType.hybrid_norm,
    )
    block = block_cfg.build(d_model=d_model, block_idx=0, n_layers=1)

    assert isinstance(block, HybridNormTransformerBlock)
    assert not hasattr(block, "feed_forward_norm")
    assert block.attention.q_norm is not None
    assert block.attention.k_norm is not None
    assert block.attention.v_norm is not None

    bs, seq_len = 2, 16
    x = torch.randn(bs, seq_len, d_model)
    y = block(x)
    assert y.shape == x.shape


def test_hybrid_norm_num_params():
    """
    HybridNormTransformerBlock num_params accounts for one block-level norm (not two),
    plus auto-added qk_norm (q_norm + k_norm) and v_norm in the attention.
    """
    from olmo_core.nn.transformer.config import (
        TransformerBlockConfig,
        TransformerBlockType,
    )

    d_model = 64
    n_heads = 4
    attn_cfg = AttentionConfig(name=AttentionType.default, n_heads=n_heads, use_flash=False)
    ff_cfg = FeedForwardConfig(hidden_size=4 * d_model)
    ln_cfg = LayerNormConfig()

    hybrid_cfg = TransformerBlockConfig(
        sequence_mixer=attn_cfg,
        feed_forward=ff_cfg,
        layer_norm=ln_cfg,
        name=TransformerBlockType.hybrid_norm,
    )

    # Build the block and count actual parameters.
    block = hybrid_cfg.build(d_model=d_model, block_idx=0, n_layers=1)
    actual_params = sum(p.numel() for p in block.parameters())
    assert hybrid_cfg.num_params(d_model) == actual_params

    # The attention should have qk_norm and v_norm set automatically.
    assert block.attention.q_norm is not None
    assert block.attention.k_norm is not None
    assert block.attention.v_norm is not None
