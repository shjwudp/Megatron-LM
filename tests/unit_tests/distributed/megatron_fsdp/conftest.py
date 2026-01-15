from functools import partial

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import megatron.core.parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model
from megatron.training.utils import is_first_or_last_pipeline_stage


@pytest.fixture
def pretrain_forward_backward_factory():
    """Factory fixture that returns a get_batch-like callable + iterator."""

    def _make_forward_backward_func(
        *, model, data_iterator, sequence_length=128, micro_batch_size=2, num_micro_batches=1
    ):
        forward_backward_func = get_forward_backward_func()
        forward_backward_func(
            forward_step_func=_forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_micro_batches,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )

    return _make_forward_backward_func


@pytest.fixture
def gpt_mock_data_iterator_factory():
    def _make_gpt_mock_data_iterator(
        dp_group, num_samples=1000, vocab_size=50257, sequence_length=128, batch_size=8, seed=42
    ):
        dataset = GPTMockDataset(
            num_samples=num_samples,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            seed=seed,
        )
        sampler = DistributedSampler(dataset, num_replicas=dp_group.size(), rank=dp_group.rank())
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        for batch in dataloader:
            batch["position_ids"] = torch.arange(sequence_length, dtype=torch.int64)
            yield batch

    return _make_gpt_mock_data_iterator


@pytest.fixture
def moe_model_factory():
    def _make_moe_model(vocab_size, max_sequence_length, **overrides):
        base_cfg = dict(
            attention_backend="unfused",
            deterministic_mode=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            num_layers=4,
            hidden_size=128,
            add_bias_linear=False,
            num_attention_heads=32,
            ffn_hidden_size=128,
            kv_channels=32,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            multi_latent_attention=True,
            num_moe_experts=8,
            deallocate_pipeline_outputs=True,
            pipeline_model_parallel_size=1,
            tensor_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=1,
            sequence_parallel=True,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=None,
        )

        base_cfg.update(overrides)
        config = TransformerConfig(**base_cfg)

        # Build model
        transformer_layer_spec = get_gpt_decoder_block_spec(
            config=config, use_transformer_engine=True
        )
        mtp_block_spec = get_gpt_mtp_block_spec(
            config, transformer_layer_spec.layer_specs[-1], True
        )
        gpt_model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            mtp_block_spec=mtp_block_spec,
            vocab_size=vocab_size,
            pre_process=mpu.is_pipeline_first_stage(),
            post_process=mpu.is_pipeline_last_stage(),
            max_sequence_length=max_sequence_length,
        )

        return gpt_model

    return _make_moe_model


class GPTMockDataset(Dataset):
    """
    Mock dataset for torchtitan GPT training tests
    Generates synthetic tokenized sequences on-the-fly
    """

    def __init__(
        self,
        num_samples=10000,
        micro_batch_size=1,
        sequence_length=2048,
        vocab_size=128256,
        seed=42,
    ):
        """
        Initialize mock dataset

        Args:
            num_samples: Total number of samples
            sequence_length: Length of each sequence
            vocab_size: Size of vocabulary
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.micro_batch_size = micro_batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.seed = seed

        # Set numpy seed for deterministic generation
        np.random.seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a single training sample

        Returns:
            dict with 'tokens' and 'labels'
        """
        # Use idx as seed for reproducible but varied samples
        rng = np.random.RandomState(self.seed + idx)

        # Generate random token sequence
        tokens = rng.randint(0, self.vocab_size, size=self.sequence_length, dtype=np.int64)

        # Labels are tokens shifted by 1 (next token prediction)
        labels = 1 + tokens

        return {
            'tokens': torch.from_numpy(tokens.copy()),
            'labels': torch.from_numpy(labels.copy()),
            "attention_mask": torch.ones(
                (1, self.sequence_length, self.sequence_length), dtype=bool
            ),
            "loss_mask": torch.ones(self.sequence_length),
        }


def _forward_step_func(data_iterator, model, device="cuda"):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    vp_stage = get_attr_wrapped_model(model, "vp_stage")

    if not is_first_or_last_pipeline_stage(vp_stage):
        tokens, labels, loss_mask, attention_mask, position_ids = None, None, None, None, None
    else:
        data = next(data_iterator)
        tokens = data["tokens"].to(device, non_blocking=True)
        labels = data["labels"].to(device, non_blocking=True)
        loss_mask = data["loss_mask"].to(device, non_blocking=True)
        attention_mask = (
            None
            if "attention_mask" not in data
            else data["attention_mask"].to(device, non_blocking=True)
        )
        position_ids = data["position_ids"].to(device, non_blocking=True)

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)
