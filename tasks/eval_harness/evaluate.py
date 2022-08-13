from ensurepip import bootstrap
from functools import reduce, partial
from logging import logMultiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))

from lm_eval.models.gpt2 import GPT2LM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from tqdm import tqdm
import torch.nn.functional as F

from lm_eval.tasks import ALL_TASKS
from pretrain_gpt import model_provider
import numpy as np
import time

import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.training import setup_model_and_optimizer, get_model
from megatron.mpu.mappings import gather_from_tensor_model_parallel_region

from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
import pickle
import json

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module
from deepspeed.runtime.pipe import schedule


class EvalHarnessAdaptor(GPT2LM):
    def __init__(self, model, tokenizer):
        args = get_args()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = tokenizer.vocab_size
        self._eot_token_id = tokenizer.eod_id

        self._max_length = args.seq_length

        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_world_size = mpu.get_data_parallel_world_size()

        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self._batch_size = args.micro_batch_size * self.dp_world_size

        self.cache_hook = CacheHook(None)
        self.is_main = args.rank == 0
        self.is_local_main = args.local_rank == 0
        self._device = torch.cuda.current_device()
        self.is_model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.is_data_parallel = mpu.get_data_parallel_world_size() > 1
        self.adaptive_seq_len = args.adaptive_seq_len
        self.is_last_stage = True if not self.is_pipe_parallel else mpu.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self._eos_token_id

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(
                tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size
            ):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen
                    if not self.adaptive_seq_len:
                        padding_length = self.max_length
                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))

                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)
                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1).cpu()  # [batch, seq, vocab]

                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                        chunk, multi_logits, inps, inplens, contlens
                    ):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                        max_equal = (greedy_tokens == cont_toks).all()
                        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))
                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                        res.append(answer)

            # broadcast results to all ranks
            if self.is_pipe_parallel:
                src_rank = self.model.grid.stage_to_global(self.model.num_stages - 1)
                if res:
                    logits_sums, max_equals = list(zip(*res))
                    logits_sums = torch.FloatTensor(logits_sums).cuda()
                    max_equals = torch.LongTensor(max_equals).cuda()
                else:
                    logits_sums = torch.zeros(res_len, dtype=torch.float32).cuda()
                    max_equals = torch.zeros(res_len, dtype=torch.int64).cuda()
                torch.distributed.broadcast(
                    tensor=logits_sums,
                    src=src_rank,
                    group=mpu.get_pipeline_model_parallel_group(),
                )
                torch.distributed.broadcast(
                    tensor=max_equals, src=src_rank, group=mpu.get_pipeline_model_parallel_group()
                )
                max_equals = [bool(i) for i in max_equals.tolist()]
                logits_sums = logits_sums.tolist()
                res = list(zip(logits_sums, max_equals))

        return reord.get_original(res)

    def _dp_scatter(self, inps):
        """
        Scatters the inputs to all data parallel ranks.
        """

        batch_size = inps.shape[0]
        padded = False
        if batch_size % self.dp_world_size != 0:
            # The last batch could potentially not fill the full batch size (if the dataset size is not divisible by batch size)
            # In this case we pad the batch
            padded_size = self.dp_world_size - (batch_size % self.dp_world_size)

            print_rank_0(
                f"WARNING: Batch size ({batch_size}) must be divisible by dp world size ({self.dp_world_size}). Padding inputs to {padded_size}."
            )

            inps = torch.cat(
                [inps] + [inps[0:1, ...] for _ in range(padded_size)], dim=0
            )  # pad with first inp item
            padded = True

        assert (
            inps.shape[0] % self.dp_world_size == 0
        ), f"batch size ({inps.shape[0]}) must be divisible by dp world size ({self.dp_world_size})"

        # get a chunk for each data parallel rank
        chunk_size = inps.shape[0] // self.dp_world_size
        inps = inps[self.dp_rank * chunk_size : (self.dp_rank + 1) * chunk_size]
        # make a dummy dataloader / iterator to pass to model
        # we need to do this because deepspeed pipe parallel only takes an iterator
        # in this format
        return iter([{"text": F.pad(inps, pad=(0, 1))}]), padded

    def _dp_gather(self, logits):
        """
        Gather logits from all data parallel ranks
        """
        if self.dp_world_size == 1:
            return logits

        if logits is not None:
            tensor_list = [torch.zeros_like(logits) for _ in range(self.dp_world_size)]
            torch.distributed.all_gather(
                tensor_list, logits, group=mpu.get_data_parallel_group()
            )
            logits = torch.cat(tensor_list, dim=0)
            return logits

    def _model_call(self, inps):
        args = get_args()

        batch_size = inps.shape[0]

        # scatter inputs to all dp ranks:
        inps, padded = self._dp_scatter(inps)

        # need these flags to stop deepspeed pipe parallel from hanging
        self.model.first_output_send = True
        self.model.pipe_recv_buf = None
        self.model.grad_layer = None
        self.model.meta_buffer = None

        _, logits = self.model.eval_batch(
            inps,
            compute_loss=False,
            reduce_output=None,
            return_logits=True)

        if logits is not None:
            logits = logits[0]

        # if logits is not None:
        #     if isinstance(output[0], (tuple, list)):
        #         output = [x[0] for x in output]
        #     output = torch.cat(output, 0)
        # else:
        #     output = None

        # hack #2 for adaptive_seq_len to work as total_loss gets appended to and shapes aren't the same
        if args.adaptive_seq_len:
            self.model.total_loss = None

        # logits = output

        # gather outputs from all dp ranks:
        logits = self._dp_gather(logits)

        # if logits have been padded (normally just last item where batch size is unequal)
        # restore to original shape
        if padded and logits is not None:
            logits = logits[:batch_size, ...]

        return logits

    def tok_encode(self, text):
        """Tokenize text *without* adding special tokens."""
        # Splitting this into its own method in case we need to handle special cases for different tokenizers
        from megatron.tokenizer.gpt2_tokenization import GPT2Tokenizer
        if isinstance(self.tokenizer.tokenizer, GPT2Tokenizer):
            return self.tokenizer.tokenizer.encode(text)
        else:
            return self.tokenizer.tokenizer.encode(text, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def run_eval(
        self,
        eval_tasks=None,
        num_fewshot=0,
        bootstrap_iters=2,
        description_dict=None,
        name="deepspeed",
        limit=None
    ):
        was_training = self.model.training
        self.model.eval()
        in_micro_batches = (
            self.model.micro_batches
        )  # store input microbatches - we need to set to 1 during eval, but want to return to its original value after
        self.model.micro_batches = 1
        if eval_tasks is None:
            eval_tasks = [
                "lambada",
                "piqa",
                "hellaswag",
                "winogrande",
                "mathqa",
                "pubmedqa",
            ]

        # **HACK INCOMING**:
        # first get task dict on local main rank
        # the tasks are downloaded *as they are initialized*, and the downloads don't like multithreading.
        # so we download them once on the local main rank, wait, and then initialize them on all other ranks, which *should* load from the cache.
        if self.is_local_main:
            task_dict = tasks.get_task_dict(eval_tasks)
        # torch barrier
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        task_dict = tasks.get_task_dict(eval_tasks)

        lm = self

        results = evaluator.evaluate(
            lm=lm,
            task_dict=tasks.get_task_dict(eval_tasks),
            description_dict=description_dict,
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
        )

        results["config"] = {
            "model": name,
            "num_fewshot": num_fewshot,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "description_dict": description_dict
        }

        if was_training:
            self.model.train()
        self.model.micro_batches = in_micro_batches
        return results



from megatron.initialize import initialize_megatron
import megatron

from tools.convert_checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
from tools.convert_checkpoint.deepspeed_to_megatron import _create_rank_checkpoint

def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)


# Note(Hesslow):
# The model loading is a bit convoluted.
# We want to parse out the model arguments from the checkpoint and use those to initialize megatron-ds.
#
# However megatron-ds expects its arguments on the command line.
# And at that point we don't know them.
#
# Instead we use Jasons way: we load the arguments form the checkpoint and then override _parse_args to return whatever args we want.
#
# If the checkpoint is old, some new arguments may have been introduced and the code will expect these arguments to exist.
# In order to support this we _first_ parse the arguments normally, and then override them with the arguments from the checkpoint.
# Keeping the default-value of newer arguments.
#
# We then use the megatron deepspeed converter to load the deepspeed checkpoints as if they we're megatron checkpoints.
def load_ds_checkpoint_and_setup_megatron(extra_args_provider):
    # parse the megatorn args. But wait with initalizing megatron.
    # avoid printing the arguments, since they will later be overridden.
    _print_args = megatron.arguments._print_args
    megatron.arguments._print_args = lambda *_args, **kwarg: None
    args = _parse_args(extra_args_provider)

    ds_checkpoint = DeepSpeedCheckpoint(args.load,
                                        tp_degree=args.tensor_model_parallel_size,
                                        pp_degree=args.pipeline_model_parallel_size,
                                        no_pp=args.no_pipeline_parallel)


    cp_args = ds_checkpoint.get_args()
    # Merge the current args with the checkpoint args.
    skip_keys = ['world_size', 'rank', 'local_rank','device_count', 'micro_batch_size','global_batch_size', 'batch_size', 'tensorboard_dir', 'deepspeed', 'deepspeed_config',
                     'data_parallel_size', 'pipeline_model_parallel_size', 'tensor_model_parallel_size', 'moe_expert_parallel_size', 'moe_token_dropping', 'load', 'rampup_batch_size', 'iteration', 'inference',
                     'consumed_train_samples', 'consumed_valid_samples', 'load_tag']

    skip_if_specified = ['merge_file', 'vocab_file']

    if args.eval_fp32:
        cp_args.fp16 = False
        cp_args.bf16 = False
        cp_args.params_dtype = torch.float32

    override_args(args, cp_args, skip_keys, skip_if_specified)

    # stop megatron from reparsing the arguments.
    megatron.global_vars._parse_args = lambda *_args, **kwarg: args
    megatron.global_vars._GLOBAL_ARGS = args

    initialize_megatron()
    torch.distributed.barrier()

    # Initializing megatron will update eg. tokenizer size. Override again.
    override_args(args, cp_args, skip_keys, skip_if_specified)

    if args.deepspeed and args.adaptive_seq_len:
        # adaptive_seq_len hack #1:
        # CL automatically enables reset_activation_shape() which allows us to change input shapes
        # and it also reshapes the attenion scores in attention_mask_func
        args.curriculum_learning = 1
        args.curriculum_seqlen = args.seq_length

    # print final arguments.
    _print_args(args)

    model, _, _ = setup_model_and_optimizer(model_provider)
    model = model[0]
    print_rank_0("Finished loading model")

    if args.eval_fp32:
        model = model.float()

    torch.distributed.barrier()
    return model

def tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Evaluation options')
    group.add_argument('--task_list', type=str, default = "all", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--results_path', type=str, default = "./results.json", help='Path to where the results will be stored.')
    group.add_argument('--adaptive_seq_len',  default = False, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--eval_fp32',  default = False, action='store_true', help='Should the evaluation run in fp32')
    return parser

from megatron.global_vars import _parse_args

def main():
    start = time.time()
    model = load_ds_checkpoint_and_setup_megatron(extra_args_provider=tasks_args)

    args = get_args()

    task_list = ALL_TASKS if args.task_list == 'all' else args.task_list.split(',')

    model.module.activation_checkpoint_interval = 0
    model._compute_loss = False
    model.fwd_outputs = []

    tokenizer = get_tokenizer()
    adaptor = EvalHarnessAdaptor(model, tokenizer)
    results = adaptor.run_eval(
        eval_tasks=task_list,
        num_fewshot=0,
        bootstrap_iters=10000,
    )

    if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        print(json.dumps(results, indent=2))
        with open(args.results_path, 'w') as outfile:
            json.dump(results, outfile, indent = 4)
    end = time.time()
    print_rank_0("evaluation of {} ends in {:.2f} sec, or {:.2f} min, or {:.2f} hr".format(args.task_list, end-start, (end-start)/60.0, (end-start)/3600.0))

if __name__ == '__main__':
    main()
