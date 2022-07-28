from megatron.model import GPTModelPipe
from megatron import get_args
from megatron.optimizer import Adafactor

import copy
import os
import torch
import numpy as np
import pandas as pd
import deepspeed
import mup
from mup.optim import MuAdam
from mup import coord_check as mup_coord_check


def coord_check(mup_flag, data_iterator, batch_fn, lr, plotdir='', legend=False):
    args = get_args()

    hidden_size_copy = args.hidden_size

    lr = 0.01
    coord_check_nseeds = args.coord_check_nseeds
    coord_check_nsteps = args.coord_check_nsteps

    def gen(w, standparam=False):
        def f():
            args.hidden_size = w
            args.ffn_hidden_size = 4 * args.hidden_size
            args.kv_channels = args.hidden_size // args.num_attention_heads

            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True
            )
            if standparam:
                mup.set_base_shapes(model, None)
            else:
                load_base_shapes = f"{args.load_base_shapes}.{torch.distributed.get_rank()}"
                mup.set_base_shapes(model, load_base_shapes)

                # mup parameter initialization
                for _, sub_module in model.named_modules():
                    if hasattr(sub_module, "mup_initialize"):
                        sub_module.mup_initialize(init_method_std=args.init_method_std)
            if args.optimizer == "adafactor":
                optimizer = Adafactor(model.parameters(), mup=True, beta1=0.9, dynamic_weight_decay=True)
            elif args.optimizer == "adam":
                optimizer = MuAdam(model.parameters(), lr=lr)
            else:
                raise Exception("Unexpected optimizer {}".format(args.optimizer))

            model, _, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                args=args,
                mpu=None,
            )
            model.set_batch_fn(batch_fn)
            return model
        return f


    widths = 2 ** np.arange(7, 11)
    models = {w: gen(w, standparam=not mup_flag) for w in widths}

    optimizer = copy.deepcopy("adam")

    df = get_coord_data(models, data_iterator, mup=mup_flag, lr=lr, optimizer=optimizer,
        nseeds=coord_check_nseeds, nsteps=coord_check_nsteps)

    prm = 'μP' if mup_flag else 'SP'
    if torch.distributed.get_rank() == 0:
        mup_coord_check.plot_coord_data(df, legend=legend,
            save_to=os.path.join(plotdir, f'{prm.lower()}_trsfmr_{optimizer}_coord.png'),
            suptitle=f'{prm} Transformer {optimizer} lr={lr} nseeds={coord_check_nseeds}',
            face_color='xkcd:light grey' if not mup_flag else None)
    torch.distributed.barrier()

    # recovery model setting
    args.hidden_size = hidden_size_copy
    args.ffn_hidden_size = 4 * args.hidden_size
    args.kv_channels = args.hidden_size // args.num_attention_heads


def get_coord_data(models, dataloader, optimizer='sgd', lr=None, mup=True,
                    filter_trainable_by_name=None,
                    **kwargs):
    '''Get coord data for coord check.

    Train the models in `models` with data from `dataloader` and optimizer
    specified by `optimizer` and `lr` for `nsteps` steps, and record coordinate
    statistics specified by `output_fdict`, `input_fdict`, `param_fdict`. By
    default, only `l1` is computed for output activations of each module.

    This function wraps around `_get_coord_data`, with the main difference being
    user can specify common optimizers via a more convenient interface.

    Inputs:
        models: 
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optimizer:
            a string in `['sgd', 'adam', 'adamw']`, with default being `'sgd'`.
        lr: 
            learning rate. By default is 0.1 for `'sgd'` and 1e-3 for others.
        mup: 
            If True, then use the optimizer from `mup.optim`; otherwise, use the
            one from `torch.optim`.
        filter_trainable_by_name: 
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be trained.
        nsteps: 
            number of steps to train the model
        dict_in_out:
            whether the data loader contains Huggingface-style dict input and
            output. Default: False
        flatten_input:
            if not `dict_in_out`, reshape the input to be
            `input.view(input.shape[0], -1)`. Typically used for testing MLPs.
        flatten_output:
            if not `dict_in_out`, reshape the label to be `label.view(-1,
            input.shape[-1])`.
        output_name:
            if `dict_in_out`, this is the key for the loss value if the output
            is a dict. If the output is not a dict, then we assume the first
            element of the output is the loss.
        lossfn:
            loss function to use if not `dict_in_out`. Default is `xent` for
            cross entropy loss. Other choices are ['mse', 'nll']
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict: 
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm.
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).
    '''
    if lr is None:
        lr = 0.1 if optimizer == 'sgd' else 1e-3
    if mup:
        from mup.optim import MuAdam as Adam
        from mup.optim import MuAdamW as AdamW
        from mup.optim import MuSGD as SGD
    else:
        from torch.optim import SGD, Adam, AdamW
    def get_trainable(model):
        params = model.parameters()
        if filter_trainable_by_name is not None:
            params = []
            for name, p in model.named_parameters():
                if filter_trainable_by_name(name):
                    params.append(p)
        return params
    if optimizer == 'sgd':
        optcls = lambda model: SGD(get_trainable(model), lr=lr)
    elif optimizer == 'adam':
        optcls = lambda model: Adam(get_trainable(model), lr=lr)
    elif optimizer == 'adamw':
        optcls = lambda model: AdamW(get_trainable(model), lr=lr)
    elif optimizer is None:
        raise ValueError('optimizer should be sgd|adam|adamw or a custom function')
    
    data = _get_coord_data(models, dataloader, optcls, **kwargs)
    data['optimizer'] = optimizer
    data['lr'] = lr
    return data


def _get_coord_data(models, dataloader, optcls, nsteps=3,
                dict_in_out=False, flatten_input=False, flatten_output=False, 
                output_name='loss', lossfn='xent', filter_module_by_name=None,
                fix_data=True, cuda=True, nseeds=1, 
                output_fdict=None, input_fdict=None, param_fdict=None,
                show_progress=True):
    '''Inner method for `get_coord_data`.

    Train the models in `models` with optimizer given by `optcls` and data from
    `dataloader` for `nsteps` steps, and record coordinate statistics specified
    by `output_fdict`, `input_fdict`, `param_fdict`. By default, only `l1` is
    computed for output activations of each module.

    Inputs:
        models: 
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optcls: 
            a function so that `optcls(model)` gives an optimizer used to train
            the model.
        nsteps: 
            number of steps to train the model
        dict_in_out:
            whether the data loader contains Huggingface-style dict input and
            output. Default: False
        flatten_input:
            if not `dict_in_out`, reshape the input to be
            `input.view(input.shape[0], -1)`. Typically used for testing MLPs.
        flatten_output:
            if not `dict_in_out`, reshape the label to be `label.view(-1,
            input.shape[-1])`.
        output_name:
            if `dict_in_out`, this is the key for the loss value if the output
            is a dict. If the output is not a dict, then we assume the first
            element of the output is the loss.
        lossfn:
            loss function to use if not `dict_in_out`. Default is `xent` for
            cross entropy loss. Other choices are ['mse', 'nll']
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict: 
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm.
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).
    '''
    df = []
    if fix_data:
        batch = next(iter(dataloader))
        def fixed_next():
            while True:
                yield batch
        dataloader = iter(fixed_next())
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=nseeds * len(models))

    for i in range(nseeds):
        torch.manual_seed(i)
        for width, model in models.items():
            model = model()
            model.train()
            if cuda:
                model = model.cuda()
            optimizer = optcls(model)
            for batch_idx in range(nsteps):
                remove_hooks = []
                # add hooks
                for name, module in model.named_modules():
                    if filter_module_by_name and not filter_module_by_name(name):
                        continue
                    remove_hooks.append(module.register_forward_hook(
                       mup_coord_check._record_coords(df, width, name, batch_idx + 1,
                            output_fdict=output_fdict,
                            input_fdict=input_fdict,
                            param_fdict=param_fdict)))

                model.train_batch(data_iter=iter(dataloader))

                # remove hooks
                for handle in remove_hooks:
                    handle.remove()
            if show_progress:
                pbar.update(1)

            # Free the memory occupied by the PipelineEngine
            import gc
            del model
            gc.collect()
    if show_progress:
        pbar.close()
    return pd.DataFrame(df)
