import json
import os
import time
from tqdm import tqdm
import datasets
import argparse
import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from pkg_resources import packaging
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import policies
from configs import train_config, fsdp_config
from policies import AnyPrecisionAdamW
from utils import fsdp_auto_wrap_policy
from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from utils.dataset_utils import get_preprocessed_dataset
from utils.train_utils import (
    train,
    evaluation,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from data import PROMPT_DICT, SuperCloudDemoDataset, SuperCloudThreadsDataset
from data import SquadCausalDataset


HOME = os.environ["HOME"]


def main(**kwargs):
    # kwargs["run_validation"] = True
    # kwargs["model_name"] = f"{HOME}/languagemodels/models/Llama-2-7b-hf-causal"
    # kwargs["quantization"] = True
    
    update_config((train_config, fsdp_config), **kwargs)
    with open(os.path.join(train_config.output_dir, "args.json"), "w") as f:
        json.dump(kwargs, f)
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    
    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        load_in_8bit=True if train_config.quantization else None,
        device_map="auto" if train_config.quantization else None,
    )
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)
    if train_config.quantization:
            model = prepare_model_for_int8_training(model)
    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    if train_config.enable_fsdp:
        raise 
    elif not train_config.quantization:
        model.to("cuda")
    
    tokenizer = LlamaTokenizer.from_pretrained(f"{HOME}/languagemodels/models/Llama-2-7b-hf-causal")  # , model_max_length=512)
    tokenizer.pad_token = "<PAD>"
    
    # Set dataset and sampling
    squad = datasets.load_from_disk(f"{HOME}/languagemodels/datasets/squad")
    max_training_data = kwargs.get("max_training_data", len(squad["train"]))
    max_validation_data = kwargs.get("max_validation_data", len(squad["validation"]))
    # if max_training_data is not None:
    dataset_train = SquadCausalDataset(squad["train"].select(range(max_training_data)), tokenizer)
    # if max_validation_data is not None:
    dataset_val = SquadCausalDataset(squad["validation"].select(range(max_validation_data)), tokenizer)
    train_sampler = None
    val_sampler = None
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=not kwargs.get("keep_last"),
        collate_fn=default_data_collator,
    )
    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=not kwargs.get("keep_last"),
            collate_fn=default_data_collator,
        )
    # Set optimizer, scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=0.0,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    if max_training_data > 0:
        results = train(
            model,
            train_dataloader,
            eval_dataloader,
            tokenizer,
            optimizer,
            scheduler,
            train_config.gradient_accumulation_steps,
            train_config,
            fsdp_config if train_config.enable_fsdp else None,
            local_rank if train_config.enable_fsdp else None,
            rank if train_config.enable_fsdp else None,
        )
        if not train_config.enable_fsdp or rank==0:
            [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
    if True:
        print("Running evaluation...")
        evaluation(
            model,
            train_config,
            eval_dataloader,
            local_rank if train_config.enable_fsdp else None,
            tokenizer,
        )
        print("Finished evaluation...")
    
    
if __name__ == "__main__":
    fire.Fire(main)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    # parser.add_argument("--form", type=str)
    # parser.add_argument("--data_path", type=str)
    # parser.add_argument("--output", type=str)
    # parser.add_argument("--quantization", action="store_true")
    # parser.add_argument("--batch_size_validation", type=int, default=16)
    # parser.add_argument("--max_length", type=int, default=128)
    # parser.add_argument("--num_workers_dataloader", type=int, default=10)
    # parser.add_argument("--skip_special_tokens", action="store_true")
    # args = parser.parse_args()
    # main(args)