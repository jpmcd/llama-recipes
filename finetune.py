import json
import os
import time
import pickle
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
from data import CustomCausalDataset
from data import SquadCausalDataset


HOME = os.environ["HOME"]


def main(**kwargs):
    # kwargs["model_name"] = f"{HOME}/languagemodels/models/Llama-2-7b-hf-causal"
    
    kwargs["val_batch_size"] = kwargs["val_batch_size"] if "val_batch_size" in kwargs else kwargs.get("batch_size_validation", 16)
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
    tokenizer.pad_token = "<PAD>"  # without adding with tok.add_special_tokens(), pad_id = unk_id = 0
    
    # Set dataset and sampling
    with open(kwargs["form"], "r") as f:
        form = f.read()
    if "context" in kwargs:
        with open(kwargs["context"], "r") as f:
            context_form = f.read()
    else:
        context_form = ""
    ext = os.path.splitext(kwargs["data_path"])[1]
    if ext == ".json":
        with open(kwargs["data_path"], "r") as f:
            data = json.load(f)
    elif ext == ".pkl":
        with open(kwargs["data_path"], "rb") as f:
            data = pickle.load(f)
    max_training_data = kwargs.get("max_training_data", len(data["train"]))
    data["train"] = data["train"][:max_training_data]
    dataset_train = CustomCausalDataset(data["train"], form, context_form, tokenizer, max_length=kwargs["max_length"])
    dataset_val = []
    # squad = datasets.load_from_disk(f"{HOME}/languagemodels/datasets/squad")
    # max_training_data = kwargs.get("max_training_data", len(squad["train"]))
    # max_validation_data = kwargs.get("max_validation_data", len(squad["validation"]))
    # dataset_train = SquadCausalDataset(squad["train"].select(range(max_training_data)), tokenizer)
    # dataset_val = SquadCausalDataset(squad["validation"].select(range(max_validation_data)), tokenizer)
    train_sampler = None
    val_sampler = None
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        shuffle=train_config.shuffle and train_sampler is None,
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
    else:
        eval_dataloader = []
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
    if train_config.run_validation:
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
