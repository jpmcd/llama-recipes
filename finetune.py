import json
import os
import time
import pickle
from tqdm import tqdm
import datasets
import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from pkg_resources import packaging
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
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
        if os.getenv('OMPI_COMM_WORLD_SIZE') is not None:
            rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
            world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
        else:
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)
    else:
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
        )
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)
    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)
    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(train_config.num_freeze_layers)
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")
    
    tokenizer = LlamaTokenizer.from_pretrained(f"{HOME}/languagemodels/models/Llama-2-7b-hf-causal")  # , model_max_length=512)
    tokenizer.pad_token = "<PAD>"  # without adding with tok.add_special_tokens(), pad_id = unk_id = 0
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    
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
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=torch.distributed.get_rank(),
            num_replicas=torch.distributed.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=torch.distributed.get_rank(),
                num_replicas=torch.distributed.get_world_size(),
            )
    # Create DataLoaders for the training and validation dataset
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
    eval_dataloader = []
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
