import json
import os
import time
from tqdm import tqdm
import argparse
# import fire
import torch
import torch.distributed as dist
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
)

from configs import train_config, fsdp_config
from utils import fsdp_auto_wrap_policy
from utils.config_utils import (
    update_config,
    generate_peft_config,
)
from utils.train_utils import (
    train,
    evaluation,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
)
from data import SquadCausalDataset, SquadGenerationDataset, CustomGenerationDataset

USER = os.getenv("USER")
DEFAULT_MODEL = f"/home/gridsan/{USER}/languagemodels/models/Llama-2-7b-hf-causal"


def main(args, **kwargs):
    # Update the configuration for the training and sharding process
    kwargs["model_name"] = args.model_path
    kwargs["quantization"] = args.quantization
    kwargs["val_batch_size"] = args.batch_size_validation
    kwargs["num_workers_dataloader"] = args.num_workers_dataloader
    update_config((train_config, fsdp_config), **kwargs)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    # model = LlamaForCausalLM.from_pretrained(
    #         model_name,
    #         return_dict=True,
    #         load_in_8bit=quantization,
    #         device_map="auto",
    #         low_cpu_mem_usage=True,
    # )

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
    if not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    # tokenizer = LlamaTokenizer.from_pretrained(f"/home/gridsan/{USER}/languagemodels/models/Llama-2-7b-hf-causal")
    tokenizer = LlamaTokenizer.from_pretrained(f"/home/gridsan/{USER}/languagemodels/models/Llama-2-7b-hf-causal", model_max_length=512, padding_side="left")
    # tokenizer.add_special_tokens({"pad_token": "<PAD>",})  # DOING THIS CAUSED A CUDA INDEX ERROR
    tokenizer.pad_token = "<PAD>"
    
    # squad = datasets.load_from_disk(f"/home/gridsan/{USER}/languagemodels/datasets/squad")
    # dataset_val = SquadCausalDataset(squad["validation"].select(range(64)), tokenizer, add_answer=False)
    # dataset_val = SquadGenerationDataset(squad["validation"].select(range(64)), tokenizer)
    
    with open(args.form, "r") as f:
        form = f.read()
    with open(args.data_path, "r") as f:
        dataset = json.load(f)
    dataset_gen = CustomGenerationDataset(dataset, form, tokenizer, max_length=args.max_length)
    
    dataloader = torch.utils.data.DataLoader(
        dataset_gen,
        batch_size=train_config.val_batch_size,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=None,
        drop_last=False,
        collate_fn=default_data_collator,
    )

    start = time.perf_counter()
    kwargs = {}
    outputs = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, colour="blue", desc="Generation")):
            for key in batch.keys():
                batch[key] = batch[key].to('cuda:0')
            output = model.generate(
                **batch,
                max_new_tokens=32,
                do_sample=True,
                top_p=.5,
                temperature=.1,
                min_length=None,
                use_cache=True,
                top_k=50,
                repetition_penalty=1.0,
                length_penalty=1,
                **kwargs
            )
            outputs.extend(output)
    e2e_inference_time = (time.perf_counter()-start)*1000
    print(f"the inference time is {e2e_inference_time} ms")
    generations = tokenizer.batch_decode(outputs, skip_special_tokens=args.skip_special_tokens)
    with open(args.output, "w") as f:
        for g in generations:
            f.write(g)
            f.write("\n\n###################################\n\n")
        
if __name__ == "__main__":
    # fire.Fire(main)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--form", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument("--batch_size_validation", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers_dataloader", type=int, default=10)
    parser.add_argument("--skip_special_tokens", action="store_true")
    args = parser.parse_args()
    main(args)
