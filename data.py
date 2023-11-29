import torch
from torch.utils.data import Dataset
# from sentencepiece import SentencePieceProcessor
import copy


PROMPT_DICT = {
    "prompt_message": (
        "Message: {text}"
    ),
    "prompt_question": (
        "Question: {input}"
    ),
    "prompt_answer": (
        "Answer: {input}"
    ),
    "prompt_squad": (
        "{context}\nQuestion:\n{question}\nAnswer:\n"
    ),
}
IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


class SuperCloudDemoDataset(Dataset):
    def __init__(self, data, tokenizer, split="train", max_length=256):
        self.tokenizer = tokenizer
        if split == "train":
            self.data = data
        else:
            self.data = data[:500]
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        prompt = PROMPT_DICT["prompt_message"].format_map(data)
        example = prompt
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_length - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_length]
        labels = copy.deepcopy(example)
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


class SuperCloudThreadsDataset(Dataset):
    def __init__(self, data, tokenizer, split="train", max_length=128):
        self.tokenizer = tokenizer
        if split == "train":
            self.data = data
        else:
            self.data = data[:500]
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        if "input" in data:
            prompt = PROMPT_DICT["prompt_question"].format_map(data)
        else:
            prompt = PROMPT_DICT["prompt_answer"].format_map(data)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_length - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_length]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


class CustomGenerationDataset(Dataset):
    def __init__(self, data, form, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.form = form
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        example = self.data[index]
        example = self.form.format_map(example)
        example = self.tokenizer.encode(example, padding="max_length", max_length=self.max_length)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        example_mask = example.ge(1)
        example_mask = example_mask.float()
        return {
            "input_ids": example,
            "attention_mask": example_mask,
        }


class CustomCausalDataset(Dataset):
    def __init__(self, data, form, context_form, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.form = form
        self.context_form = context_form

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        context = self.context_form.format_map(example)
        example = self.form.format_map(example)
        context = self.tokenizer.encode(context, add_special_tokens=False)
        example = self.tokenizer.encode(example, add_special_tokens=False, padding="max_length", max_length=self.max_length)  # , return_tensors="pt")
        # if self.add_answer:
        #     example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        # padding = self.max_length - example.shape[0]
        # if padding > 0:
        #     example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        # elif padding < 0:
        #     example = example[:self.max_length]
        example = example[:self.max_length]
        labels = copy.deepcopy(example)
        labels[:len(context)] = 0
        example_mask = example.ge(1)
        label_mask = labels.ge(1)
        labels[~label_mask] = IGNORE_INDEX
        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


class SquadGenerationDataset(CustomGenerationDataset):
    def __init__(self, data, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.form = PROMPT_DICT["prompt_squad"]


class SquadCausalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, add_answer=True):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.add_answer = add_answer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        prompt = PROMPT_DICT["prompt_squad"].format_map(data)
        answer = data["answers"]["text"][0] if self.add_answer else ""
        example = prompt + answer
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        if self.add_answer:
            example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_length - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_length]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }
