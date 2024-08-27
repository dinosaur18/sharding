import torch
from transformers import PhiConfig, PhiForCausalLM, AutoTokenizer
from transformers import default_data_collator
from llm_shearing.models.mask import Mask
from llm_shearing.models.sheared_phi import ShearedPhiForCausalLM
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from itertools import chain

# ACCELERATE
accelerator = Accelerator()
accelerator.wait_for_everyone()

# model
MODEL_NAME = "/workspace/NLP_CORE/llm_application/LLM_Compression/base_models/phi-2"
# model = PhiForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

config = PhiConfig(
    vocab_size=51200,
    hidden_size=16,
    intermediate_size=128,
    num_hidden_layers=16,
    num_attention_heads=8,
    max_position_embeddings=1000
)

model = PhiForCausalLM(config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
block_size = 128
opt = torch.optim.Adam(model.parameters())

def tokenize_function(examples):
    output = tokenizer(examples["text"])
    return output

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    """
    We drop the small remainder, and if the total_length < block_size we exclude 
    this batch and return an empty dict. We could add padding if the model 
    supported it instead of this drop, you can customize this part to your needs.
    """
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

with accelerator.main_process_first():
    raw_dataset = load_dataset(
        "json", 
        data_files="/workspace/NLP_CORE/llm_application/LLM_Compression/data_raw/train_part_1/sample/5k.jsonl", 
        split="train"
    )
    column_names = raw_dataset.column_names
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )


    train_loader = DataLoader(
        lm_dataset, shuffle=True, 
        collate_fn=default_data_collator, 
        batch_size=16
    )


# class DS(Dataset):
#     def __init__(self):
#         super().__init__()
#         self.data = torch.randint(0, 10, [4, 10])
       
#     def __getitem__(self, idx):
#         return self.data[idx]
    
#     def __len__(self):
#         return len(self.data)


# def count_params(model):
#     n_params = sum(p.numel() for p in model.parameters())
        
#     print(f'Parameters: {n_params:,}')
#     return n_params





# train_loader = DataLoader(DS(), 1)

print("INIT")
model, opt, train_loader = accelerator.prepare(model, opt, train_loader)
opt.zero_grad()

# input_ids = next(iter(train_loader))
# loss = model(input_ids.long(), labels=input_ids).loss
model.train()
for step, batch in enumerate(train_loader):
    print(f"step: {step}")
    with accelerator.accumulate(model):
        inputs = batch
        outputs = model(**inputs)
        loss = outputs.loss
        accelerator.backward(loss)
        opt.step()
        opt.zero_grad()
        
    if accelerator.sync_gradients:
        import pdb; pdb.set_trace()

import pdb; pdb.set_trace()
accelerator.backward(loss)
opt.step()


print(model.model.layers[0].mlp.fc1.weight.grad)
# print(model.model.z_hidden.z_latent.grad)
