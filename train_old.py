from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm import tqdm


# Initialize accelerator
accelerator = Accelerator()

print("Hello from device ", accelerator.device)

# Load tokenizer and dataset
model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


dataset = load_dataset("Abirate/english_quotes")  # Replace with your dataset
max_length = 256

# Tokenization function
def tokenize_fn(example):
    inputs = tokenizer(example["quote"], padding="max_length", truncation=True, max_length=max_length)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# Tokenize the dataset
tokenized = dataset.map(tokenize_fn, batched=True, batch_size=100, remove_columns=dataset["train"].column_names)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# DataLoader
batch_size = 1
train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True)
print(f"Loaded {len(train_loader)} batches")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
)

# # Load model (4-bit quantization + LoRA)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     load_in_4bit=True,
#     torch_dtype=torch.bfloat16,  # Or torch.float16
#     trust_remote_code=True
# )
# model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-4)

# Prepare everything with accelerator
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

# Training Loop
model.train()
num_epochs = 1

for epoch in range(num_epochs):
    total_loss = 0
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        
        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    accelerator.print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
