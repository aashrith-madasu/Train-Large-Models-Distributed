from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm import tqdm

from dataset import get_dataloader


# Initialize accelerator
accelerator = Accelerator()

print("Hello from device ", accelerator.device)

# Load tokenizer and dataset
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.chat_template = open("new_chat_template.txt").read()


train_loader = get_dataloader(tokenizer=tokenizer, split="train")
print(f"Loaded {len(train_loader)} batches")


# Load model (4-bit quant + LoRA)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",  # or "fp4"
# )
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_storage=torch.bfloat16,
)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=None,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2"
)
model = prepare_model_for_kbit_training(model)
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
