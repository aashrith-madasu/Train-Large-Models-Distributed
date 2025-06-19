from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModelForCausalLM
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from dataset import get_dataloader
from utils import save_fsdp_peft_model

EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
MAX_LENGTH = 8000
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET_NAME = "FractalAIResearch/Fathom-V0.6-Iterative-Curriculum-Learning"

wandb.init(
    project="qwen-7b",
    name="run-1",
    config={
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "model_name": MODEL_NAME
    },
)

# Initialize accelerator
accelerator = Accelerator()
print("Hello from device ", accelerator.device)

# Load tokenizer and dataset

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.chat_template = open("new_chat_template.txt").read()


train_loader = get_dataloader(tokenizer=tokenizer, 
                              batch_size=BATCH_SIZE,
                              max_length=MAX_LENGTH,
                              dataset_name=DATASET_NAME,
                              split="train")
num_batches = len(train_loader)
print(f"Loaded {num_batches} batches")


# Load model (4-bit quant + LoRA)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",  # or "fp4"
# )
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     # bnb_4bit_quant_storage=torch.bfloat16,
# )
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # quantization_config=bnb_config,
    device_map=None,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2"
)
# model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

wandb.watch(model, log="all", log_freq=num_batches)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Prepare everything with accelerator
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)


# Training Loop
for epoch in range(EPOCHS):
    
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        
        wandb.log({"train/loss": loss.item()})
        
    # >>> SAVE model >>>
    save_fsdp_peft_model(model)

    avg_loss = total_loss / len(train_loader)
    accelerator.print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
