from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


def get_dataloader(
    tokenizer, batch_size, max_length, dataset_name, split="train"
):
    if split != "train": 
        raise Exception("Currently only supports Train split")
    
    dataset = load_dataset(dataset_name)
    
    def tokenize_fn(examples: Dict[str, List]):
        
        N = len(examples['problem'])
        
        input_texts = []
        
        for i in range(N):
            chat = [
                {"role": "system", "content": examples['system prompt'][i]},
                {"role": "user", "content": examples['problem'][i]},
                {"role": "assistant", "content": examples['solution'][i]}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            input_texts.append(formatted_prompt)
            
        inputs = tokenizer(input_texts, truncation=True, max_length=max_length)
        
        return inputs
    
    
    
    tokenized_train = dataset["train"].map(tokenize_fn, batched=True, batch_size=100, 
                                        remove_columns=dataset["train"].column_names)
    
    # This will handle:
    #     1. Dynamic padding
    #     2. Creates "label" tensor (copy from input_ids and set -100 for pad positions)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors='pt'
    )

    train_dataloader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    return train_dataloader