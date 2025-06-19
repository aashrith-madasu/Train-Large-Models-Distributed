

def save_fsdp_peft_model(model, accelerator):
    # unwrap accelerator FSDP/Distributed wrappers
    unwrapped_model = accelerator.unwrap_model(model)
    # if using PEFT, unwrap PEFT wrapper to get the base model
    base_model = unwrapped_model.base_model
    base_model.save_pretrained(
        "checkpoints",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )