import logging
import sys
from datetime import timedelta

import torch
import wandb

from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import DatasetDict, load_dataset
from datasets.utils import logging as datasets_logging
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.utils import logging as transformers_logging
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    datasets_logging.set_verbosity(logging.INFO)
    transformers_logging.set_verbosity(logging.INFO)
    transformers_logging.enable_default_handler()
    transformers_logging.enable_explicit_format()

    # Set random seed for reproducibility
    set_seed(42)

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=12 * 1800))]
    )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.use_cache = False

    model, tokenizer = setup_chat_format(model, tokenizer)

    def chatml_format(example: dict) -> dict:
        return {
            "prompt": tokenizer.apply_chat_template(
                example["prompt"], tokenize=False, add_generation_prompt=True
            ),
            "chosen": f"{example['chosen'][0]['content']}<|im_end|>\n",
            "rejected": f"{example['rejected'][0]['content']}<|im_end|>\n",
        }

    dataset = load_dataset("alvarobartt/dpo-mix-7k-simplified")
    logger.info("*** Dataset loaded ***")

    if not isinstance(dataset, DatasetDict):
        logger.warn("No test split found in the dataset, creating one with 0.1 ratio")
        dataset = dataset.train_test_split(test_size=0.1, seed=42)  # type: ignore

    dataset = dataset.map(
        chatml_format,
        num_proc=4,
        remove_columns=list(
            set(dataset["train"].column_names) - {"prompt", "chosen", "rejected"}  # type: ignore
        ),
    )

    dataset = dataset.shuffle(seed=42)

    # Initialize `wandb` run
    if accelerator.is_main_process:
        wandb.init(
            entity="alvarobartt",
            project="Mistral-7B-v0.1-ORPO",
            name="full-beta-0.1-lr-5e-6",
        )

    training_args = ORPOConfig(
        # ORPOTrainer
        beta=0.1,  # official: alpha=0.05
        max_length=2048,  # former: 1024,
        max_prompt_length=1792,  # former: 512,
        # Trainer (train)
        output_dir="./mistral-7b-v0.1-orpo",
        bf16=True,
        do_train=True,
        seed=42,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # former: 2
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=5.0e-6,  # former: 5.0e-7
        lr_scheduler_type="cosine",  # official: "inverse_sqrt"
        num_train_epochs=3,
        optim="adamw_bnb_8bit",  # former: "adamw_torch"
        # Trainer (warmup)
        warmup_ratio=0.1,
        warmup_steps=100,
        # Trainer (logging)
        logging_steps=10,
        report_to=["wandb", "tensorboard"],
        # Trainer (eval)
        do_eval=True,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        # Trainer (save)
        hub_model_id="alvarobartt/Mistral-7B-v0.1-ORPO-full-beta-0.1",
        hub_private_repo=True,
        push_to_hub=False,
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
    )

    trainer = ORPOTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
    )
    logger.info("*** ORPOTrainer initialized ***")

    result = trainer.train()
    logger.info("*** Training completed ***")

    # metrics = result.metrics
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("*** Training metrics logged/saved ***")

    if accelerator.is_main_process:
        wandb.finish()

        trainer.save_model(training_args.output_dir)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True  # type: ignore

        logger.info("*** Saving model locally ***")
        trainer.model.config.save_pretrained(
            f"{training_args.output_dir}/final_checkpoint"
        )  # type: ignore
        tokenizer.save_pretrained(f"{training_args.output_dir}/final_checkpoint")
        logger.info("*** Model successfully saved locally ***")

        logger.info("*** Pushing model to Hub ***")
        trainer.push_to_hub()
        logger.info("*** Model successfully pushed to Hub ***")

    logger.info("*** Run complete! ***")
