# ORPO: Monolithic Preference Optimization without Reference Model

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 🦾 Training

## ⚡️DeepSpeed Zero 3

```bash
ACCELERATE_LOG_LEVEL=INFO accelerate launch --config_file deepspeed-zero3.yaml train.py
```

## ⚡️ FSDP

Due to the model embedding resizing due to the added ChatML tokens, the optimizer saving is
failing on each save, while the model saving is fine.

> [!ERROR]
> RuntimeError: shape '[32002, 4096]' is invalid for input of size 64005

```bash
ACCELERATE_LOG_LEVEL=INFO accelerate launch --config_file fsdp.yaml train.py
```


## 🧪 Evaluation

### IF-Eval

Install it as:

```bash
pip install "lm_eval[ifeval,wandb]" --quiet
```

Then run it as:

```bash
lm_eval --model hf \
    --model_args pretrained=alvarobartt/mistral-orpo-mix,dtype=bfloat16,attn_implementation=flash_attention_2 \
    --tasks ifeval \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output/mistral-orpo-mix \
    --wandb_args project=Mistral-7B-v0.1-ORPO \
    --log_samples
```

### AlpacaEval 2.0

> [!NOTE]
> The files under `benchmarks/alpaca_eval/model_configs` need to be copied to `alpaca_eval/model_configs` to run the script below,
> or just install `alpaca_eval` from the fork as `pip install git+https://github.com/alvarobartt/alpaca_eval.git@main --quiet`.

Then run it as:

```bash
alpaca_eval evaluate_from_model --model_configs "mistral-orpo-mix" --annotators_config "alpaca_eval_gpt4"
alpaca_eval evaluate_from_model --model_configs "mistral-orpo-mix" --annotators_config "weighted_alpaca_eval_gpt4_turbo"
```

Or if already generated:

```bash
alpaca_eval evaluate --model_outputs "results/mistral-orpo-mix/model_outputs.json" --annotators_config "alpaca_eval_gpt4" 
alpaca_eval evaluate --model_outputs "results/mistral-orpo-mix/model_outputs.json" --annotators_config "weighted_alpaca_eval_gpt4_turbo"
```

### MT-Bench

Install from source as:

```bash
pip install git+https://github.com/alvarobartt/FastChat.git@main --quiet
```

Then run it as described at [FastChat - LLM Judge Evaluation](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#mt-bench).