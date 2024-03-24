# ORPO: Monolithic Preference Optimization without Reference Model

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 🦾 Training

## ⚡️ FSDP

```bash
ACCELERATE_LOG_LEVEL=INFO accelerate launch --config_file fsdp.yaml train.py
```

## ⚡️DeepSpeed Zero 3

```bash
ACCELERATE_LOG_LEVEL=INFO accelerate launch --config_file deepspeed-zero3.yaml train.py
```

## 🧪 Evaluation

```bash
lm_eval --model hf \
    --model_args pretrained=alvarobartt/mistral-orpo-mix,dtype=bfloat16,attn_implementation=flash_attention_2 \
    --tasks ifeval \
    --device cuda:0 \
    --batch_size 16 \
    --output_path output/mistral-orpo-mix \
    --wandb_args project=Mistral-7B-v0.1-ORPO \
    --log_samples
```
