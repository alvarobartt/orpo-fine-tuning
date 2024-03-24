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
