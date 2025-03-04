# LoRA Finetune Single Device

This example follows the official [end-to-end workflow](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html) example created by the [torchtune](https://github.com/pytorch/torchtune) team, and updates the example with the following commands to account for the need to supply a Hugging Face token to download the model, and to enable logging to Weights and Biases.

## Download the model

```bash
tune download meta-llama/Llama-3.2-3B-Instruct \
--ignore-patterns "original/consolidated.00.pth" \
--hf-token <YOUR HF TOKEN> \
--output-dir ckpts/Llama-3.2-3B-Instruct
```

## Tune the model and log the run

1. copy the config so that we can update the experiment logger

```bash
tune cp llama3_2/3B_lora_single_device lora_finetune_single_device.yaml
```

2. open the new yaml and ensure the metric logger setting reflects the following

```yaml
metric_logger:
  _component_: torchtune.training.metric_logging.WandbBLogger
  # the W&B project to log to
  project: llama-3.2-3B-lora-finetune-single-device
```

3. set a W&B key or login in via the terminal. a script is provided to login with python. to use the script do the following in terminal:

```bash
python wb.py
```

> [!NOTE]
> ensure you have a local .env file that uses `WANDB_API_KEY` to set your W&B login

> [!NOTE]
> W&B will create a .netrc file if one does not exist. do not share this file

4. tune the model

```bash
tune run lora_finetune_single_device --config lora_finetune_single_device.yaml
```

> [!NOTE]
> the example uses the default dataset [`alpaca_cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned)
