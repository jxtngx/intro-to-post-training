# LoRA Finetune Single Device

This example follows the official [end-to-end workflow](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html) example created by the [torchtune](https://github.com/pytorch/torchtune) team, and updates the example to account for the need to supply a Hugging Face token to download the model, and to enable logging to Weights and Biases by first copying and then editing a default config. 

Use the badge below to run the recipe in a pre-configured environment hosted on Lightning AI.

<a target="_blank" href="https://lightning.ai/jxtngx/studios/lora-finetune-single-device">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a><br/><br/>


> [!TIP]
> need more info on [LoRA](https://arxiv.org/abs/2106.09685) before getting started? Here's a deep dive on [LoRA with Llama 2](https://pytorch.org/torchtune/main/tutorials/lora_finetune.html) by the torchtune team, and a helpful [Hugging Chat Assistant](https://hf.co/chat/assistant/67c713539a0a2709d56261c9)

## Lora finetuning workflow

### Setup a virtual environment

If running locally or in a compute cloud other than Lightning AI, we will need to create a new Python environment with the following steps in terminal:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Test that `torchtune` has been installed correctly with the following command in terminal:

```bash
tune --help
```

> [!IMPORTANT]
> macOS and Linux/CUDA environments each have different effects on the dependency installations, this example is optimized for Linux/CUDA

### Prepare the dataset

The example uses the default [`alpaca_cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset, and does not require any additional preparation on our part. 

### Prepare a tuning config

A [config](lora_finetune_single_device.yaml) is supplied, there is no need to use the torchtune CLI to copy a default config.

### Download the model

> [!IMPORTANT]
> Make certain you have been granted access to the [model on Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

> [!TIP]
> find or create your HF API key with [these instructions](https://huggingface.co/docs/hub/en/security-tokens)

To download the model, use the `torchtune` CLI in the terminal as follows:

```bash
tune download meta-llama/Llama-3.2-3B-Instruct \
--ignore-patterns "original/consolidated.00.pth" \
--hf-token <YOUR HF TOKEN> \
--output-dir ckpts/Llama-3.2-3B-Instruct
```

### Setup Weights and Biases 

Next, set a W&B key or login in via the terminal. A [script](wb.py) is provided to login with Python.

> [!TIP]
> find your W&B API key with [these instructions](https://docs.wandb.ai/support/find_api_key/)

> [!NOTE]
> if using the script, ensure you have a local .env file that uses `WANDB_API_KEY` to set your W&B login

To use the script, do the following in terminal:

```bash
python wb.py
```

> [!IMPORTANT]
> W&B will create a .netrc file if one does not exist. Do not share this file!

### Tune the model

Finally, start the tuning run by using the following command in terminal:

```bash
tune run lora_finetune_single_device --config lora_finetune_single_device.yaml
```

### Monitor the run

On running the above command, W&B will log the run link to terminal. We can follow that link to observe our tuning run. Here is an [example](https://wandb.ai/justingoheen/lora-finetune-single-device).

