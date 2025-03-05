# LoRA Finetune Single Device

This example follows the official [end-to-end workflow](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html) example created by the [torchtune](https://github.com/pytorch/torchtune) team, and updates the example to account for the need to supply a Hugging Face token to download the model, and to enable logging to Weights and Biases by first copying and then editing a default config. 

Use the badge below to run the recipe in a pre-configured environment hosted on Lightning AI.

<a target="_blank" href="https://lightning.ai/jxtngx/studios/lora-finetune-single-device">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a><br/><br/>


> [!TIP]
> need more info on [LoRA](https://arxiv.org/abs/2106.09685) before getting started? Here's a deep dive on [LoRA with Llama 2](https://pytorch.org/torchtune/main/tutorials/lora_finetune.html) by the torchtune team, and a helpful [Hugging Chat Assistant](https://hf.co/chat/assistant/67c713539a0a2709d56261c9) powered by Llama 3.3 70B Instruct

## LoRA finetuning workflow

> [!IMPORTANT]
> this example is best ran on CUDA compatible devices, including consumer hardware in the RTX family of devices

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
tune run --recipe lora_finetune_single_device.py --config lora_finetune_single_device.yaml
```

> [!NOTE]
> the [provided Recipe](lora_finetune_single_device.py) is copied from [`torchtune/recipes/lora_finetune_single_device`](https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_single_device.py)

### Monitor the run

On running the above command, W&B will log the run link to console (terminal). We can follow that link to observe our tuning run. Here is an [example](https://wandb.ai/justingoheen/lora-finetune-single-device) Weights & Biases project tracking demo runs.

## Under the hood

The pre-built Recipe that is being ran is [`torchtune/recipes/lora_finetune_single_device`](https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_single_device.py), and the pre-configured config used to define our new config is [`torchtune/recipes/configs/llama3_2/3B_lora_single_device.yaml`](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_2/3B_lora_single_device.yaml). 

The pre-built Recipes and configs are good starting points for the majority of projects, and provide excellent templates for creating custom recipes for post-training tasks that might require greater control over the training loop and data pipelines. Configs ought to follow a convention in order to correctly structure the defined settings for the Recipe. Recipes in `torchtune` are minimal and extensible by design.

Let's take a look at what makes a config and a Recipe, according to `torchtune`.

### Configs

> [!TIP]
> see the [torchtune docs](https://pytorch.org/torchtune/stable/deep_dives/configs.html) for more information on configs

Configs are yaml files that define the settings for a `torchtune` Recipe, the sections (keys) of a config are as follows:

- Model Arguments
- Tokenizer
- Checkpointer
- Dataset and DataLoader
- Optimizer and Scheduler
- Training
- Logging
- Environment 
- Activation Memory
- Profiler

Since yaml is a way to define key-value pairs, Python will treat the config as a dictionary, making the values accessible via the section keys, and able to be passed to the Recipe as arguments.

If we create a custom config, we can use the `torchtune` CLI to validate the config with the following command:

```bash
tune validate <CONFIG FILE PATH>
```

Using our config, this would look like:

```bash
tune validate lora_finetune_single_device.yaml
```

> [!NOTE]
> `torchtune` uses [OmegaConf](https://omegaconf.readthedocs.io) to parse the config files

> [!TIP]
> the config files are excellent entry points for exploring topics that are important to post-training techniques

### Recipes

> [!TIP]
> see the [torchtune docs](https://pytorch.org/torchtune/stable/index.html) for more information on Recipes

Pre-built Recipes in `torchtune` are Python interfaces (Classes) that prepare the model, data, and training loop for the tuning run, and manage the training loop – including saving the checkpoint when tuning is complete, and handling post-tuning cleanup. Recipes are [`FTRecipeInterface`](https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07041c12ffde830332d7/torchtune/recipe_interfaces.py#L10) classes, with several base methods shown below:


```python
class FTRecipeInterface(Protocol):
    def load_checkpoint(self, **kwargs) -> None:
        """
        Responsible for loading ALL of the state for the recipe from the
        checkpoint file, including state for the model, optimizer, dataloader and training
        parameters such as the epoch and seed.
        """
        ...

    def setup(self, **kwargs) -> None:
        """
        Responsible for setting up all of the components necessary for training. This includes
        model, optimizer, loss function and dataloader.
        """
        ...

    def train(self, **kwargs) -> None:
        """
        All of the training logic, including the core loop, loss computation, gradient
        accumulation, and backward.
        """
        ...

    def save_checkpoint(self, **kwargs) -> None:
        """
        Responsible for saving ALL of the state for the recipe,
        including state for the model, optimizer, dataloader and training
        parameters such as the epoch and seed.
        """
        ...

    def cleanup(self, **kwargs) -> None:
        """
        Any cleaning up needed for the recipe.
        """
        ...

```

If we were to define a custom Recipe, we would subclass `FTRecipeInterface` and implement the methods above, extending the custom recipe as needed. For instance – there is a need to add intermediate methods for setup. The `LoRAFinetuneRecipeSingleDevice` recipe adds the following methods to support the `setup` step:

```python
class LoRAFinetuneRecipeSingleDevice(FTRecipeInterface):
    def setup(self):
        ...
    def _setup_profiler(self):
        ...
    def _setup_model(self):
        ...
    def _setup_optimizer(self):
        ...
    def _setup_lr_scheduler(self):
        ...
    def _setup_data(self):
        ...
```

Creating new Recipes with `torchtune` has a wide degree of freedom for the engineer, and the pre-built Recipes are excellent starting points for creating custom Recipes, though – the pre-built recipes are examples, and not doctrine. We can experiment!


## More examples

See the [torchtune docs](https://pytorch.org/torchtune/stable/index.html) for additional examples, and join the [official Discord](https://discord.gg/tyRWHtHgV7).

