# TorchTune Cookbook

[TorchTune](https://pytorch.org/torchtune/stable/index.html) is a PyTorch native post-training library. It provides a set of pre-built recipes and configs for common post-training tasks, as well as a flexible framework for creating custom recipes.

## Under the hood

The pre-built recipes and configs are good starting points for the majority of projects, and provide excellent templates for creating custom recipes for post-training tasks that might require greater control over the training loop and data pipelines. Configs follow a convention in order to correctly structure the defined settings for the Recipe. Recipes are minimal and extensible by design.

Let's take a look at what makes a config and a Recipe, according to torchtune.

### Configs

> [!TIP]
> see the [torchtune docs](https://pytorch.org/torchtune/stable/deep_dives/configs.html) for more information on configs

Configs are yaml files that define the settings for a torchtune Recipe, the sections (keys) of a config are as follows:

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

If we create a custom config, we can use the torchtune CLI to validate the config with the following command:

```bash
tune validate <CONFIG FILE PATH>
```

> [!NOTE]
> torchtune uses [OmegaConf](https://omegaconf.readthedocs.io) to parse the config files

### Recipes

> [!TIP]
> see the [torchtune docs](https://pytorch.org/torchtune/stable/index.html) for more information on Recipes

Pre-built Recipes are Python interfaces (Classes) that can be used to prepare the data, model, and checkpointer, and manage the training loop, including post-experiment tasks like saving the final checkpoint and shutting down the experiment manager. Recipes are [`FTRecipeInterface`](https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07041c12ffde830332d7/torchtune/recipe_interfaces.py#L10) classes, with several base methods shown below:


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

> [!NOTE]
> [Protocols](https://peps.python.org/pep-0544/) in Python are an instance of abc.ABC that allow for structural subtyping i.e. easier duck typing.

When creating a custom Recipe, we subclass `FTRecipeInterface` and implement the methods above, extending the custom recipe as needed. For instance – there is a need to add intermediate methods for setup. The `LoRAFinetuneRecipeSingleDevice` recipe adds the following methods to support the `setup` method:

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

Creating new Recipes with torchtune has a wide degree of freedom for the engineer, and the pre-built Recipes are excellent starting points for creating custom Recipes, though – the pre-built recipes are examples, and not doctrine. We can experiment!


## More examples

See the [torchtune docs](https://pytorch.org/torchtune/stable/index.html) for additional examples, and join the [official Discord](https://discord.gg/tyRWHtHgV7).

