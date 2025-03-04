import wandb
import dotenv
import os

dotenv.load_dotenv()

key: str = os.getenv("WANDB_API_KEY", "")

wandb.login(key=key, verify=True)
