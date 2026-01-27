from pathlib import Path
import yaml
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, TrainingArguments, Trainer


CONFIG_PATH_model= Path(__file__).resolve().parent.parent / "configs" / "model.yaml"


def load_model_id(config_path: Path = CONFIG_PATH_model) -> str:
	"""Read the model id from the YAML config."""
	with open(config_path, "r", encoding="utf-8") as file:
		config = yaml.safe_load(file) or {}

	try:
		return config["model"]["id"]
	except (KeyError, TypeError) as exc:
		raise KeyError(f"Expected 'model.id' in {config_path}") from exc
	
def load_training_config(config_path: Path = CONFIG_PATH_model) -> dict:
    """Read the training config from the YAML config."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    try:
        return config["training"]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Expected 'training' in {config_path}") from exc


def create_model():
	"""Instantiate model and processor from a pretrained identifier."""
	model_id = load_model_id()
	image_processor = RTDetrImageProcessor.from_pretrained(model_id)
	model = RTDetrForObjectDetection.from_pretrained(model_id)
	return image_processor, model



if __name__ == "__main__":
	image_processor, model = create_model()
	training_config = load_training_config()