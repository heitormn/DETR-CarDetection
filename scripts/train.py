from pathlib import Path
import yaml
import torch
from torch.optim import AdamW
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, TrainingArguments, Trainer
from transformers import get_cosine_schedule_with_warmup
from utils import KITTIDetectionDataset


def collate_fn(batch):
	"""Custom collate function for RT-DETR batch processing."""
	pixel_values = torch.stack([item['pixel_values'] for item in batch])
	
	# Convert labels to proper format for RT-DETR
	batch_labels = []
	for item in batch:
		labels = item['labels']
		
		if isinstance(labels, list):
			if len(labels) > 0:
				class_labels = torch.cat([ann["class_labels"] for ann in labels])
				boxes = torch.cat([ann["boxes"] for ann in labels])
			else:
				class_labels = torch.tensor([], dtype=torch.int64)
				boxes = torch.tensor([], dtype=torch.float32).reshape(0, 4)
		else:
			# Extract class_labels and boxes from BatchFeature or dict
			try:
				class_labels = labels["class_labels"]
				boxes = labels["boxes"]
			except (TypeError, KeyError):
				# Fallback: empty tensors
				class_labels = torch.tensor([], dtype=torch.int64)
				boxes = torch.tensor([], dtype=torch.float32).reshape(0, 4)
		
		# Ensure tensors
		if not isinstance(class_labels, torch.Tensor):
			class_labels = torch.tensor(class_labels, dtype=torch.int64)
		if not isinstance(boxes, torch.Tensor):
			boxes = torch.tensor(boxes, dtype=torch.float32)
		
		batch_labels.append({
			'class_labels': class_labels,
			'boxes': boxes
		})
	
	return {
		'pixel_values': pixel_values,
		'labels': batch_labels
	}


class RTDetrTrainer(Trainer):
	"""Custom Trainer with separate learning rates for backbone and head."""
	
	def __init__(self, *args, backbone_lr_factor=0.1, **kwargs):
		self.backbone_lr_factor = backbone_lr_factor
		super().__init__(*args, **kwargs)
	
	def create_optimizer(self):
		"""Create optimizer with different learning rates for backbone vs head."""
		if self.optimizer is not None:
			return self.optimizer
		
		# Separate parameters into backbone and head
		backbone_params = []
		head_params = []
		
		for name, param in self.model.named_parameters():
			if not param.requires_grad:
				continue
			# Classification head and box head get higher learning rate
			if 'class_embed' in name or 'bbox_embed' in name or 'query' in name:
				head_params.append(param)
			else:
				backbone_params.append(param)
		
		base_lr = self.args.learning_rate
		backbone_lr = base_lr * self.backbone_lr_factor
		
		print(f"\n[Optimizer] Using differential learning rates:")
		print(f"  - Backbone params: {len(backbone_params)} tensors, LR={backbone_lr:.2e}")
		print(f"  - Head params: {len(head_params)} tensors, LR={base_lr:.2e}")
		
		optimizer_grouped_parameters = [
			{
				"params": backbone_params,
				"lr": backbone_lr,
				"weight_decay": self.args.weight_decay,
			},
			{
				"params": head_params,
				"lr": base_lr,
				"weight_decay": self.args.weight_decay,
			},
		]
		
		self.optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr)
		return self.optimizer




CONFIG_PATH_MODEL = Path(__file__).resolve().parent.parent / "configs" / "model.yaml"
CONFIG_PATH_TRAIN = Path(__file__).resolve().parent.parent / "configs" / "train.yaml"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_model_id(config_path: Path = CONFIG_PATH_MODEL) -> str:
	"""Read the model id from the YAML config."""
	with open(config_path, "r", encoding="utf-8") as file:
		config = yaml.safe_load(file) or {}

	try:
		return config["model"]["id"]
	except (KeyError, TypeError) as exc:
		raise KeyError(f"Expected 'model.id' in {config_path}") from exc


def load_training_config(config_path: Path = CONFIG_PATH_TRAIN) -> dict:
	"""Read the training config from the YAML config."""
	with open(config_path, "r", encoding="utf-8") as file:
		config = yaml.safe_load(file) or {}

	try:
		return config["training"]
	except (KeyError, TypeError) as exc:
		raise KeyError(f"Expected 'training' in {config_path}") from exc


def create_model(ignore_mismatched_sizes=True):
	"""
	Instantiate model and processor from a pretrained identifier.
	
	Args:
		ignore_mismatched_sizes: If True, ignore size mismatches when loading 
			pretrained weights (needed when changing num_labels)
	"""
	model_id = load_model_id()
	image_processor = RTDetrImageProcessor.from_pretrained(model_id)
	
	# Load model with new number of labels
	# ignore_mismatched_sizes=True allows loading pretrained weights even though
	# the classification head has different dimensions
	model = RTDetrForObjectDetection.from_pretrained(
		model_id, 
		num_labels=3,
		ignore_mismatched_sizes=ignore_mismatched_sizes
	)
	
	# Reinitialize classification head with proper initialization
	# This is crucial for the model to learn properly
	print("\n[Model] Reinitializing classification head for 3 classes...")
	_reinit_classification_head(model)
	
	return image_processor, model


def _reinit_classification_head(model):
	"""
	Properly reinitialize the classification head.
	
	The default initialization when changing num_labels can be poor.
	We use a bias initialization that assumes uniform class distribution.
	"""
	import math
	
	# Get the class embedding layers (there are multiple for auxiliary losses)
	for name, module in model.named_modules():
		if 'class_embed' in name and hasattr(module, 'weight'):
			# Xavier/Glorot uniform initialization for weights
			torch.nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				# Initialize bias based on prior probability
				# prior_prob = 0.01 means we start with low confidence
				prior_prob = 0.01
				bias_value = -math.log((1 - prior_prob) / prior_prob)
				torch.nn.init.constant_(module.bias, bias_value)
			print(f"  Reinitialized: {name}")


def create_datasets():
	"""Create train and validation datasets."""
	# Create model and processor
	image_processor, _ = create_model()
	
	# Create datasets
	train_dataset = KITTIDetectionDataset(
		split_dir=PROCESSED_DATA_DIR / 'train',
		image_processor=image_processor,
		augment=True
	)
	
	val_dataset = KITTIDetectionDataset(
		split_dir=PROCESSED_DATA_DIR / 'val',
		image_processor=image_processor,
		augment=False
	)
	
	print(f"Train dataset: {len(train_dataset)} images")
	print(f"Val dataset: {len(val_dataset)} images")
	
	return train_dataset, val_dataset, image_processor



if __name__ == "__main__":
	# Load training config
	training_config = load_training_config()
	
	# Create datasets
	print("Creating datasets...")
	train_dataset, val_dataset, image_processor = create_datasets()
	
	# Create model with properly initialized classification head
	print("\nLoading model...")
	_, model = create_model()
	
	# Training hyperparameters
	# Use higher learning rate for the head (which needs to learn from scratch)
	# and lower learning rate for backbone (which is pretrained)
	base_lr = float(training_config.get('learning_rate', 1e-4))
	backbone_lr_factor = float(training_config.get('backbone_lr_factor', 0.1))
	
	print(f"\n[Training Config]")
	print(f"  Base LR (head): {base_lr:.2e}")
	print(f"  Backbone LR: {base_lr * backbone_lr_factor:.2e}")
	print(f"  Epochs: {training_config.get('epochs', 20)}")
	print(f"  Batch size: {training_config.get('batch_size', 8)}")
	print("\nSetup complete! Ready for training.")

	# Setup training arguments from config
	training_args = TrainingArguments(
		output_dir="./results",
		num_train_epochs=int(training_config.get('epochs', 20)),
		per_device_train_batch_size=int(training_config.get('batch_size', 8)),
		per_device_eval_batch_size=int(training_config.get('batch_size', 8)),
		learning_rate=base_lr,
		warmup_ratio=float(training_config.get('warmup_ratio', 0.1)),
		weight_decay=float(training_config.get('weight_decay', 0.0001)),
		gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps', 4)),
		eval_strategy=training_config.get('eval_strategy', 'epoch'),
		save_strategy=training_config.get('save_strategy', 'epoch'),
		logging_steps=int(training_config.get('logging_steps', 50)),
		report_to=training_config.get('report_to', 'none'),
		load_best_model_at_end=training_config.get('load_best_model_at_end', True),
		metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
		lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
		max_grad_norm=float(training_config.get('max_grad_norm', 0.1)),  # Lower gradient clipping
		save_total_limit=int(training_config.get('save_total_limit', 3)),
		fp16=torch.cuda.is_available(),  # Use mixed precision if available
		dataloader_num_workers=int(training_config.get('num_workers', 4)),
		remove_unused_columns=False,  # Important for custom collate_fn
	)
	
	# Create custom Trainer with differential learning rates
	trainer = RTDetrTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		data_collator=collate_fn,
		backbone_lr_factor=backbone_lr_factor,
	)
	
	print("\n" + "="*60)
	print("Starting training...")
	print("="*60 + "\n")
	
	trainer.train()
