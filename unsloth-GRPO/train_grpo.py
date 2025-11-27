import os
import torch
import wandb
import sys

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Import custom components
from data_loader import load_gap_dataset, LEGACY_PATHS, GAPDataError
from rewards import correctness_reward_func, efficiency_reward_func, format_reward_func, GAPRewardError

class GAPTrainingError(Exception):
    """Custom exception for training errors"""
    pass

class GAPGRPOTrainer:
    def __init__(self, model_path=None, dataset_path=None, project_root=None, max_samples=None, output_dir=None):
        self.model_path = model_path or LEGACY_PATHS['BASE_MODEL']
        self.dataset_path = dataset_path or LEGACY_PATHS['TRAIN_DATASETS'] 
        self.project_root = project_root or LEGACY_PATHS['PROJECT_ROOT']
        self.max_samples = max_samples
        
        self.output_dir = output_dir or os.path.join(
            self.project_root, "unsloth-GRPO", "outputs", "gap_grpo_model"
        )
        
        self._validate_paths()
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        
        print(f"Initializing GAP GRPO Trainer")
        print(f"   Model: {self.model_path}")
        print(f"   Dataset: {self.dataset_path}")

    def _validate_paths(self):
        if not os.path.exists(self.model_path): raise GAPTrainingError(f"Base model not found: {self.model_path}")
        if not os.path.exists(self.dataset_path): raise GAPTrainingError(f"Dataset not found: {self.dataset_path}")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        print("Loading model ...")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=4096,
                load_in_4bit=True,
                fast_inference=True,
                gpu_memory_utilization=0.3,
            )
            
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=64, 
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=64,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
            print(f"Model loaded successfully")
        except Exception as e:
            raise GAPTrainingError(f"Failed to load model: {e}") from e
    
    def load_dataset(self) -> None:
        """Load and prepare dataset for GRPO training"""
        print("Loading GAP dataset...")
        try:
            self.dataset = load_gap_dataset(
                project_root=self.project_root,
                dataset_path=self.dataset_path,
                max_samples=self.max_samples
            )
            print(f"Dataset loaded: {len(self.dataset)} samples")
        except (GAPDataError, Exception) as e:
            raise GAPTrainingError(f"Failed to load dataset: {e}") from e
    
    def setup_trainer(self):
            print("Setting up GRPO trainer...")
            if not self.model or not self.tokenizer or not self.dataset:
                raise GAPTrainingError("Model/Dataset not loaded")
            
            try:
                per_device_batch_size = 1
                num_generations = 4
                gradient_accumulation_steps = 8 # Effective Batch = 32
                
                grpo_config = GRPOConfig(
                    output_dir=self.output_dir,
                    learning_rate=1e-6,
                    per_device_train_batch_size=per_device_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    num_generations=num_generations,
                    max_prompt_length=1024,
                    max_completion_length=1500,
                    num_train_epochs=1,
                    bf16=True, 
                    gradient_checkpointing=True,
                    logging_steps=1,
                    save_steps=50,
                    report_to="wandb",
                    use_vllm=True,
                    vllm_gpu_memory_utilization=0.2,
                )
                
                self.trainer = GRPOTrainer(
                    model=self.model,
                    processing_class=self.tokenizer,
                    reward_funcs=[correctness_reward_func, efficiency_reward_func, format_reward_func],
                    args=grpo_config,
                    train_dataset=self.dataset,
                )
                print(f"GRPO configured. Group Size: {num_generations}")
            except Exception as e:
                raise GAPTrainingError(f"Failed to setup trainer: {e}") from e
    
    def train(self) -> None:
        """Execute GRPO training"""
        if self.trainer is None:
            raise GAPTrainingError("Trainer must be setup before training")

        print("Starting GRPO training...")
        try:
            wandb.init(
                project="GAP-GRPO",
                name="gap_parallel_search_training",
                config={
                    "model_path": self.model_path,
                    "dataset_size": len(self.dataset),
                    "max_samples": self.max_samples,
                }
            )
            
            self.trainer.train()

            print("Saving trained model...")
            self.model.save_lora(os.path.join(self.output_dir, "final_lora"))
            self.tokenizer.save_pretrained(self.output_dir)
            print(f"Training completed!")

        except Exception as e:
            raise GAPTrainingError(f"Training failed: {e}") from e
        finally:
            wandb.finish()
    
    def run_full_training(self) -> None:
        print("Starting GAP GRPO Training Pipeline")
        try:
            self.load_model()
            self.load_dataset() 
            self.setup_trainer()
            self.train()
            print("GAP GRPO training completed successfully!")
        except Exception as e:
            print(f"Training pipeline failed: {e}")
            raise

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID (e.g., '0')")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"Using GPU: {args.gpu_id}")

    trainer = GAPGRPOTrainer(max_samples=args.max_samples)
    trainer.run_full_training()

if __name__ == "__main__":
    main()