import pandas as pd
import os
import numpy as np 
from typing import List, Dict, Any, Tuple
from datasets import Dataset

class GAPDataError(Exception):
    """Custom exception for data loading errors"""
    pass

class GAPDataLoader:
    def __init__(self, dataset_path: str, project_root: str):
        self.project_root = project_root
        
        if os.path.exists(dataset_path):
            self.dataset_path = dataset_path
        elif os.path.exists(os.path.join(project_root, dataset_path)):
            self.dataset_path = os.path.join(project_root, dataset_path)
        elif os.path.isabs(dataset_path) and os.path.exists(dataset_path):
            self.dataset_path = dataset_path
        else:
            cwd = os.getcwd()
            raise GAPDataError(
                f"Dataset file not found.\n"
                f" - Checked relative: {dataset_path}\n"
                f" - Checked joined: {os.path.join(project_root, dataset_path)}\n"
                f" - Current Dir: {cwd}"
            )
        
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        try:
            print(f"Reading parquet from: {self.dataset_path}")
            self.df = pd.read_parquet(self.dataset_path)
        except Exception as e:
            raise GAPDataError(f"Failed to load parquet file {self.dataset_path}: {e}") from e
        
        if len(self.df) == 0:
            raise GAPDataError(f"Dataset is empty: {self.dataset_path}")
        
        print(f"Loaded GAP dataset: {len(self.df)} samples")
    
    def _extract_ground_truth(self, row: pd.Series) -> List[str]:
        """
        Extract answer from dataset
        """
        try:
            if 'answer' in row:
                val = row['answer']
                if val is None: return []
                if isinstance(val, str): return [val]
                # add support for list/array
                if isinstance(val, (list, tuple, np.ndarray)): 
                    return [str(v) for v in val]
            
            if 'reward_model' in row:
                rm = row['reward_model']
                if isinstance(rm, dict):
                    ground_truth = rm.get('ground_truth', {})
                    target = ground_truth.get('target', [])
                    if isinstance(target, str): return [target]
                    if isinstance(target, (list, tuple, np.ndarray)):
                        return [str(item) for item in target]
            
            return []
        except Exception as e:
            return [] 
    
    def _extract_prompt_content(self, raw_prompt: Any) -> str:
        """
        Extract 'content' from the 'prompt' field
        """
        try:
            if isinstance(raw_prompt, (list, np.ndarray, tuple)) and len(raw_prompt) > 0:
                first_item = raw_prompt[0]
                if isinstance(first_item, dict) and 'content' in first_item:
                    return first_item['content']
            
            if isinstance(raw_prompt, str):
                return raw_prompt
                
            return ""
        except Exception:
            return ""

    def prepare_grpo_dataset(self, max_samples: int = None) -> Dataset:
        """Prepare dataset for GRPO training"""
        df_subset = self.df.head(max_samples) if max_samples else self.df
        
        processed_data = {
            'prompt': [],
            'answer': [] 
        }
        
        print("⚙️ Processing dataset rows...")
        error_count = 0
        first_error = None

        for idx, row in df_subset.iterrows():
            try:
                user_content = self._extract_prompt_content(row['prompt'])
                if not user_content:
                    if error_count == 0:
                        print(f"DEBUG: Failed to extract prompt at row {idx}. Raw type: {type(row['prompt'])}")
                        print(f"DEBUG: Raw content: {str(row['prompt'])[:100]}...")
                    error_count += 1
                    continue

                full_prompt = (
                    f"<|im_start|>user\n{user_content}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                
                ground_truth = self._extract_ground_truth(row)
                if not ground_truth:
                    if error_count == 0:
                        print(f"DEBUG: No ground truth found at row {idx}")
                    error_count += 1
                    continue 
                
                processed_data['prompt'].append(full_prompt)
                processed_data['answer'].append(ground_truth)
                
            except Exception as e:
                error_count += 1
                if first_error is None:
                    first_error = str(e)
                continue
        
        if len(processed_data['prompt']) == 0:
            msg = f"No valid samples after processing! Failed rows: {error_count}."
            if first_error:
                msg += f" First error: {first_error}"
            raise GAPDataError(msg)

        print(f"Processed {len(processed_data['prompt'])} valid samples for GRPO")
        return Dataset.from_dict(processed_data)

def load_gap_dataset(
    project_root: str = ".",
    dataset_path: str = "GAP-MHQA-RL-Dataset/GAP-RL-16w.parquet",
    max_samples: int = None
) -> Dataset:
    if not project_root: project_root = "."
    loader = GAPDataLoader(dataset_path, project_root)
    return loader.prepare_grpo_dataset(max_samples)

LEGACY_PATHS = {
    'BASE_MODEL': 'experiments/merged_model',
    'TRAIN_DATASETS': 'GAP-MHQA-RL-Dataset/GAP-RL-16w.parquet',
    'PROJECT_ROOT': '.'
}