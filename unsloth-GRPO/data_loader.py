import pandas as pd
import os
from typing import List, Dict, Any, Tuple
from datasets import Dataset


class GAPDataError(Exception):
    """Custom exception for data loading errors"""
    pass


GAP_SYSTEM_PROMPT = """You are an intelligent AI assistant that can search for information and provide accurate answers. You have access to a wiki_search function that can retrieve relevant information.

CRITICAL INSTRUCTION: If you need to search for multiple independent entities, you MUST combine them into a single tool call separated by the '|' character. For example:
- Good: <wiki_search>Entity1 | Entity2 | Entity3</wiki_search>
- Bad: <wiki_search>Entity1</wiki_search> then <wiki_search>Entity2</wiki_search>

This parallel search approach is more efficient and is the preferred method.

Available functions:
- wiki_search: Search for information. Use | to separate multiple queries for parallel execution.
- answer: Provide your final answer in <answer>...</answer> tags.

Always end your response with <answer>your final answer</answer>."""


class GAPDataLoader:
    def __init__(self, dataset_path: str, project_root: str):
        """
        Initialize GAP Data Loader
        
        Args:
            dataset_path: Path to parquet file relative to project_root
            project_root: Absolute path to Graph-Agent-Planning directory
        """
        self.project_root = project_root
        
        if os.path.isabs(dataset_path):
            self.dataset_path = dataset_path
        else:
            self.dataset_path = os.path.join(project_root, dataset_path)
        
        if not os.path.exists(self.project_root):
            raise GAPDataError(f"Project root not found: {self.project_root}")
        
        if not os.path.exists(self.dataset_path):
            raise GAPDataError(f"Dataset file not found: {self.dataset_path}")
        
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load and validate parquet dataset"""
        try:
            self.df = pd.read_parquet(self.dataset_path)
        except Exception as e:
            raise GAPDataError(f"Failed to load parquet file {self.dataset_path}: {e}") from e
        
        required_columns = ['question', 'answer', 'prompt', 'reward_model']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise GAPDataError(f"Missing required columns: {missing_columns}. Found: {list(self.df.columns)}")
        
        if len(self.df) == 0:
            raise GAPDataError(f"Dataset is empty: {self.dataset_path}")
        
        print(f"Loaded GAP dataset: {len(self.df)} samples from {self.dataset_path}")
    
    def _extract_ground_truth(self, reward_model_data: Dict[str, Any]) -> List[str]:
        """Extract ground truth from reward_model column"""
        try:
            ground_truth = reward_model_data.get('ground_truth', {})
            target = ground_truth.get('target', [])
            
            if isinstance(target, str):
                return [target]
            elif isinstance(target, (list, tuple)):
                return [str(item) for item in target]
            else:
                raise GAPDataError(f"Invalid ground truth format: {type(target)}")
                
        except Exception as e:
            raise GAPDataError(f"Failed to extract ground truth: {e}") from e
    
    def _format_conversation(self, question: str) -> str:
        """Format question into conversation with system prompt"""
        return f"System: {GAP_SYSTEM_PROMPT}\n\nUser: {question}\n\nAssistant: "
    
    def prepare_grpo_dataset(self, max_samples: int = None) -> Dataset:
        """
        Prepare dataset for GRPO training with proper format
        
        Args:
            max_samples: Limit number of samples for testing (None = use all)
            
        Returns:
            HuggingFace Dataset with prompt, answer columns
        """
        df_subset = self.df.head(max_samples) if max_samples else self.df
        
        processed_data = {
            'prompt': [],
            'answer': []  # Ground truth for reward calculation
        }
        
        for idx, row in df_subset.iterrows():
            try:
                formatted_prompt = self._format_conversation(row['question'])
                
                ground_truth = self._extract_ground_truth(row['reward_model'])
                
                processed_data['prompt'].append(formatted_prompt)
                processed_data['answer'].append(ground_truth)
                
            except Exception as e:
                print(f"Skipping corrupted row {idx}: {e}")
                continue
        
        if len(processed_data['prompt']) == 0:
            raise GAPDataError("No valid samples after processing")

        print(f"Processed {len(processed_data['prompt'])} valid samples for GRPO training")

        return Dataset.from_dict(processed_data)


def load_gap_dataset(
    project_root: str = "Graph-Agent-Planning",
    dataset_path: str = "GAP-MHQA-RL-Dataset/GAP-RL-16w.parquet",
    max_samples: int = None
) -> Dataset:
    """
    Convenience function to load GAP dataset for GRPO training
    
    Args:
        project_root: Path to Graph-Agent-Planning directory  
        dataset_path: Relative path to parquet file
        max_samples: Limit samples for testing
        
    Returns:
        HuggingFace Dataset ready for GRPO training
    """
    loader = GAPDataLoader(dataset_path, project_root)
    return loader.prepare_grpo_dataset(max_samples)


LEGACY_PATHS = {
    'BASE_MODEL': 'experiments/merged_model',
    'TRAIN_DATASETS': 'GAP-MHQA-RL-Dataset/GAP-RL-16w.parquet',
    'PROJECT_ROOT': 'Graph-Agent-Planning'
}
