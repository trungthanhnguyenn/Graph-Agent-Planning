import string
import re
from typing import List, Any, Dict
from gap_executor import GAPExecutor, GAPParsingError

_gap_executor = GAPExecutor()

class GAPRewardError(Exception):
    """Custom exception for reward calculation errors"""
    pass

def normalize_answer(s: str) -> str:
    """
    Normalize answer for comparison.
    Retains minimal punctuation for numbers/dates context.
    """
    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()
    
    return white_space_fix(lower(s))

def check_correctness(predictions: List[str], golden_answers: List[str]) -> float:
    """
    Check accuracy between predictions and golden answers.
    Logic: If ANY valid prediction matches ANY golden answer -> Correct.
    """
    if not predictions:
        return 0.0
        
    normalized_golden = [normalize_answer(ans) for ans in golden_answers]
    
    for pred in predictions:
        norm_pred = normalize_answer(pred)
        
        if norm_pred in normalized_golden:
            return 1.0
            
        for gold in normalized_golden:
            if len(gold) > 2 and gold in norm_pred:
                return 1.0
                
    return 0.0

# ============================================================================
# REWARD FUNCTIONS (WEIGHTED INTERNALLY)
# ============================================================================

def correctness_reward_func(prompts: List[str], completions: List[str], answer: List[Any], **kwargs) -> List[float]:
    """
    Reward for CORRECTNESS.
    Weight: HIGH (2.0) - First priority.
    """
    scores = []
    
    # Validation
    if len(completions) != len(answer):
        raise GAPRewardError(f"Batch mismatch: {len(completions)} vs {len(answer)}")
    
    for completion, ground_truth in zip(completions, answer):
        try:
            extracted_answers = _gap_executor.get_final_answer(completion)
            
            if isinstance(ground_truth, str):
                gt_answers = [ground_truth]
            elif isinstance(ground_truth, (list, tuple)):
                gt_answers = [str(ans) for ans in ground_truth]
            else:
                gt_answers = []

            score = check_correctness(extracted_answers, gt_answers)
            
            scores.append(score * 2.0)
            
        except GAPParsingError:
            scores.append(0.0)
        except Exception as e:
            raise GAPRewardError(f"System error in correctness_reward: {e}") from e
            
    return scores

def efficiency_reward_func(prompts: List[str], completions: List[str], answer: List[Any], **kwargs) -> List[float]:
    """
    Reward for PARALLELISM (Using '|').
    Weight: MEDIUM (0.5) - Key constraint for parallel execution.
    """
    scores = []
    for completion in completions:
        try:
            raw_score = _gap_executor.calculate_parallelism_score(completion)
            
            scores.append(raw_score * 0.5)
            
        except GAPParsingError:
            scores.append(0.0)
        except Exception as e:
            raise GAPRewardError(f"System error in efficiency_reward: {e}") from e
    return scores

def format_reward_func(prompts: List[str], completions: List[str], answer: List[Any], **kwargs) -> List[float]:
    """
    Reward for XML FORMAT.
    Weight: LOW (0.1) - Soft constraint.
    """
    scores = []
    for completion in completions:
        try:
            score = 0.0
            
            try:
                _gap_executor.get_final_answer(completion)
                score += 0.05
            except GAPParsingError:
                pass
            
            wiki_count = _gap_executor.get_wiki_search_count(completion)
            if wiki_count > 0:
                score += 0.05
            
            scores.append(min(score, 0.1))
            
        except Exception as e:
            raise GAPRewardError(f"System error in format_reward: {e}") from e
    return scores