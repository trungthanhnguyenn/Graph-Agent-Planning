import string
import re
from typing import List, Any, Dict
from gap_executor import GAPExecutor, GAPParsingError


class GAPRewardError(Exception):
    """Custom exception for reward calculation errors"""
    pass

_gap_executor = GAPExecutor()


def normalize_answer(s: str) -> str:
    """
    Normalize answer for exact match comparison
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(predictions: List[str], golden_answers: List[str]) -> int:
    """
    Check exact match between predictions and golden answers
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    normalized_golden = [normalize_answer(ans) for ans in golden_answers]
    
    predictions = [prediction for prediction in predictions if prediction]

    if not predictions:
        return 0
    
    for prediction in predictions:
        normalized_pred = normalize_answer(prediction)
        if normalized_pred not in normalized_golden:
            return 0
    return 1


def correctness_reward_func(
    prompts: List[str], 
    completions: List[str], 
    answer: List[Any], 
    **kwargs
) -> List[float]:
    """
    Correctness reward based on Exact Match (EM) with ground truth
    Following exact logic from mhqa_train.py compute_score_em_batch
    
    Args:
        prompts: List of input questions
        completions: List of model generated responses  
        answer: List of ground truth answers from dataset
        
    Returns:
        List[float]: Correctness scores (1.0 for exact match, 0.0 otherwise)
    """
    if len(completions) != len(answer):
        raise GAPRewardError(
            f"Batch size mismatch: {len(completions)} completions vs {len(answer)} answers"
        )
    
    scores = []
    
    for completion, ground_truth in zip(completions, answer):
        try:
            extracted_answers = _gap_executor.get_final_answer(completion)
            
            if isinstance(ground_truth, str):
                gt_answers = [ground_truth]
            elif isinstance(ground_truth, (list, tuple)):
                gt_answers = [str(ans) for ans in ground_truth]
            else:
                raise GAPRewardError(f"Invalid ground truth format: {type(ground_truth)}")
            
            em_score = em_check(extracted_answers, gt_answers)
            scores.append(float(em_score))
            
        except GAPParsingError as e:
            scores.append(0.0)
        
        except Exception as e:
            raise GAPRewardError(f"Unexpected error in correctness reward: {e}") from e
    
    return scores


def efficiency_reward_func(
    prompts: List[str], 
    completions: List[str], 
    answer: List[Any], 
    **kwargs
) -> List[float]:
    """
    Efficiency reward based on parallelism score
    Rewards models that use parallel wiki_search queries (Query1 | Query2)
    
    Args:
        prompts: List of input questions
        completions: List of model generated responses
        answer: List of ground truth (not used for efficiency)
        
    Returns:
        List[float]: Parallelism efficiency scores
    """
    scores = []
    
    for completion in completions:
        try:
            parallelism_score = _gap_executor.calculate_parallelism_score(completion)
            scores.append(parallelism_score)
            
        except GAPParsingError as e:
            scores.append(0.0)
            
        except Exception as e:
            raise GAPRewardError(f"Unexpected error in efficiency reward: {e}") from e
    
    return scores


def format_reward_func(
    prompts: List[str], 
    completions: List[str], 
    answer: List[Any], 
    **kwargs
) -> List[float]:
    """
    Format reward for proper response structure
    Checks if model follows the expected format with proper tags
    
    Args:
        prompts: List of input questions
        completions: List of model generated responses
        answer: List of ground truth (not used for format)
        
    Returns:
        List[float]: Format compliance scores (1.0 for proper format, 0.0 otherwise)
    """
    scores = []
    
    for completion in completions:
        try:
            score = 0.0
            
            try:
                _gap_executor.get_final_answer(completion)
                score += 0.5
            except GAPParsingError:
                pass
            
            wiki_count = _gap_executor.get_wiki_search_count(completion)
            if wiki_count > 0:
                score += 0.3
                
                tag_count = _gap_executor.get_tool_call_tags_count(completion)
                if wiki_count > tag_count:
                    score += 0.2
            
            scores.append(min(score, 1.0))
            
        except Exception as e:
            raise GAPRewardError(f"Unexpected error in format reward: {e}") from e
    
    return scores


def combined_reward_func(
    prompts: List[str], 
    completions: List[str], 
    answer: List[Any], 
    **kwargs
) -> List[float]:
    """
    Combined reward function with weighted components
    Combines correctness (primary), efficiency, and format rewards
    
    Weights:
    - Correctness: 0.7 (most important - getting right answer)
    - Efficiency: 0.2 (parallelism optimization)
    - Format: 0.1 (proper structure)
    """
    correctness_scores = correctness_reward_func(prompts, completions, answer, **kwargs)
    efficiency_scores = efficiency_reward_func(prompts, completions, answer, **kwargs)
    format_scores = format_reward_func(prompts, completions, answer, **kwargs)
    
    combined_scores = []
    for corr, eff, fmt in zip(correctness_scores, efficiency_scores, format_scores):
        # Weighted combination - correctness is most important
        combined = 0.7 * corr + 0.2 * eff + 0.1 * fmt
        combined_scores.append(combined)
    
    return combined_scores
