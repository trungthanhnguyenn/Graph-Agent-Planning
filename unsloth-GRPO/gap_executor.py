import re
import json
from typing import List, Dict, Any, Optional


class GAPParsingError(Exception):
    """Custom exception for GAP parsing errors"""
    pass


class GAPExecutor:
    def __init__(self):
        """
        Initialize GAP Executor with exact regex patterns from xml_tool_parser.py
        """
        # follow logic pattern from verl/verl/tools/xml_tool_parser.py line 37-38
        # pattern = rf"<{escaped_name}>((?:(?!<{escaped_name}>).)*?)</{escaped_name}>"
        self.wiki_search_pattern = re.compile(
            r"<wiki_search>((?:(?!<wiki_search>).)*?)</wiki_search>", 
            re.DOTALL
        )

        # follow logic pattern from verl/verl/utils/reward_score/mhqa_train.py line 95
        # answer_pattern = r'<answer>(.*?)</answer>'
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

    def _parse_wiki_search_queries(self, content: str) -> List[str]:
        """
        EXACT replication of xml_tool_parser.py _parse_wiki_search_queries method
        Parse wiki_search content to handle parallel queries separated by |.
        """
        if '|' in content:
            # Split by | and clean up each query
            queries = [query.strip() for query in content.split('|')]
            # Filter out empty queries
            return [q for q in queries if q]
        else:
            return [content.strip()] if content.strip() else []

    def get_wiki_search_count(self, text: str) -> int:
        """
        EXACT replication of xml_tool_parser.py get_wiki_search_count method
        Get the total number of wiki_search queries (including parallel ones).
        """
        total_count = 0
        for match in self.wiki_search_pattern.finditer(text):
            arguments = match.group(1).strip()
            queries = self._parse_wiki_search_queries(arguments)
            total_count += len(queries)
        return total_count
    
    def get_tool_call_tags_count(self, text: str) -> int:
        """
        Count number of <wiki_search> tags (not queries inside them)
        This is for calculating parallelism efficiency
        """
        return len(self.wiki_search_pattern.findall(text))

    def get_final_answer(self, text: str) -> List[str]:
        """
        EXACT replication of extract_solution from mhqa_train.py
        Extract the final answer from <answer> tags - NO FALLBACKS per directive
        Returns List[str] to handle multiple answers separated by "|"
        """
        matches = list(self.answer_pattern.finditer(text))
        
        if not matches:
            raise GAPParsingError(f"No <answer> tags found in text: {text[:100]}...")
        
        last_content = matches[-1].group(1).strip()
        
        if not last_content:
            raise GAPParsingError(f"Empty <answer> tag found in text: {text[:100]}...")
        
        if '|' in last_content:
            answers = [ans.strip() for ans in last_content.split('|') if ans.strip()]
            if not answers:
                raise GAPParsingError(f"Invalid answer format with empty splits: {last_content}")
            return answers
        
        return [last_content]

    def calculate_parallelism_score(self, text: str) -> float:
        """
        Calculate parallelism efficiency score per specification:
        Score = (Total Queries) / (Number of Tool Call Tags)
        
        This measures how efficiently the model groups queries together
        Higher score = more parallel queries per tool call
        """
        total_queries = self.get_wiki_search_count(text)
        num_tool_tags = self.get_tool_call_tags_count(text)
        
        if num_tool_tags == 0:
            if total_queries > 0:
                raise GAPParsingError(f"Found {total_queries} queries but 0 tool tags - parsing error")
            return 0.0  # Valid case: no tools needed
        
        return float(total_queries) / float(num_tool_tags)