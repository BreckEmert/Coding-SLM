# Training_Data/text_filtering.py

import re
import unicodedata
from unidecode import unidecode
from langdetect import detect, DetectorFactory, lang_detect_exception

# Fix the DetectorFactory seed
DetectorFactory.seed = 0


class TextFiltering:
    """Various functions to clean and filter text (including code)."""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    @staticmethod
    def replace_special_characters(text: str) -> str:
        return unidecode(text)

    @staticmethod
    def is_english(text: str) -> bool:
        try:
            return detect(text) == 'en'
        except lang_detect_exception.LangDetectException:
            return False

    def filter_language(self, text: str) -> str:
        """Remove comments in languages other than English."""
        single_line_comment_pattern = re.compile(r'#.*$', re.MULTILINE)
        multi_line_comment_pattern = re.compile(r'\'\'\'.*?\'\'\'|\"\"\".*?\"\"\"', re.DOTALL)

        def replace_comment(match: re.Match) -> str:
            comment = match.group(0)
            return comment if self.is_english(comment) else ''

        text = re.sub(single_line_comment_pattern, replace_comment, text)
        text = re.sub(multi_line_comment_pattern, replace_comment, text)
        text = "\n".join(line for line in text.splitlines() if line.strip())

        return text

    def format_text(self, text: str) -> str:
        """Normalize and replace special characters."""
        text = unicodedata.normalize('NFKD', text)
        text = self.replace_special_characters(text)
        return text

    def format_examples(self, problems: list[str], solutions: list[str]) -> tuple[list[str], list[str]]:
        """Format examples by normalizing and replacing special characters."""
        formatted_problems = [self.format_text(problem) for problem in problems]
        formatted_solutions = [self.format_text(solution) for solution in solutions]
        return formatted_problems, formatted_solutions

    def filter_examples(
            self, problems: list[str], solutions: list[str], check_language: bool = True
    ) -> tuple[list[str], list[str]]:
        """Filter out non-English examples if check_language is True."""
        formatted_problems, formatted_solutions = self.format_examples(problems, solutions)
        filtered_problems = []
        filtered_solutions = []

        for problem, solution in zip(formatted_problems, formatted_solutions):
            texts = [problem, solution]
            is_english = all(self.is_english(text) for text in texts) if check_language else True

            if is_english:
                filtered_problems.append(problem)
                filtered_solutions.append(solution)

        return filtered_problems, filtered_solutions
