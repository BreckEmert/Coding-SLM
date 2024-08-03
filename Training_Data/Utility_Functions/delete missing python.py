import json
import os


class JSONLFilter:
    """Filter JSONL to only lines that have code."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def filter(self, input_filepath: str, output_filepath: str) -> None:
        """Filter JSONL lines that contain '```python' in the 'python' field."""
        with open(input_filepath, 'r', encoding='utf-8') as infile, open(output_filepath, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    data = json.loads(line)
                    if 'python' in data and '```python' in data['python']:
                        outfile.write(json.dumps(data) + '\n')
                except json.JSONDecodeError:
                    continue


if __name__ == "__main__":
    # Paths
    base_dir = os.getenv('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))
    input_filepath = os.path.join(base_dir, "Training_Data", "LeetCode_Train", "leetcode-train.jsonl")
    output_filepath = os.path.join(base_dir, "Training_Data", "LeetCode_Train", "leetcode-train-filtered.jsonl")

    # Filter lines without code
    filter = JSONLFilter(base_dir)
    filter.filter(input_filepath, output_filepath)
