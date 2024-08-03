import json
import re
import os


class FixUnicode:
    """Removes entries with excessive unicode sequences in JSON files."""

    def __init__(self, base_dir: str):
        """Initialize paths."""
        self.base_dir = base_dir

    def remove_entries_with_unicode_sequences(self, input_file: str, output_file: str) -> None:
        """Remove entries with 5 or more consecutive unicode sequences from the JSON file."""
        unicode_pattern = re.compile(r'(\\u[0-9A-Fa-f]{4}){5,}')
        
        # Read the JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Filter entries that match the pattern
        cleaned_data = [
            entry for entry in data
            if not unicode_pattern.search(json.dumps(entry))
        ]

        # Write out the cleaned JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Paths
    base_dir = os.getenv('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "Training_Data", "Individual_Datasets", "Evol_Instruct", "Evol-Instruction-66k.json")
    output_file = os.path.join(base_dir, "Training_Data", "Individual_Datasets", "Evol_Instruct", "Evol-Instruction-66k-cleaned.json")
    
    # Remove bad entries
    fixer = FixUnicode(base_dir)
    fixer.remove_entries_with_unicode_sequences(input_file, output_file)
