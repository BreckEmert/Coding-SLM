# Training_Data/Utility_Functions/fix-unicode.py

import os
import re


class FixUnicode:
    def __init__(self):
        """Holds paths"""
        self.base_dir = os.getenv('WORKSPACE_DIR', os.path.dirname(os.path.abspath(__file__)))
        self.input_file = os.path.join(self.base_dir, "Training_Data", "Individual_Datasets", "Python_Text_to_Code", "combined.json")
        self.output_file = os.path.join(self.base_dir, "Training_Data", "Individual_Datasets", "Python_Text_to_Code", "combined-fixed.json")

    @staticmethod
    def escape_json_strings(s: str) -> str:
        """Fixes unparseable jsons due to escaping, likely from unicode characters"""
        # Escape backslashes first
        s = re.sub(r'(?<!\\)\\(?!u[0-9A-Fa-f]{4})', r'\\\\', s)
        # Escape double quotes
        s = s.replace('"', '\\"')
        return s

    def fix_json_strings(self) -> None:
        with open(self.input_file, 'r', encoding='utf-8') as file:
            strings = file.read()

        fixed_strings = re.sub(r'(".*?")', lambda m: '"' + self.escape_json_strings(m.group(1)[1:-1]) + '"', strings)

        with open(self.output_file, 'w', encoding='utf-8') as file:
            file.write(fixed_strings)

if __name__ == "__main__":
    fixer = FixUnicode()
    fixer.fix_json_strings()
