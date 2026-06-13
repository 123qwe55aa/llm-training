#!/usr/bin/env python3
"""Fix collapsed code blocks in MIT 6.036 exercise markdown files.

The original converter collapsed multi-line code templates into single lines
with digit-sequence line-number prefixes. This script detects those and restores
proper ```-fenced multi-line code blocks.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [
    ROOT / "course-data" / "content",
    ROOT / "week1-blog" / "content",
]

# Matches a line that starts with digit-run and ends with XX...X
# e.g. "12345def one_hot(x, k): passXXXXXXXXXXXXXXXXXXXX"
COLLAPSED_RE = re.compile(r"^(\d+)(.*?)(X{3,})$")


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    lines = text.split("\n")
    new_lines = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        m = COLLAPSED_RE.match(line)
        if m:
            # This is a collapsed code block — expand it
            line_numbers = m.group(1)   # "12345678910..."
            code_body = m.group(2)       # "def one_hot(x, k): pass"
            # Collapse digits into actual line numbers
            # The digits are just sequential indices: 1, 2, 3, 4, 5...
            # The code might have line-number prefixes embedded inside
            # Let's try to split the code body intelligently
            
            # First, try to split the line_numbers into individual digits
            # and use them as delimiter positions
            num_part = line_numbers  # e.g. "12345678910..."
            code = code_body         # e.g. "def one_hot(x, k): pass"
            
            # Method: walk through line_numbers, each digit says how many chars
            # of code_body belong to that line... but this is unreliable.
            # 
            # Better approach: the code template is often Python with pass as placeholder.
            # Just wrap the entire thing in ``` fences with a python marker.
            # Remove the leading digit run and trailing XXX before wrapping.
            
            clean_code = line
            # Remove leading digits
            clean_code = re.sub(r"^\d+", "", clean_code)
            # Remove trailing XXX
            clean_code = re.sub(r"X{3,}$", "", clean_code)
            clean_code = clean_code.strip()
            
            if clean_code:
                changed = True
                new_lines.append("```python")
                # Try to insert newlines before 'def ', 'class ', 'import ' 
                # to make it readable
                formatted = re.sub(r"(?=def |class |import )", "\n", clean_code).strip()
                new_lines.append(formatted)
                new_lines.append("```")
            else:
                # If nothing meaningful, keep original line
                new_lines.append(line)
        else:
            new_lines.append(line)
        i += 1

    result = "\n".join(new_lines)
    if result != original:
        path.write_text(result, encoding="utf-8")
        return True
    return False


def fix_heading_section_titles(path: Path) -> bool:
    """Add ## prefix to standalone section title lines that should be headings."""
    text = path.read_text(encoding="utf-8")
    original = text
    lines = text.split("\n")
    new_lines = []
    
    # Section title patterns found in the exercise files
    section_titles = {
        "Feature Transformations",
        "Scaling",
        "Encoding Discrete Values",
        "Polynomial Features",
        "Experiments",
        "Practice Problems",
        "Preliminaries",
        "Classification",
        "Evaluating algorithmic and feature choices for AUTO data",
        "Making choices",
    }
    
    for line in lines:
        stripped = line.strip()
        if stripped in section_titles:
            new_lines.append(f"## {stripped}")
        else:
            new_lines.append(line)
    
    result = "\n".join(new_lines)
    if result != original:
        path.write_text(result, encoding="utf-8")
        return True
    return False


def main():
    md_files = []
    for d in DIRS:
        md_files.extend(d.rglob("*.md"))
    
    fixed_code = 0
    fixed_heading = 0
    
    for path in md_files:
        if fix_file(path):
            print(f"  code:  {path.relative_to(ROOT)}")
            fixed_code += 1
        if fix_heading_section_titles(path):
            print(f"  title: {path.relative_to(ROOT)}")
            fixed_heading += 1
    
    print(f"\nFixed {fixed_code} files with collapsed code blocks")
    print(f"Fixed {fixed_heading} files with missing section headings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
