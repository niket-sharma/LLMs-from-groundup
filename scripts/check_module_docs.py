#!/usr/bin/env python3
"""Lightweight, dependency-free validation for module learning guides."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs" / "modules"
REQUIRED_HEADINGS = ("understand", "run", "validate", "test")
LINK_PATTERN = re.compile(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)")


def main() -> int:
    errors = []
    if not DOCS_ROOT.exists():
        print("docs-check: no module guides yet")
        return 0
    for guide in sorted(DOCS_ROOT.glob("*/index.md")):
        text = guide.read_text(encoding="utf-8").lower()
        missing = [heading for heading in REQUIRED_HEADINGS if heading not in text]
        if missing:
            errors.append(f"{guide.relative_to(ROOT)}: missing sections {', '.join(missing)}")
        original = guide.read_text(encoding="utf-8")
        for target in LINK_PATTERN.findall(original):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (guide.parent / target).resolve().exists():
                errors.append(f"{guide.relative_to(ROOT)}: broken local link {target}")
    if errors:
        print("docs-check failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"docs-check: validated {len(list(DOCS_ROOT.glob('*/index.md')))} module guide(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
