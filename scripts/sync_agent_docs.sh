#!/usr/bin/env bash
# Mirror AGENTS.md (the canonical agent instruction file) into the file each
# coding harness auto-loads, so every agent reads identical rules.
#
# Usage: scripts/sync_agent_docs.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE="AGENTS.md"
HEADER="<!-- AUTO-GENERATED from AGENTS.md by scripts/sync_agent_docs.sh — edit AGENTS.md instead. -->"

targets=(
  "CLAUDE.md"                          # Claude Code
  "GEMINI.md"                          # Gemini CLI
  "CONVENTIONS.md"                     # Aider (via --read)
  ".github/copilot-instructions.md"    # GitHub Copilot
  ".cursor/rules/agents.mdc"           # Cursor
)

for target in "${targets[@]}"; do
  mkdir -p "$(dirname "$target")"
  if [[ "$target" == *.mdc ]]; then
    # Cursor rules need frontmatter to always apply
    {
      printf -- '---\ndescription: Repo-wide agent rules (synced from AGENTS.md)\nalwaysApply: true\n---\n\n%s\n\n' "$HEADER"
      cat "$SOURCE"
    } > "$target"
  else
    {
      printf '%s\n\n' "$HEADER"
      cat "$SOURCE"
    } > "$target"
  fi
  echo "synced $SOURCE -> $target"
done
