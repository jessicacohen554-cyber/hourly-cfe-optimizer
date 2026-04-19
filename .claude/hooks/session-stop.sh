#!/bin/bash
# Proactive Session Management (N4) — surface uncommitted work at session stop.
# See CLAUDE.md "Proactive Session Management" section for the full rule.

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

if [ -z "$(git status --porcelain 2>/dev/null)" ]; then
  exit 0
fi

{
  echo "Uncommitted work detected on branch $(git branch --show-current):"
  git status --short
  echo ""
  echo "Per CLAUDE.md Proactive Session Management:"
  echo "  1. Stop starting new work."
  echo "  2. Commit + push WIP to the current branch."
  echo "  3. Update SPEC.md ## Current Status (accomplishments, in-progress state, next steps, open questions)."
  echo "  4. Emit a resume prompt focused on task context (no branch name)."
} >&2

exit 1
