---
description: Strip self-referential project shorthand and define industry acronyms on first use in user-facing HTML or markdown
argument-hint: [file-path]
---

Run the `jargon-fixer` agent against the target file.

**Target resolution:**
- If `$ARGUMENTS` is non-empty, treat it as the file path to operate on.
- If `$ARGUMENTS` is empty, the agent will fall back to `git diff --name-only HEAD` and operate on every changed `.html` or `.md` file.

Invoke the agent with the `Agent` tool, `subagent_type: "jargon-fixer"`, and pass the target as part of the prompt. Wait for the agent's report and surface its one-line summary back to me. Do **not** do the editing yourself — the agent handles edits in place.

After the agent finishes:
1. Show me `git diff --stat` for the changed files.
2. Wait for me to review the diff before suggesting any commit.

Target: $ARGUMENTS
