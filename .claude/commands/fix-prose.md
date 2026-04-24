---
description: Run jargon-fixer and voice-fixer in parallel against the target file (combines /fix-jargon and /fix-voice)
argument-hint: [file-path]
---

Run **both** the `jargon-fixer` and the `voice-fixer` agents against the same target.

**Target resolution:**
- If `$ARGUMENTS` is non-empty, treat it as the file path to operate on.
- If `$ARGUMENTS` is empty, both agents will fall back to `git diff --name-only HEAD` and operate on every changed `.html` or `.md` file.

**Execution order — important.**
Run the agents **sequentially**, not in parallel. Each agent uses the Edit tool in place; running them in parallel would cause edit conflicts.

1. First: invoke `jargon-fixer` (subagent_type: "jargon-fixer") with the target. Wait for its report.
2. Then: invoke `voice-fixer` (subagent_type: "voice-fixer") with the same target. Wait for its report.
3. Surface both reports to me, clearly labeled.
4. Show me `git diff --stat` for the changed files.
5. Wait for me to review the diff before suggesting any commit.

Do **not** do the editing yourself — the agents handle edits in place.

Target: $ARGUMENTS
