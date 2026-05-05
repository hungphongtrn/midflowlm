# Engineering Workflow

Use this workflow to choose the right skill under `skills/engineering/`. This file is a router, not a replacement for each skill's `SKILL.md`; once a skill is selected, follow that skill's detailed procedure.

## Default Loop

1. Understand the request and state any important assumptions.
2. Inspect the relevant code, docs, domain glossary, and ADRs before deciding.
3. Choose the smallest workflow that can reliably complete the task.
4. Define success criteria and a verification command or feedback loop.
5. Implement, diagnose, plan, or route to issues using the selected skill.
6. Verify the result with tests, reproduction steps, or review.
7. Request review for substantial or risky work.
8. Finish the branch deliberately when implementation is complete.

## Skill Router

| Situation | Use |
| --- | --- |
| First time using engineering skills in a repo, or skill context is missing | `setup-project-skills` |
| Unfamiliar code area needs a higher-level map | `zoom-out` |
| Plan, terminology, or domain model needs stress-testing | `grill-with-docs` |
| Current conversation should become a PRD | `to-prd` |
| PRD, plan, or spec should become implementation issues | `to-issues` |
| Incoming issues need labeling, clarification, or agent briefs | `triage` |
| Multi-step implementation needs a durable plan | `writing-plans` |
| Feature or bug fix should be built behavior-first | `tdd` |
| Bug, failing test, flaky behavior, or performance regression needs root cause analysis | `diagnose` |
| Architecture, testability, or module depth needs improvement | `improve-codebase-architecture` |
| A concrete plan has mostly independent tasks to execute in this session | `subagent-driven-development` |
| Substantial work needs an independent review | `requesting-code-review` |
| Work is complete and the branch needs merge, PR, keep, or discard handling | `finishing-a-development-branch` |

## Decision Rules

- Prefer direct implementation for small, well-scoped edits; do not force a heavyweight skill onto trivial work.
- Use `zoom-out` before touching code when the relevant area is unfamiliar.
- Use `grill-with-docs` before planning when language, domain concepts, or design tradeoffs are unclear.
- Use `writing-plans` only when sequencing matters or the task is too large to hold safely in a short plan.
- Use `subagent-driven-development` only after there is a concrete implementation plan with tasks that can be reviewed independently.
- Use `diagnose` for hard bugs; do not guess at fixes until there is a credible feedback loop or reproduction path.
- Use `improve-codebase-architecture` after evidence of architectural friction, such as missing test seams, shallow modules, or repeated cross-module coupling.
- Use `requesting-code-review` before trusting substantial, risky, or merge-bound changes.
- Use `finishing-a-development-branch` only after implementation is complete and tests have been verified.

## Common Paths

### Small Code Change

1. Inspect the relevant code.
2. Implement the smallest correct change.
3. Run focused verification.
4. Request review only if the change is substantial or risky.

### Feature From Rough Idea

1. Use `grill-with-docs` if concepts or tradeoffs are unclear.
2. Use `to-prd` if the idea needs product-level capture.
3. Use `to-issues` to create vertical tracer-bullet slices.
4. For each issue, use `writing-plans` to create a progressive-disclosure plan: detail the current phase, stub future phases, and update the plan as implementation reveals reality. Progressive disclosure means revealing detail only when needed—write detailed tasks for the current phase, keep later phases stubbed, then expand them as you learn.
5. Use `subagent-driven-development` to handle the planned issues, with `tdd` as the implementation discipline for each behavior-focused task.
6. Use `requesting-code-review` and then `finishing-a-development-branch`.

### Bug Or Regression

1. Use `diagnose` to build a feedback loop and reproduce the problem.
2. Rank hypotheses before instrumenting.
3. Add a regression test at the correct seam when possible.
4. Fix the bug and rerun the original feedback loop.
5. If no good test seam exists, consider `improve-codebase-architecture` after the fix.

### Architecture Improvement

1. Use `zoom-out` to map the relevant modules and callers.
2. Use `improve-codebase-architecture` to surface deepening opportunities.
3. Let the user choose which candidate to explore.
4. Use `grill-with-docs` for naming, domain language, and ADR-worthy decisions.
5. Use `writing-plans` or `tdd` for implementation.

## Issue Workflow

- Use `triage` for incoming bugs and enhancements.
- Use `to-prd` when the current conversation should become a product document.
- Use `to-issues` to split approved work into independently grabbable vertical slices.
- Issues should describe end-to-end behavior and acceptance criteria, not isolated implementation layers.
- For each ready issue, use `writing-plans` in a progressive-disclosure manner before implementation.
- Use `subagent-driven-development` to execute each issue plan, following `tdd` for behavior-focused changes.

## Review And Finish

- Review early for substantial changes and before merge-bound work.
- Fix Critical review findings immediately.
- Fix Important review findings before continuing unless there is a documented reason not to.
- Treat Minor findings as optional follow-up unless they affect correctness or maintainability.
- Before finishing a branch, verify tests, then use `finishing-a-development-branch` to choose merge, PR, keep, or discard.
