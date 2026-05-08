# Engineering Workflow

Use this workflow to choose the right skill under `skills/engineering/`. This file is the router and operating model; each skill's `SKILL.md` remains the detailed procedure.

## Operating Principle

Coding agents are executioners. GitHub Issues are the durable project memory.

Use GitHub Issues to preserve intent, rationale, acceptance criteria, decomposition, implementation status, review outcomes, and closure. Do not rely on transient chat history for work that future agents or maintainers may need to understand.

## Bootstrap Rule

Run `setup-project-skills` once in every greenfield or brownfield repo before using the engineering workflow. It verifies GitHub issue access, required labels, workflow instructions, `WORKFLOW.md`, and repo-local agent docs.

If setup has not been completed, stop and run `setup-project-skills` first.

## Default Loop

1. Understand the request and state important assumptions.
2. Inspect relevant issues, code, docs, domain glossary, and ADRs before deciding.
3. Choose the smallest workflow that preserves necessary rationale.
4. Define success criteria and a verification command or feedback loop.
5. Route discussion, PRD, issue, plan, implementation, diagnosis, review, or finish through the selected skill.
6. Verify the result with tests, reproduction steps, review, or issue acceptance criteria.
7. Update GitHub Issues with decisions, plans, implementation status, verification, and follow-ups.
8. Finish the branch deliberately when implementation is complete.

## Skill Router

| Situation | Use |
| --- | --- |
| First time using engineering skills in a repo, or skill context is missing | `setup-project-skills` |
| Substantive discussion may affect behavior, domain language, architecture, scope, or future implementation | `grill-with-docs` |
| Current conversation should become an intent-level PRD | `to-prd` |
| PRD or issue needs decomposition into smaller GitHub issues | `grill-with-docs`, then `to-issues` |
| User asks what to work on next, or an issue needs labeling, clarification, or briefs | `triage` |
| User asks to execute issue `#N` | issue execution workflow |
| Implementation work needs an isolated workspace | `using-git-worktrees` |
| Ready issue needs a committed progressive plan | `writing-plans` |
| Feature, bug fix, integration change, or behavior-preserving refactor should be built behavior-first | `tdd` |
| Bug, failing test, flaky behavior, or performance regression needs root cause analysis | `diagnose` |
| Unfamiliar code area needs a higher-level map | `zoom-out` |
| Architecture, testability, or module depth needs improvement | `improve-codebase-architecture` |
| Concrete plan has mostly independent tasks to execute in this session | `subagent-driven-development` |
| Substantial work needs independent review | `requesting-code-review` |
| Work is complete and the branch, PR, and source issue need finish handling | `finishing-a-development-branch` |

## Issue Memory Rule

GitHub Issues are required for decision-bearing product, research, architecture, bug, and implementation work. Default to creating or updating an issue before implementation.

A direct code change may skip issue creation only when all are true:

- The user's request is already precise.
- No product, domain, architecture, interface, or scope decision is being made.
- The change is low-risk and locally verifiable.
- The rationale fits in the final response, commit message, or PR summary.

If future maintainers or agents would benefit from knowing why the change happened, create or update an issue instead.

### Bug Fixes

Bug fixes normally enter the issue workflow because reproduction, root cause, and verification are durable project memory.

Use `diagnose` before fixing any bug whose cause is not immediately obvious. A direct bug fix is allowed only for obvious mechanical breakages. Even then, record symptom, cause, fix, and verification in the final response, commit message, or PR summary.

## Issue-Centered Lifecycle

### Discussion To PRD

1. Use `grill-with-docs` for substantive discussion that may affect behavior, domain language, architecture, scope, or future implementation.
2. Document resolved domain terms and ADR-worthy decisions immediately.
3. If the discussion is tied to a GitHub issue or PRD, summarize resolved decisions and link any updated `CONTEXT.md` or ADR files in an issue comment.
4. Use `to-prd` to synthesize the current conversation into an intent-level PRD issue with `needs-triage`.

`to-prd` captures goals, motivation, desired outcomes, constraints, non-goals, and open questions. It should not pretend to be an implementation plan.

### PRD Or Issue Decomposition

1. Before decomposing a PRD or issue with `to-issues`, use `grill-with-docs` to stress-test scope, terminology, dependencies, acceptance criteria, and HITL/AFK boundaries unless the split is purely mechanical and already explicit.
2. Use `to-issues` recursively when a PRD or issue is still too large or ambiguous for one agent to execute safely.
3. Publish child issues with `needs-triage`.
4. Stop decomposing when each leaf issue is independently understandable, small enough for one branch/PR, has clear acceptance criteria, has no unresolved intent/domain/architecture/scope decisions, and can be planned with `writing-plans` without reopening the parent discussion.

### Triage Gate

Only `ready-for-agent` leaf issues enter planning and implementation. If an issue is not `ready-for-agent`, use `triage`, `grill-with-docs`, or `to-issues` first.

If the user asks what to work on next, use `triage` to inspect GitHub Issues. Present maintainer-attention issues first, then unblocked `ready-for-agent` issues for execution.

State meanings:

- `needs-triage` — triage agent must evaluate the issue.
- `needs-info` — reporter or maintainer must answer specific questions.
- `ready-for-human` — valid work requires human judgment, access, or manual action before agent execution is safe.
- `ready-for-agent` — leaf issue is AFK-safe and can proceed to planning and implementation.
- `wontfix` — issue will not be actioned; preserve rationale before closing.

If triage determines an issue is `ready-for-agent`, continue into planning/execution only when the user requested execution or explicitly says to proceed. Otherwise, stop after labeling/commenting and present the next recommended action.

### Issue Execution Shortcut

`Execute issue #N` means:

1. Fetch issue `#N` from GitHub.
2. Verify it is a `ready-for-agent` leaf issue.
3. If it is not `ready-for-agent`, stop before implementation and use `triage`.
4. Use `using-git-worktrees` to establish an isolated implementation workspace.
5. Use `writing-plans` to create `docs/plans/YYYY-MM-DD-issue-<number>-<slug>/`.
6. Commit the plan before code changes begin.
7. Comment the plan path/link on issue `#N`.
8. Execute the current phase with `subagent-driven-development`.
9. Use `tdd` for behavior-bearing tasks and focused verification for purely mechanical tasks.
10. Request code review.
11. Use `finishing-a-development-branch` to finish the branch, PR, and source issue.

Every `ready-for-agent` implementation issue must have a committed progressive plan before code changes begin. Direct-change escape hatches are exempt.

## Common Paths

### New Project Setup

1. Install the engineering skill pack.
2. Run `setup-project-skills` in the target repo.
3. Confirm GitHub issue access and required labels.
4. Let setup write or update workflow instructions and repo-local agent docs.

### Feature Or Research Idea

1. Use `grill-with-docs` to resolve important terminology, scope, and tradeoffs.
2. Use `to-prd` to create an intent-level PRD issue.
3. Use `grill-with-docs`, then `to-issues`, to recursively create small leaf issues.
4. Use `triage` to move leaf issues into `ready-for-agent`, `ready-for-human`, `needs-info`, or `wontfix`.
5. Execute only `ready-for-agent` leaf issues.

### Bug Or Regression

1. Create or update the GitHub issue unless this is an obvious mechanical breakage.
2. Use `diagnose` to build a feedback loop and reproduce the problem.
3. Rank hypotheses before instrumenting.
4. Add a regression test at the correct seam when possible.
5. Fix the bug and rerun the original feedback loop.
6. If no good test seam exists, consider `improve-codebase-architecture` after the fix.
7. Update the issue with symptom, cause, fix, and verification.

### Architecture Improvement

1. Use `zoom-out` to map the relevant modules and callers.
2. Use `improve-codebase-architecture` to surface deepening opportunities.
3. Let the user choose which candidate to explore.
4. Use `grill-with-docs` for naming, domain language, and ADR-worthy decisions.
5. Convert approved work into issues before implementation.

## Review And Finish

- Review early for substantial changes and before merge-bound work.
- Fix Critical review findings immediately.
- Fix Important review findings before continuing unless there is a documented reason not to.
- Treat Minor findings as optional follow-up unless they affect correctness or maintainability.
- Before finishing a branch, verify tests and update the source issue with implementation summary, verification performed, PR link or commit reference, follow-up issues created if any, and acceptance-criteria status.
- Close the issue only when acceptance criteria are complete and the work is merged, or represented by an open PR that GitHub can auto-close.
