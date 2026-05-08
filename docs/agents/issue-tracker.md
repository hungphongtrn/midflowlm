# Issue tracker: GitHub

GitHub Issues are the durable project memory for this repo. PRDs, issue decomposition, triage, implementation plans, status updates, review outcomes, and closure rationale live in GitHub issues and comments.

Use the `gh` CLI for all issue operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v` — `gh` does this automatically when run inside a clone.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

If the issue captures uncertain intent or unresolved scope, publish it anyway and make uncertainty explicit with `Open Questions`, `Assumptions`, or `Needs Triage` sections.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Before implementation starts

The source issue must be a `ready-for-agent` leaf issue. During implementation, comment with the committed plan path, for example:

```markdown
## Implementation Plan

Plan committed at:

- `docs/plans/YYYY-MM-DD-issue-<number>-<slug>/README.md`
```

## Before finishing work

Update the source issue with:

- implementation summary
- verification performed
- PR link or commit reference
- follow-up issues created, if any
- whether acceptance criteria are complete
