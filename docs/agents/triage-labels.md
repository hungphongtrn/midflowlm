# Triage Labels

The skills speak in terms of two category roles and five state roles. This file maps those roles to the actual label strings used in this repo's GitHub Issues.

Every triaged issue should carry exactly one category role and one state role.

## Category Roles

| Canonical label | Label in GitHub | Meaning |
| --- | --- | --- |
| `bug` | `bug` | Something is broken |
| `enhancement` | `enhancement` | New feature, research idea, or improvement |

## State Roles

| Canonical label | Label in GitHub | Meaning |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | Maintainer or triage agent needs to evaluate this issue |
| `needs-info` | `needs-info` | Waiting on reporter or maintainer for more information |
| `ready-for-agent` | `ready-for-agent` | Fully specified leaf issue, ready for an AFK agent |
| `ready-for-human` | `ready-for-human` | Valid work that needs human judgment, access, or manual action before agent execution is safe |
| `wontfix` | `wontfix` | Will not be actioned |

When a skill mentions a role, use the corresponding GitHub label string from this table.

Edit the right-hand column to match repo-specific label names only if the maintainer intentionally uses different GitHub labels.
