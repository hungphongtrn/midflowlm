---
# midflowlm-nfgx
title: Reduce online-no-cache context length to 1024
status: completed
type: task
priority: normal
created_at: 2026-03-23T06:56:31Z
updated_at: 2026-03-23T06:57:27Z
---

Update the dedicated online-no-cache 2048 config to use seq_len 1024 after OOM.
- [x] Change seq_len from 2048 to 1024
- [x] Verify config value updated
- [x] Summarize rerun command

## Summary of Changes

- Updated `configs/v0_online_no_cache_2048.yaml` to use `seq_len: 1024`.
- Renamed the experiment and output/log directories inside that config to `*_1024_kl` so the run metadata matches the new context length.
