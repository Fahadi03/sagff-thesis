# Translation instruction (Spanish and Hindi extension, PROTOCOL §12)

The Spanish (`es_text`) and Hindi (`hi_text`) fields of the extension were written
by Claude (Anthropic, model Opus 5.5) in a Claude Code session in VS Code on
another PC on 2026-11-10 (docs/PROTOCOL.md §12, amendment 2026-11-10). The user's
instruction to that session, verbatim, with its original spelling:

```
here are all file of data already have english and bangla.. now i want to added aso hindi and spnais fully do it in all file
```

Only the English `text` field was used as the source. No translation library,
model or external API was used. The intermediate per-file text files and the merge
script were not retained. The reference translations are `data/extension/es.jsonl`
and `data/extension/hi.jsonl`, identified by the SHA-256 hashes recorded in
docs/PROTOCOL.md §12. They are gitignored and not committed.
