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

## Bangla translation-method control (track `bnintent30-bn-cl`)

The Bangla text of the control track was written by Claude (Anthropic, model
Fable 5.2, model ID `claude-fable-5-2`) in a Claude Code session of the Claude
desktop app on the repository's PC, on 2026-11-12 and 2026-11-13 (docs/PROTOCOL.md
§12, amendment 2026-11-12 and note 2026-11-13). The parts of the user's instruction
to that session that define the translation, verbatim:

```
- Track bnintent30-bn-cl: Bangla written by Claude (record the exact model of this session) from the English text only, for the 4,096 clean uids, 150-sentence batches per intent, same procedure as es/hi. The translator must not read the dataset's existing bn_text.
```

```
Step B: translate. Read only uid, English text and intent. Write data/extension/bn_cl.jsonl (gitignored). Never modify data/raw, data/clean or data/splits. Record its SHA-256 in a further dated note. Keep the intermediate batch files under data/extension/provenance/ (gitignored).
```

Only the English text was used as the source. No translation library, model or
external API was used. The batch files are kept under
`data/extension/provenance/bn_cl/` (gitignored). The reference translation is
`data/extension/bn_cl.jsonl`, identified by the SHA-256 recorded in
docs/PROTOCOL.md §12. It is gitignored and not committed.
