# Digital Religions @ UZH — 25-minute talk

LaTeX Beamer deck for the Digital Religions conference, based on the MA
thesis *Transformer-Based Cross-Lingual Approaches to Religious Hate
Speech*.

## Files

- `slides.tex` — the Beamer source (21 slides, ~25 min).
- `images/` — figures copied from the thesis (`RQ1`, `RQ2`, `RQ3`) plus
  the BERTopic datamap.

## Build

```
xelatex slides.tex
xelatex slides.tex
```

Requires `xeCJK` (already in your thesis toolchain) so that the Chinese
/ Japanese anchor words render. The preamble loads xeCJK only when
`ctexhook.sty` is present, so the file still compiles on machines
without CJK support (CJK chars simply fall back to the default font).

## Slide map vs. timing

| # | Slide | Minutes |
|---|-------|---------|
| 1  | Title                                    | 1 |
| 2  | Why this talk belongs at Digital Religions | 2 |
| 3  | Research gap                             | 2 |
| 4  | Why anti-Christian (comparative probe)   | 1 |
| 5  | Who–how–why research questions           | 2 |
| 6  | Trilingual corpus (data)                 | 2 |
| 7–8| Pipeline + one technical slide           | 3 |
| 9  | BERTopic datamap                         | 1 |
| 10 | *Finding 1 — target chooses the frame*   | — |
| 11 | RQ1 who is targeted                      | 2 |
| 12 | RQ2 ten frames + by-language bars        | 2 |
| 13 | Three national frame signatures          | 1 |
| 14 | Target–frame rule (strongest slide)      | 2 |
| 15 | *Finding 2 — two moral architectures*    | — |
| 16 | How to read bias scores                  | 1 |
| 17 | Three moral fingerprints (line chart)    | 2 |
| 18 | Headline one-liner                       | 1 |
| 19 | Beyond anti-Christian speech             | 1 |
| 20 | Limits + discussion questions            | 2 |
| 21 | Thank you                                | — |
