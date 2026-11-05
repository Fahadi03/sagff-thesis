# AI-use disclosure (Elsevier structure)

Target journals: Knowledge-Based Systems (0950-7051), Expert Systems with
Applications (0957-4174), Information Processing & Management (0306-4573), all
Elsevier. Elsevier's policy requires two separate disclosures in two places:
(1) AI that forms part of the research is described in the **Methods** section,
and (2) a formal **declaration** sits at the end of the manuscript, immediately
before the references.

**Check at submission time.** The heading, placement and required content below
were read from Elsevier's policy page ("Generative AI policies for journals",
https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals)
on 2026-10-26. A journal's Guide for Authors can add its own requirements, and
Elsevier can revise the policy, so re-read both the policy page and the specific
journal's Guide for Authors on Elsevier's site before submitting. (The Knowledge-
Based Systems Guide for Authors could not be retrieved automatically on
2026-10-26; check it in a browser.)

What the policy page states, as read on 2026-10-26:
- AI forming part of the research design or methods: "the use of the tools must
  be described in a reproducible manner in the Methods section".
- The declaration: a separate section "at the end of your manuscript, immediately
  before the references", headed **"Declaration of generative AI and AI-assisted
  technologies in the manuscript preparation process"**; it names the tool and the
  purpose of its use and confirms that the authors reviewed the output and take
  full responsibility for the content.
- Figures: AI may be used for explanatory images and for data visualisations
  derived from underlying data, but not to create or alter primary research images.
- AI-generated references "can be incorrect, hallucinated, or fabricated"
  (see `docs/citation_checklist.md`).


---

## 1a. Methods section (§3 subsection)

> **3.x Use of AI-assisted tools.** AI-assisted tools were used in this research.
> Claude Code (Anthropic) implemented the experimental pipeline, its test suite and
> the project documentation. Claude (Anthropic) proposed elements of the
> experimental design and the control conditions; the author evaluated each
> proposal and accepted, modified or rejected it. All decisions are recorded with
> their justifications in the project protocol, the analysis plan was registered
> before any grid run, no reported number was entered by hand, and an
> automated test suite gates the experiments. All figures were generated
> programmatically from the run records by `src/analysis.py`; no generative AI was
> used to create or alter any figure.

## 1b. Formal declaration (end of manuscript, immediately before the references)

Heading as stated on Elsevier's policy page on 2026-10-26; re-check it, and the
journal's own Guide for Authors, at submission:

> **Declaration of generative AI and AI-assisted technologies in the manuscript preparation process**
>
> During the preparation of this work the authors used Claude and Claude Code
> (Anthropic) to implement the experimental pipeline and to draft portions of the
> manuscript. The authors reviewed and edited the content and take full
> responsibility for the content of the published article.

## Acknowledgements (optional; the university has no AI-disclosure requirement)

> The experimental software for this thesis was implemented with Claude Code
> (Anthropic), and Claude (Anthropic) was used as a methodological advisor during
> the design of the experiments. All decisions, analyses and interpretations are
> the author's, and the full research protocol and decision record are available in
> the project repository.
