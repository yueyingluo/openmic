# OpenMic midterm presentation

Open `midterm.html` in a browser. The deck is self-contained, uses system fonts,
and needs no server, build step or network connection.

## Slide ownership

| Slide | Owner | Content |
|---|---|---|
| 1 | Team 1234 | Cover |
| 2 | lyy | Goal; defining humor, writing jokes and delivering them |
| 3 | lyy | Five-agent architecture, revision loop and TTS flow |
| 4–6 | znjq | Content generation: whole-story view, prompted generation, evaluation and revision |
| 7 | csw / jyb | TODO: speech generation |
| 8 | szq | TODO: evaluation and current progress |
| 9 | szq | TODO: limitations, next steps and closing |

The lyy section describes the local implementation and project goals. The
3–5 minute duration is a target, not a measured result. Slides 4–6 contain the
2.5-minute text-generation section. Speech generation and the evaluation/closing
sections remain placeholders until their owners supply content.

Slide 2 presents our working challenge analysis, not a universal theory of humor.
Audience ratings and matched text/audio comparisons are proposed validation
methods. Additional speaking notes are in the HTML comment above that slide.

## Present and edit

- Use arrow keys, Space, Page Up/Down or the on-screen arrows to change slides.
- Press E, or use the top-left edit button, to edit headings and placeholder text.
- Press Esc to leave editing mode. Text changes are cached in this browser only.
- Choose **Save HTML** or press Ctrl/Cmd+S to download the edited deck and share it.
- Use **Print / PDF** for a static copy; check the browser's print preview before saving.
- SVG diagram labels and layout are edited in the HTML source.

This is a local file, not a real-time shared editor. For parallel AI editing,
each owner should modify only their commented section in `midterm.html` and
submit the change for integration. Shared CSS and navigation belong to the
integrator. Keep all visible slide text in English and leave unknown results as TODO.

## Design and evidence

The canvas is fixed at 1920 × 1080 and scales uniformly. Use a white background,
system fonts, simple blue/gray diagrams and little text. Keep each page to one
main visual. Avoid adding unverified progress or results to placeholders.

The diagrams on slides 1–3 are explanatory graphics, not application screenshots.
Architecture references: `src/openmic/autogen_v02.py`,
`src/openmic/streamlit_app.py`, and `src/openmic/tts.py`.
The TTS action is user-triggered; the quality gate stops at approval or the
revision limit, and the UI displays the outcome in either case.

## Task 2 text-generation presentation

The shared [midterm.html](midterm.html) contains section **02 / znjq** on
**slides 4–6**, with three slides for **2 minutes 30 seconds**.
The section covers the whole-story view, prompted generation, and evaluation
with revision. The English script and timing follow below.

This directory tracks only `README.md` and `midterm.html`. Standalone decks
and private notes remain local and are excluded from Git.

Implementation and development evidence: [release summary](../docs/TASK2_RELEASE.md)
and [compact results](../docs/task2_v39_results.json).

## English speaking notes

| Team slide | Topic | Time | Elapsed |
|---|---|---:|---:|
| 4 | The whole-story view | 40 sec | 0:40 |
| 5 | Prompting and generation | 55 sec | 1:35 |
| 6 | Evaluation and revision | 55 sec | 2:30 |

### Slide 4: Humor in the whole story

I work on text generation. I treat a joke as a complete experience: what someone wants, what they do, and how the result changes our expectations.

During optimization, polishing every setup and punchline separately can make the whole story feel artificial. So I check the overall motivation and rhythm. Writing can start from either a funny moment or an experience.

### Slide 5: CFunSet-guided generation

We combine CFunSet retrieval with prompt engineering. Given a topic, audience, and real context, we retrieve examples from the training split and identify their core expectation and reveal.

DeepSeek Flash then handles five roles: direction, audience analysis, writing, delivery coaching, and quality control. The prompts give each role a clear job, while leaving the writer room to improve the plan.

We can adapt an existing joke or transfer its mechanism into a new situation. This uses examples in prompts without updating DeepSeek's weights.

### Slide 6: Evaluation and revision

In the additional quality loop, DeepSeek V4 Pro scores the whole script from one to five on humor, cultural fit, spoken naturalness, and structure. It also checks realism and continuity.

Clear but weak humor stays around three. Four needs a specific, effective comic turn. The reviewer quotes problems and suggests repairs.

Pro then receives the original draft and feedback, rewrites it, and reviews the revision again. We retain the best candidate within a revision limit. These are model judgments, so human ratings are still necessary.

### Presenter reminders (not spoken)

- Aim for about 95–100 words per minute, with brief pauses. The 2:30 timing is a plan, not a recorded duration.
- Initial drafting uses API model ID `deepseek-flash`. Independent audit and rewriting use `deepseek-v4-pro` in separate calls. These prompts do not train or fine-tune DeepSeek.
- The ordinary five-role CLI does not automatically run the additional Pro loop.
- Independent audit uses 1–5 and additionally reports `realism`. Internal five-role QC uses 0–10. CFunSet binary labels train a separate local classifier, whose probability is only a weak auxiliary signal.
- CFunSet contains multiple tasks. Generation uses references from its training split. Adaptation preserves the source mechanism and necessary context. Short jokes need no forced expansion or callback quota.
- Sources: `src/openmic/autogen_v02.py`, `humor_generation.py`, `humor_data.py`, `humor_craft.py`, `humor_quality_loop.py` and `humor_scorer.py`.
