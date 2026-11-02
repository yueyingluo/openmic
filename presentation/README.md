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
| 7 | csw | TTS reliability and validation |
| 8 | jyb | Executable delivery control and tagged deadline excerpts |
| 9 | szq | Evaluation: rank correlations with human ratings |
| 10 | szq | Limitations and proposed audience evaluation |

The lyy section describes the local implementation and project goals. The
3–5 minute duration is a target, not a measured result. Slides 4–6 contain the
2.5-minute text-generation section. CSW's page documents adapter reliability
checks and mocked verification; it does not claim speech quality improvements.
JYB's page presents delivery controls and same-script comparisons. The evaluation
section reports measured results and the missing audience validation.

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

## Evaluation section

Slides 9–10 use the frozen results in [V3_REPORT.md](../evaluation/V3_REPORT.md).
All evaluation results are presented as Spearman rank correlations with human
ratings. On the same 407 CMHL confirmation examples, the continuous RoBERTa +
CORN score reaches 0.366 (95% CI 0.265–0.443), versus 0.292 for the lightweight
baseline. A constant majority prediction has undefined correlation and is not
plotted. The chart spans −1 to +1. The external CCL relative-humor diagnostic
reaches 0.032 on 800 items; this is a different task, not a paired decline estimate.
The two speech results are correlations on public short-clip human labels;
they are not full-performance humor validation or official model results.
English speaking notes and source links are in the comments above each slide.

The external CCL diagnostic uses the public relative-rating data from
[Chinese Humor Generation](https://github.com/DHfusion/chinese_humor_generation),
with [the paper](https://aclanthology.org/2025.ccl-1.3/) describing that task.
The published workbook SHA-256 is
`59bd49461dc4427240f4721fcc0d2363152dd417c64ed1b272bfec4daaacda82`.
The primary diagnostic covers 800 generated versions of 200 jokes; its
200-joke bootstrap interval is −0.0453 to 0.1069. This does not establish
reliable agreement with human relative humor judgments.

The text-generation slides and shared navigation are unchanged. The two
evaluation slides keep their English speaking notes in the HTML comments.
No scripts, fonts or data are loaded over the network.

## JYB delivery section and local demo

Page 8 presents the delivery pipeline, four same-script conditions, a delivery
control box, and two tagged excerpts from the deadline script. CSW's adapter
reliability page precedes it, and SZQ's evaluation pages follow it. All ten
slides are direct children of the deck stage; no speech-generation TODO
pages remain.

The two deadline excerpts retain their original Chinese words and prompt tags,
as requested for this example; the surrounding slide text remains English.

Speaking notes: [English](../docs/JYB_MIDTERM_SCRIPT_EN.md) and
[Chinese](../docs/JYB_MIDTERM_SCRIPT_ZH.md).

The player is separate from the slide: [three-version player](../docs/jyb_audio_compare.html).
It compares the same frozen canteen, deadline or fitness script using the
Nov 1 run's baseline and performance, then the integrity run's safe_club.
Raw audio and whole-file gain-matched audio are separate selections.
Only the canteen example has additional source-interval excerpts, in both raw
and matched forms.

Generated audio is excluded from Git under CONTRIBUTING.md. The local delivery
bundle has a ready-to-open player and all 24 WAV files. To play them from this
repository, copy the contents of its `local_demo/audio/` into
`docs/media/jyb_compare/`, then click **加载已复制音频**. Alternatively,
select that local audio folder in the player; matching file paths and SHA256
are checked locally. No API or upload occurs. The external JSON manifest is
an inspectable copy; the page embeds it, so file:// playback needs no fetch.

The shopping script is withheld from the new complete-audio demo because one
sentence still omits two particles. The historical 71-test result is not a
fresh test count for this slide revision. No listener MOS or causal humor gain
is claimed. See [results and limits](../docs/JYB_DELIVERY_RESULTS.md).
