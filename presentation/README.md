# OpenMic midterm presentation

Open `midterm.html` in a browser. The deck uses system fonts and needs no server, build step or network
connection for local playback. JYB's three classroom audio excerpts are
included in the repository under `assets/jyb_midterm_audio/`.

## Slide ownership

| Slide | Owner | Content |
|---|---|---|
| 1 | Team 1234 | Cover |
| 2 | lyy | Goal; defining humor, writing jokes and delivering them |
| 3 | lyy | Five-agent architecture, revision loop and TTS flow |
| 4–6 | znjq | Content generation: whole-story view, prompted generation, evaluation and revision |
| 7 | csw | TTS reliability and validation |
| 8–9 | jyb | Executable delivery cues; stage annotations and sentence integrity |
| 10 | szq | Evaluation: rank correlations with human ratings |
| 11 | szq | Limitations and proposed audience evaluation |

The lyy section describes the local implementation and project goals. The
3–5 minute duration is a target, not a measured result. Slides 4–6 contain the
2.5-minute text-generation section. CSW's page documents adapter reliability
checks and mocked verification; it does not claim speech quality improvements.
JYB's two pages cover executable delivery cues and sentence integrity. The evaluation
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

Slides 10–11 use the frozen results in [V3_REPORT.md](../evaluation/V3_REPORT.md).
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

## JYB delivery slides, audio and final scripts

Slides **8–9** are JYB's two pages. Slide 8 covers executable delivery cues,
the four original same-script conditions, and two tagged deadline examples.
`Speech` keeps a spoken segment and its controls; `Pause` is local silence.
Slide 9 shows stage annotations, deterministic sentence splitting with source
offsets, and assembly after every sentence candidate has been accepted.
It lists the eight delivery intents and explains the boundary and endpoint
problems that led to the integrity layer. These are delivery and assembly
changes, not TTS model training.

The two approved slide layouts and their visible text are unchanged in this
revision. Only the media locations and missing-file message have changed.
The stale JYB and joint CSW/JYB TODO blocks remain removed: one had a missing
closing tag that nested completed pages inside a hidden placeholder.
The nine completed non-JYB pages, shared CSS and existing navigation retain
their substantive content. The full deck has eleven pages.

### Open and play

**Clone or download the complete repository**, extract the ZIP if needed,
then open `presentation/midterm.html` in a browser. Go to slides 8–9.
Before this change is merged, use the `jyb` branch; after merging, use `main`.
The three audio files are now committed with the slides, so no separate media
copy, API key, TTS setup, server or network connection is needed for local
playback. Keep the repository's folder structure. The GitHub file browser
shows HTML source; it does not run the slide deck. Pushing this change does
not automatically publish a GitHub Pages website.

- Slide 8: **Original reading** and **First full performance**.
- Slide 9: **Compact · integrity repair**.
- Click Play/Pause to start or pause. Starting another clip stops and resets
  the previous one. Leaving its slide also stops and resets playback.
- Arrow and Page Up/Down navigation works while a playback button has focus.
- Save HTML exports and PDFs do not embed audio. If sharing an edited HTML,
  keep it in `presentation/` alongside the unchanged `../assets/` directory.

| Button | Tracked path from the repository root | Duration |
|---|---|---:|
| Original reading | `assets/jyb_midterm_audio/deadline_01_baseline.wav` | 14.456375 s |
| First full performance | `assets/jyb_midterm_audio/deadline_02_first_performance.wav` | 20.1016875 s |
| Compact · integrity repair | `assets/jyb_midterm_audio/deadline_03_integrity_repair.wav` | 17.3570625 s |

These are the same previously prepared, whole-file gain-matched deadline
excerpts. They contain the same words and end after **“不是灵感，是情绪。”**
They were copied byte for byte: no new synthesis, cropping, gain change or
speed adjustment was performed for this revision. The three WAVs total
1,661,416 bytes. Their different durations reflect the existing delivery.

The three named excerpts are an explicit classroom-demo exception documented
in `CONTRIBUTING.md`. `.gitignore` allows only these three WAV paths; other
generated media retain the existing ignore rules. No `git add -f` is needed.
The tracked `presentation/` directory still contains exactly `midterm.html`
and `README.md`; audio is under `assets/`, and speaking scripts are under `docs/`.

### Final speaking script

- [Final English script](../docs/JYB_MIDTERM_FINAL_SCRIPT_EN.txt)
- [Original final Chinese script](../docs/JYB_MIDTERM_FINAL_SCRIPT_ZH.txt)

The Chinese script preserves the presenter's final wording; only the
repository copy's line endings are normalized to LF for Git checks.
The English version follows its order and three playback cues, using short
sentences. It describes candidate checks without claiming perfect speech.
These final scripts supersede the earlier JYB speaking notes for this talk.

The older [three-version comparison player](../docs/jyb_audio_compare.html)
still uses its separate local `docs/media/jyb_compare/` files. The three
tracked classroom excerpts do not supply that player's full recordings.
See [results and limits](../docs/JYB_DELIVERY_RESULTS.md) for the existing
experiment limitations. No listener MOS or causal humor gain is claimed.
