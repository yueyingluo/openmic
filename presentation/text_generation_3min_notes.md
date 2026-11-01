# Text generation: English script for 2 minutes 30 seconds

Integrated deck: [midterm.html](midterm.html), **slides 4–6**, section **02 / znjq**.
The [standalone three-slide version](text_generation_3min.html) contains the same content.
The existing filenames remain unchanged so earlier links continue to work.

| Team slide | Topic | Time | Elapsed |
|---|---|---:|---:|
| 4 | The whole-story view | 40 sec | 0:40 |
| 5 | Prompting and generation | 55 sec | 1:35 |
| 6 | Evaluation and revision | 55 sec | 2:30 |

## Slide 4: Humor in the whole story

I work on text generation. I treat a joke as a complete experience: what someone wants, what they do, and how the result changes our expectations.

During optimization, polishing every setup and punchline separately can make the whole story feel artificial. So I check the overall motivation and rhythm. Writing can start from either a funny moment or an experience.

## Slide 5: CFunSet-guided generation

We combine CFunSet retrieval with prompt engineering. Given a topic, audience, and real context, we retrieve examples from the training split and identify their core expectation and reveal.

DeepSeek Flash then handles five roles: direction, audience analysis, writing, delivery coaching, and quality control. The prompts give each role a clear job, while leaving the writer room to improve the plan.

We can adapt an existing joke or transfer its mechanism into a new situation. This uses examples in prompts without updating DeepSeek's weights.

## Slide 6: Evaluation and revision

In the additional quality loop, DeepSeek V4 Pro scores the whole script from one to five on humor, cultural fit, spoken naturalness, and structure. It also checks realism and continuity.

Clear but weak humor stays around three. Four needs a specific, effective comic turn. The reviewer quotes problems and suggests repairs.

Pro then receives the original draft and feedback, rewrites it, and reviews the revision again. We retain the best candidate within a revision limit. These are model judgments, so human ratings are still necessary.

## Presenter reminders (not spoken)

- Aim for about 95–100 words per minute, with brief pauses. The 2:30 timing is a plan, not a recorded duration.
- Initial drafting uses API model ID `deepseek-flash`. Independent audit and rewriting use `deepseek-v4-pro` in separate calls. These prompts do not train or fine-tune DeepSeek.
- The ordinary five-role CLI does not automatically run the additional Pro loop.
- Independent audit uses 1–5 and additionally reports `realism`. Internal five-role QC uses 0–10. CFunSet binary labels train a separate local classifier, whose probability is only a weak auxiliary signal.
- CFunSet contains multiple tasks. Generation uses references from its training split. Adaptation preserves the source mechanism and necessary context. Short jokes need no forced expansion or callback quota.
- Sources: `src/openmic/autogen_v02.py`, `humor_generation.py`, `humor_data.py`, `humor_craft.py`, `humor_quality_loop.py` and `humor_scorer.py`.
