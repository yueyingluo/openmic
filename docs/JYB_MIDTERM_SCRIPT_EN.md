# JYB: English speaking notes

Current version: **one slide**, page **8** in the team deck, or page **1** in the standalone preview. Allow roughly **1–2 minutes** for this script; this is a timing plan, not a recorded duration.

## Making delivery cues executable

My work builds on Chen Shouwen's basic CosyVoice2 request and WAV validation. I added a layer that turns the coach's delivery cues into executable instructions.

The four boxes show the process. We start with a tagged script. A strict compiler converts it into Speech and Pause events and rejects unknown tags. The renderer then sends expression hints to the speech model and inserts silence between the returned clips. Delivery control uses a fixed planning scheme for intent and focus, so a contrast or reveal can receive different emphasis from its setup.

The table separates four conditions on the same words. Baseline uses neutral speech. Timing adds the planned pauses. Emotion changes the emotion prompt. Full performance combines emotion, speed, emphasis and timing. Emotion instructions and native emphasis remain soft controls.

At the bottom are two selected excerpts from our deadline script. The setup says the assignment allowed three weeks, but the speaker did not open the document for the first two. The punchline is that they were waiting for their anxiety to be ready. The tags mark the setup, the punchline, a pause, a self-deprecating tone and the phrase to emphasize. The program reads these as controls rather than spoken words.

## Source excerpts shown on the slide

```text
[SETUP] [EMOTION=self_deprecating] 期末作业给了三周时间。但我前两周一直没打开那个文档。
[PAUSE=0.8] [PUNCHLINE] [EMOTION=self_deprecating] 我要等[EMPHASIS]焦虑到位[/EMPHASIS]。
```

These are selected excerpts from the recorded deadline script, not a claim that the two lines are adjacent or that they reproduce the entire synthesis input. The slide retains the original Chinese wording.

## Presenter notes, not spoken

- The former second JYB slide has been removed. The shared deck now has ten pages; the original CSW/JYB placeholder remains on page 7, and JYB occupies page 8.
- The delivery-control box summarizes the stage-planning work in the prepared update. The slide does not claim that every displayed feature was already merged into the supplied main folder.
- The three-version player remains available separately at `docs/jyb_audio_compare.html`; it is no longer a page or file inside `presentation/`.
- Source and implementation details remain in `docs/JYB_DELIVERY_UPDATE.md`. Historical test and audio results remain in `docs/JYB_DELIVERY_RESULTS.md` and are not presented as fresh measurements from this slide-only revision.
