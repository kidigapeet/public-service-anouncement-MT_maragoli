/**
 * gen_deck.js — the project presentation.
 *
 * Palette is USIU-Africa's own, sampled from the official logo: navy #293D94
 * dominant, gold #FFCA08 as the single sharp accent. Navy carries the title and
 * closing slides; content slides are light. Gold is used sparingly enough that
 * it still means something when it appears.
 *
 * Motif: tinted cards with a numbered or lettered navy disc. Repeated on every
 * content slide so the deck reads as one object.
 *
 * Fonts are Calibri and Cambria — both ship with Office and render true to
 * width in LibreOffice, so the visual QA pass can be trusted on text fit.
 */
const pptxgen = require("pptxgenjs");
const fs = require("fs");

const NAVY = "293D94";
const NAVY_D = "1B2A63";
const GOLD = "FFCA08";
const INK = "16181D";
const INK2 = "4B5563";
const INK3 = "8A919E";
const TINT = "F1F3F9";      // navy at ~4%
const TINT_G = "FFF7DE";    // gold at ~8%
const LINE = "DFE3EC";
const WHITE = "FFFFFF";

const H1 = "Cambria";
const BODY = "Calibri";

const LOGO = "image/png;base64," + fs.readFileSync("usiu_logo.png").toString("base64");

// The project team. Every name the deck shows comes from here, so the title
// slide and the closing slide cannot drift apart.
const TEAM = [
  "Weldesenbet Zeray",
  "Samuel Abrha",
  "Hetal Kumbharana",
  "Halima Mohammed",
  "Peter Kidiga",
  "Mitchelle Moraa",
];
const SUPERVISOR = "Prof. Edward Ombui";
const TEAM_LINE = TEAM.join("  ·  ");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";              // 13.3 x 7.5 in — set BEFORE any slide
pres.author = TEAM.join(", ");
pres.title = "Fine-Tuning NMT for Kenyan Public Service Announcements";

const W = 13.3, MG = 0.7;
const CW = W - MG * 2;

/* ---------- helpers ---------------------------------------------------- */

function titleSlide(s, kicker, title) {
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: MG, y: 0.42, w: CW, h: 0.28, fontFace: BODY, fontSize: 12,
      color: NAVY, bold: true, charSpacing: 2.4, margin: 0,
    });
  }
  s.addText(title, {
    x: MG, y: 0.72, w: CW, h: 0.72, fontFace: H1, fontSize: 30, bold: true,
    color: INK, margin: 0, valign: "top",
  });
}

// The repeated motif: a tinted card with a navy disc holding a short label.
function card(s, { x, y, w, h, disc, head, body, tint = TINT, discColor = NAVY }) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: tint }, rectRadius: 0.09, line: { color: tint },
  });
  if (disc) {
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.26, y: y + 0.26, w: 0.44, h: 0.44, fill: { color: discColor },
      line: { color: discColor },
    });
    s.addText(disc, {
      x: x + 0.26, y: y + 0.26, w: 0.44, h: 0.44, align: "center",
      valign: "middle", fontFace: BODY, fontSize: 13, bold: true,
      color: discColor === GOLD ? INK : WHITE, margin: 0,
    });
  }
  const tx = disc ? x + 0.86 : x + 0.3;
  const tw = w - (disc ? 1.16 : 0.6);
  s.addText(head, {
    x: tx, y: y + 0.24, w: tw, h: 0.32, fontFace: BODY, fontSize: 15,
    bold: true, color: INK, margin: 0, valign: "middle",
  });
  if (body) {
    s.addText(body, {
      x: tx, y: y + 0.6, w: tw, h: h - 0.82, fontFace: BODY, fontSize: 12.5,
      color: INK2, margin: 0, valign: "top", lineSpacingMultiple: 1.12,
    });
  }
}

function stat(s, { x, y, w, value, unit, label, color = NAVY }) {
  s.addText(
    [{ text: value, options: { fontSize: 46, bold: true, color } },
     { text: unit ? " " + unit : "", options: { fontSize: 17, bold: true, color } }],
    { x, y, w, h: 0.72, fontFace: BODY, margin: 0, valign: "middle" });
  s.addText(label, {
    x, y: y + 0.68, w, h: 0.5, fontFace: BODY, fontSize: 11.5, color: INK2,
    margin: 0, valign: "top", lineSpacingMultiple: 1.05,
  });
}

function footer(s, n) {
  s.addText("Ekegusii NMT · USIU-Africa · 2026", {
    x: MG, y: 7.02, w: 6, h: 0.28, fontFace: BODY, fontSize: 9.5,
    color: INK3, margin: 0,
  });
  s.addText(String(n), {
    x: W - MG - 0.6, y: 7.02, w: 0.6, h: 0.28, fontFace: BODY, fontSize: 9.5,
    color: INK3, align: "right", margin: 0,
  });
}

/* ====================================================================== 1 */
let s = pres.addSlide();
s.background = { color: NAVY };
s.addImage({ data: LOGO, x: MG, y: 0.62, h: 0.62, w: 1.22 });
s.addShape(pres.ShapeType.rect, {
  x: MG, y: 0.55, w: 1.34, h: 0.76, fill: { color: WHITE }, line: { color: WHITE },
});
s.addImage({ data: LOGO, x: MG + 0.06, y: 0.62, h: 0.62, w: 1.22 });
s.addText("Teaching a Translation Model a Language It Never Knew", {
  x: MG, y: 1.95, w: 10.6, h: 1.9, fontFace: H1, fontSize: 40, bold: true,
  color: WHITE, margin: 0, lineSpacingMultiple: 1.06,
});
s.addText("Fine-tuning NLLB-200 to add Ekegusii and adapt it to Kenyan public service announcements", {
  x: MG, y: 3.95, w: 10.2, h: 0.8, fontFace: BODY, fontSize: 17,
  color: "C9D2F0", margin: 0, lineSpacingMultiple: 1.1,
});
s.addShape(pres.ShapeType.rect, {
  x: MG, y: 5.05, w: 1.5, h: 0.055, fill: { color: GOLD }, line: { color: GOLD },
});
s.addText(TEAM_LINE, {
  x: MG, y: 5.34, w: 11.9, h: 0.34, fontFace: BODY, fontSize: 13.5,
  bold: true, color: WHITE, margin: 0,
});
s.addText("Supervisor: " + SUPERVISOR, {
  x: MG, y: 5.7, w: 11.9, h: 0.3, fontFace: BODY, fontSize: 13,
  color: "C9D2F0", margin: 0,
});
s.addText("Natural Language Processing · School of Science and Technology · 2026", {
  x: MG, y: 6.02, w: 11.9, h: 0.3, fontFace: BODY, fontSize: 12,
  color: "9AA8D8", margin: 0,
});
s.addNotes("Ekegusii, also called Kisii, is spoken by about 2.7 million people in " +
  "south-western Kenya. NLLB-200 covers 200 languages and Ekegusii is not one of them. " +
  "This project adds it and adapts the model to the register of public service announcements.");

/* ====================================================================== 2 */
s = pres.addSlide();
titleSlide(s, "The gap", "A model for 200 languages, and Ekegusii is not one of them");
stat(s, { x: MG, y: 1.85, w: 3.0, value: "202", label: "languages supported by NLLB-200" });
stat(s, { x: MG + 3.4, y: 1.85, w: 3.0, value: "0", label: "of them are Ekegusii", color: "B23B3A" });
stat(s, { x: MG + 6.8, y: 1.85, w: 3.6, value: "2.7", unit: "million", label: "Ekegusii speakers in Kenya" });
card(s, { x: MG, y: 3.5, w: CW, h: 1.25, disc: "!",
  head: "This is not a quality problem — it is an absence",
  body: "The model has no token for Ekegusii, so it cannot be asked for it at all. " +
        "Ask NLLB for the nearest language it does know, Kikuyu, and you score 14.6 chrF2++ — " +
        "the floor this project has to beat.", discColor: GOLD });
card(s, { x: MG, y: 5.0, w: CW, h: 1.45,
  head: "Why public service announcements",
  body: "Cholera advisories, HELB loan deadlines, KRA filing dates, NTSA road-safety notices. " +
        "These are issued in English and Kiswahili. A Kenyan who reads Ekegusii most comfortably " +
        "receives health and legal information in a second or third language, or not at all.",
  tint: TINT_G });
footer(s, 2);
s.addNotes("The point to land: this is not 'the translation is poor'. The language is absent from " +
  "the model's vocabulary. Nothing can be generated at all until the vocabulary is extended.");

/* ====================================================================== 3 */
s = pres.addSlide();
titleSlide(s, "Objectives", "What the project set out to do");
const objectives = [
  ["1", "Add Ekegusii by transfer learning", "Extend the tokenizer with guz_Latn and transfer what NLLB already knows about related Bantu languages into it, rather than initialising from noise."],
  ["2", "Build a parallel corpus", "No Ekegusii MT corpus existed. Align scripture, storybooks and contemporary sentences into English–Ekegusii–Kiswahili triples."],
  ["3", "Adapt to PSA register", "Scripture teaches the language but not the voice of a government notice. Use real Kenyan PSAs to close that gap."],
  ["4", "Test a curriculum hypothesis", "Does learning the language first, then the register, beat training on everything at once? Run the control that can falsify it."],
];
objectives.forEach(([n, h, b], i) => {
  card(s, { x: MG + (i % 2) * (CW / 2 + 0.15), y: 1.85 + Math.floor(i / 2) * 1.62,
            w: CW / 2 - 0.15, h: 1.42, disc: n, head: h, body: b });
});
footer(s, 3);
s.addNotes("Objective 4 is the one that produced the interesting result, and it is the one that " +
  "did not go as planned.");

/* ====================================================================== 4 */
s = pres.addSlide();
titleSlide(s, "Data", "What the model actually trained on");
s.addTable([
  [{ text: "Source", options: { bold: true } },
   { text: "Direction", options: { bold: true } },
   { text: "Pairs", options: { bold: true, align: "right" } }],
  ["Ekegusii Bible", "English to Ekegusii", { text: "28,439", options: { align: "right" } }],
  ["Ekegusii Bible", "Kiswahili to Ekegusii", { text: "28,482", options: { align: "right" } }],
  ["Lughayangu everyday sentences", "English to Ekegusii", { text: "111", options: { align: "right" } }],
  ["Duplicates removed", "", { text: "-55", options: { align: "right" } }],
  ["Kenyan PSAs", "English to Ekegusii", { text: "3,509", options: { align: "right" } }],
  ["Kenyan PSAs", "Kiswahili to Ekegusii", { text: "2,183", options: { align: "right" } }],
  [{ text: "Total unique training pairs", options: { bold: true } }, "",
   { text: "62,669", options: { bold: true, align: "right" } }],
], {
  x: MG, y: 1.78, w: 7.5, colW: [3.0, 2.6, 1.9],
  fontFace: BODY, fontSize: 11.5, color: INK,
  border: { type: "solid", color: LINE, pt: 0.5 },
  fill: { color: WHITE }, rowH: 0.35, valign: "middle",
});
card(s, { x: MG + 7.8, y: 1.78, w: CW - 7.8, h: 1.65, disc: "x4",
  head: "Why 79,745 examples, not 62,669",
  body: "The 5,692 PSA pairs are repeated four times per epoch so the domain is " +
        "not swamped by scripture. Unique pairs stay 62,669.", tint: TINT_G, discColor: GOLD });
card(s, { x: MG + 7.8, y: 3.6, w: CW - 7.8, h: 1.98, disc: "0",
  head: "Two sources contributed nothing",
  body: "4laws gave 27,575 pairs, every one a duplicate of the eBible text. The 110 " +
        "African Storybook rows were merged into the corpus after training finished, so " +
        "the released model never saw them." });
card(s, { x: MG, y: 4.5, w: 7.5, h: 1.9, disc: "✓",
  head: "Held out and never trained on",
  body: "2,993 scripture pairs, 200 everyday sentences and 944 PSA pairs. The PSA " +
        "held-out set is what every headline number is measured on." });
footer(s, 4);
s.addNotes("If asked how much data: 62,669 unique parallel sentence pairs. Not 80,000 - that " +
  "figure counts the PSA data four times. Storybooks are in the repository but reached the " +
  "corpus after training, so do not claim them.");

/* ====================================================================== 5 */
s = pres.addSlide();
titleSlide(s, "The core risk", "Scripture teaches the language, not the vocabulary");
s.addText([
  { text: "53.6%", options: { fontSize: 62, bold: true, color: "B23B3A" } },
], { x: MG, y: 1.8, w: 3.2, h: 0.95, fontFace: BODY, margin: 0, valign: "middle" });
s.addText("of English content-word types in the PSA corpus never appear in the aligned training data",
  { x: MG, y: 2.75, w: 3.4, h: 1.0, fontFace: BODY, fontSize: 13, color: INK2,
    margin: 0, lineSpacingMultiple: 1.15 });
s.addText("42.3% of tokens", { x: MG, y: 3.75, w: 3.4, h: 0.3, fontFace: BODY,
  fontSize: 12, color: INK3, margin: 0, italic: true });

card(s, { x: MG + 3.9, y: 1.8, w: CW - 3.9, h: 1.45,
  head: "Words the Bible has no reason to contain",
  body: "portal · bursary · HELB · KUCCPS · iTax · NTSA · Huduma Centre · county government · " +
        "e-citizen · registration deadline", tint: TINT_G });
card(s, { x: MG + 3.9, y: 3.45, w: CW - 3.9, h: 1.35,
  head: "Why this drives the whole design",
  body: "A model trained only on scripture will translate a cholera advisory as though it were a " +
        "psalm. The register gap — not the language gap — is what the second phase of training exists to close." });
card(s, { x: MG, y: 5.05, w: CW, h: 1.35, disc: "→",
  head: "Measured before training, not discovered afterwards",
  body: "The exploratory analysis quantified this gap first. It set the expectation that institutional " +
        "vocabulary would remain the model's weakest point — which the final error analysis confirmed." });
footer(s, 5);
s.addNotes("This is the slide to slow down on. It is the honest statement of what the model cannot " +
  "be expected to do well, and it was known before any GPU time was spent.");

/* ====================================================================== 6 */
s = pres.addSlide();
titleSlide(s, "Method · step one", "Adding a language to a frozen vocabulary");
card(s, { x: MG, y: 1.8, w: CW / 3 - 0.2, h: 2.1, disc: "1",
  head: "Add the token",
  body: "guz_Latn is appended to the tokenizer as a special token and the embedding matrix is " +
        "resized from 256,204 to 256,205." });
card(s, { x: MG + CW / 3 + 0.1, y: 1.8, w: CW / 3 - 0.2, h: 2.1, disc: "2",
  head: "Seed it, don't randomise it",
  body: "The new row is copied from kik_Latn (Kikuyu) — the closest Kenyan Bantu language NLLB " +
        "supports — plus 1% noise so the two can diverge during training.",
  tint: TINT_G, discColor: GOLD });
card(s, { x: MG + 2 * (CW / 3) + 0.2, y: 1.8, w: CW / 3 - 0.2, h: 2.1, disc: "3",
  head: "Build inputs by hand",
  body: "NLLB's tokenizer does not know about a language added after pretraining, so sequences are " +
        "assembled explicitly as [lang] tokens [eos]." });
card(s, { x: MG, y: 4.15, w: CW, h: 1.5,
  head: "This is transfer learning, and the seed is where the transfer happens",
  body: "Transfer learning reuses what a model already knows instead of starting over. NLLB has " +
        "learned Bantu structure from Kikuyu, Kiswahili and others; seeding guz_Latn from kik_Latn " +
        "hands Ekegusii that knowledge as a starting point. A random embedding gives the decoder no " +
        "prior at all, and 62,669 pairs is nowhere near enough to learn a language from nothing.",
  tint: TINT_G, discColor: GOLD, disc: "\u21ba" });
s.addText("Same initialisation, same data, same seed for every run — so what the experiment compares is the ordering of training, and nothing else.",
  { x: MG, y: 5.85, w: CW, h: 0.5, fontFace: BODY, fontSize: 13, color: INK2,
    italic: true, margin: 0 });
footer(s, 6);
s.addNotes("If asked why Kikuyu: it is the nearest Kenyan Bantu language in NLLB. Not the nearest " +
  "genetically — Ekegusii is in a different subgroup — but the nearest available, and the honest " +
  "framing is 'best available proxy'.");

/* ====================================================================== 7 */
s = pres.addSlide();
titleSlide(s, "Method · step two", "The hypothesis, and the control that could kill it");
card(s, { x: MG, y: 1.8, w: CW / 3 - 0.2, h: 2.35, disc: "A",
  head: "Stage 1 — learn the language",
  body: "57,000 examples of Bible, storybooks and everyday sentences. The model learns Ekegusii. It " +
        "has never seen a public notice." });
card(s, { x: MG + CW / 3 + 0.1, y: 1.8, w: CW / 3 - 0.2, h: 2.35, disc: "B",
  head: "Stage 2 — learn the register",
  body: "Continue from stage 1 on 30,000 examples: real PSAs upsampled four times, with 25% scripture " +
        "replayed to prevent the model forgetting what stage 1 taught it." });
card(s, { x: MG + 2 * (CW / 3) + 0.2, y: 1.8, w: CW / 3 - 0.2, h: 2.35, disc: "C",
  head: "Control — everything at once",
  body: "The same 79,745 training examples in a single pass, no ordering. Exists solely to answer: was the " +
        "curriculum worth anything?", tint: TINT_G, discColor: GOLD });
card(s, { x: MG, y: 4.4, w: CW, h: 1.55, disc: "?",
  head: "The control is the point, not an afterthought",
  body: "Without run C, 'the two-stage curriculum improved PSA translation by 11 chrF2++' is " +
        "unfalsifiable — the gain could come entirely from seeing PSA data at all, in any order. " +
        "A curriculum claim needs a same-data, different-order comparison, and that is the first " +
        "thing a reviewer will ask for." });
footer(s, 7);
s.addNotes("Stress that all three runs share initialisation, data, hyperparameters and seed. Only " +
  "the ordering differs.");

/* ====================================================================== 8 */
s = pres.addSlide();
titleSlide(s, "Evaluation", "Why chrF2++, and why BLEU would have misled us");
card(s, { x: MG, y: 1.8, w: CW / 2 - 0.15, h: 2.2, disc: "✕",
  head: "BLEU counts whole words",
  body: "Ekegusii is agglutinative: subject, tense, object and negation attach to a verb stem, so one " +
        "word carries what English spreads over five. Get the stem right and one affix wrong and BLEU " +
        "scores that word zero — a near-perfect translation reads as a failure.",
  tint: TINT_G, discColor: GOLD });
card(s, { x: MG + CW / 2 + 0.15, y: 1.8, w: CW / 2 - 0.15, h: 2.2, disc: "✓",
  head: "chrF2++ counts character n-grams",
  body: "It scores overlapping character sequences, plus word unigrams and bigrams. A correct stem " +
        "with an imperfect affix earns most of the credit it deserves, which is what a human rater " +
        "would also give it." });
// Our own numbers make the argument better than any explanation could.
s.addText("What that looks like in our results, English into Ekegusii on real PSAs:",
  { x: MG, y: 4.28, w: CW, h: 0.3, fontFace: BODY, fontSize: 13, bold: true,
    color: INK, margin: 0 });
s.addTable([
  [{ text: "System", options: { bold: true } },
   { text: "BLEU", options: { bold: true, align: "right" } },
   { text: "chrF2++", options: { bold: true, align: "right" } },
   { text: "What BLEU would have told you", options: { bold: true } }],
  ["Stock NLLB-200", { text: "1.63", options: { align: "right" } },
   { text: "14.56", options: { align: "right" } }, "near zero"],
  ["Stage 1, language only", { text: "2.54", options: { align: "right" } },
   { text: "23.96", options: { align: "right" } },
   "still near zero, though the model now speaks Ekegusii"],
  ["Two-stage curriculum", { text: "7.21", options: { align: "right" } },
   { text: "34.93", options: { align: "right" } }, "a fraction of the real gain"],
  [{ text: "Single pass (released)", options: { bold: true } },
   { text: "12.33", options: { align: "right", bold: true } },
   { text: "40.97", options: { align: "right", bold: true } },
   "7.6x the floor, against 2.8x on chrF2++"],
], {
  x: MG, y: 4.66, w: CW, colW: [2.9, 1.1, 1.3, 6.6],
  fontFace: BODY, fontSize: 11.5, color: INK,
  border: { type: "solid", color: LINE, pt: 0.5 },
  fill: { color: WHITE }, rowH: 0.32, valign: "middle",
});
s.addText("Stage 1 reaches 23.96 chrF2++ and scores 2.54 BLEU. A metric that cannot separate a working model from a broken one is not measuring the right thing.",
  { x: MG, y: 6.35, w: CW, h: 0.5, fontFace: BODY, fontSize: 13, color: INK2,
    italic: true, margin: 0 });
footer(s, 8);
s.addNotes("Expect the question 'why not BLEU'. The answer is morphology. Also mention COMET was " +
  "not used because it has no Ekegusii coverage — its encoder has never seen the language, so its " +
  "scores would be meaningless here.");

/* ====================================================================== 9 */
s = pres.addSlide();
titleSlide(s, "Setup", "What was actually run");
const setup = [
  ["Base model", "facebook/nllb-200-distilled-600M"],
  ["Hardware", "1 × NVIDIA A100-SXM4-80GB"],
  ["Precision", "bf16, gradient checkpointing"],
  ["Optimiser", "8-bit AdamW, effective batch 48"],
  ["Learning rate", "5e-5 stage 1 and control · 1.5e-5 stage 2"],
  ["Epochs", "3 per run, best checkpoint by dev loss"],
  ["Decoding", "beam 4, max 128 new tokens"],
  ["Seed", "42, fixed across all three runs"],
];
setup.forEach(([k, v], i) => {
  const x = MG + (i % 2) * (CW / 2 + 0.15);
  const y = 1.85 + Math.floor(i / 2) * 0.62;
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w: CW / 2 - 0.15, h: 0.5, fill: { color: TINT }, rectRadius: 0.06,
    line: { color: TINT },
  });
  s.addText(k, { x: x + 0.22, y, w: 1.9, h: 0.5, fontFace: BODY, fontSize: 11.5,
    color: INK3, margin: 0, valign: "middle" });
  s.addText(v, { x: x + 2.15, y, w: CW / 2 - 2.5, h: 0.5, fontFace: BODY,
    fontSize: 12.5, color: INK, bold: true, margin: 0, valign: "middle" });
});
card(s, { x: MG, y: 4.5, w: CW, h: 1.5, disc: "⚙",
  head: "Training is a script, not a notebook",
  body: "train_stages.py waits for free VRAM on a shared card, recovers from out-of-memory by halving " +
        "the batch, and can resume a single stage. None of that survives a kernel restart — which is " +
        "why the released weights come from the script, and why the training notebook was removed " +
        "rather than left as a second, subtly different path." });
footer(s, 9);
s.addNotes("The GPU was shared. A large part of the engineering effort went into surviving that: " +
  "waiting for memory, recovering from OOM, and checkpointing so an interrupted run was not a lost day.");

/* ===================================================================== 10 */
s = pres.addSlide();
titleSlide(s, "Results", "Both directions, four systems, chrF2++");

// Two charts, not one with eight series per group. Cramming both directions
// into a single plot would put 8 bars in every category - unreadable from the
// back of a room, and the comparison people actually make is within a
// direction, not across them. Small multiples, shared legend, same y-scale so
// the two panels can be compared by eye.
const SYSCOLORS = ["C8CDD8", "9AA8D8", "6C7FC4", NAVY];
const SYSLABELS = ["Stock NLLB-200", "Stage 1 (language only)",
                   "Two-stage curriculum", "Single pass (released)"];

// one shared legend, drawn by hand so both panels answer to it
let lx = MG;
SYSLABELS.forEach((lab, i) => {
  s.addShape(pres.ShapeType.roundRect, {
    x: lx, y: 1.79, w: 0.17, h: 0.17, fill: { color: SYSCOLORS[i] },
    rectRadius: 0.04, line: { color: SYSCOLORS[i] },
  });
  s.addText(lab, { x: lx + 0.24, y: 1.74, w: 2.5, h: 0.28, fontFace: BODY,
    fontSize: 11, color: INK2, margin: 0, valign: "middle" });
  lx += 0.3 + lab.length * 0.072;
});

const CHART_OPTS = {
  barDir: "col", barGapWidthPct: 45,
  chartColors: SYSCOLORS,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9,
  dataLabelColor: INK2, dataLabelFormatCode: "0.0",
  showLegend: false,
  catAxisLabelColor: INK2, catAxisLabelFontSize: 10.5,
  valAxisLabelColor: INK3, valAxisLabelFontSize: 9,
  valAxisMaxVal: 60, valAxisMinVal: 0,
  valGridLine: { color: LINE, size: 0.75 }, catGridLine: { style: "none" },
  showTitle: false,
};

const ENG_SETS = ["Real PSAs", "Scripture", "Everyday prose"];
s.addText("English into Ekegusii", { x: MG, y: 2.16, w: 6, h: 0.28,
  fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
s.addChart(pres.ChartType.bar, [
  { name: SYSLABELS[0], labels: ENG_SETS, values: [14.56, 14.88, 11.41] },
  { name: SYSLABELS[1], labels: ENG_SETS, values: [23.96, 49.66, 28.97] },
  { name: SYSLABELS[2], labels: ENG_SETS, values: [34.93, 48.95, 29.37] },
  { name: SYSLABELS[3], labels: ENG_SETS, values: [40.97, 49.81, 32.98] },
], { ...CHART_OPTS, x: MG - 0.15, y: 2.46, w: 6.6, h: 3.35 });

const SWH_SETS = ["Real PSAs", "Scripture"];
s.addText("Kiswahili into Ekegusii", { x: MG + 6.7, y: 2.16, w: 5, h: 0.28,
  fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
s.addChart(pres.ChartType.bar, [
  { name: SYSLABELS[0], labels: SWH_SETS, values: [14.13, 14.65] },
  { name: SYSLABELS[1], labels: SWH_SETS, values: [24.49, 49.11] },
  { name: SYSLABELS[2], labels: SWH_SETS, values: [34.50, 48.80] },
  { name: SYSLABELS[3], labels: SWH_SETS, values: [39.61, 49.30] },
], { ...CHART_OPTS, x: MG + 6.55, y: 2.46, w: 5.35, h: 3.35 });

card(s, { x: MG, y: 5.95, w: CW, h: 1.0,
  head: "The two directions agree, which is itself the point: the gain is not an artifact of one source language.",
  body: "" });
footer(s, 10);
s.addNotes("Read within a panel, not across. The released model is the darkest bar and it is " +
  "highest everywhere. Kiswahili has no everyday-prose column because the Lughayangu corpus is " +
  "English-Ekegusii only - say that before someone asks.");

/* ============================================================== 10b metrics */
s = pres.addSlide();
titleSlide(s, "Results in full", "Every system, both metrics, both directions");
s.addTable([
  [{ text: "Direction", options: { bold: true } },
   { text: "Test set", options: { bold: true } },
   { text: "n", options: { bold: true, align: "right" } },
   { text: "Stock", options: { bold: true, align: "right" } },
   { text: "Stage 1", options: { bold: true, align: "right" } },
   { text: "Two-stage", options: { bold: true, align: "right" } },
   { text: "Released", options: { bold: true, align: "right" } }],
  [{ text: "chrF2++", options: { bold: true, color: NAVY } }, "", "", "", "", "", ""],
  ["English to Ekegusii", "Real PSAs", { text: "570", options: { align: "right" } }, { text: "14.56", options: { align: "right" } }, { text: "23.96", options: { align: "right" } }, { text: "34.93", options: { align: "right" } },
   { text: "40.97", options: { bold: true } }],
  ["English to Ekegusii", "Scripture", { text: "1,459", options: { align: "right" } }, { text: "14.88", options: { align: "right" } }, { text: "49.66", options: { align: "right" } }, { text: "48.95", options: { align: "right" } },
   { text: "49.81", options: { bold: true } }],
  ["English to Ekegusii", "Everyday prose", { text: "200", options: { align: "right" } }, { text: "11.41", options: { align: "right" } }, { text: "28.97", options: { align: "right" } }, { text: "29.37", options: { align: "right" } },
   { text: "32.98", options: { bold: true } }],
  ["Kiswahili to Ekegusii", "Real PSAs", { text: "371", options: { align: "right" } }, { text: "14.13", options: { align: "right" } }, { text: "24.49", options: { align: "right" } }, { text: "34.50", options: { align: "right" } },
   { text: "39.61", options: { bold: true } }],
  ["Kiswahili to Ekegusii", "Scripture", { text: "1,463", options: { align: "right" } }, { text: "14.65", options: { align: "right" } }, { text: "49.11", options: { align: "right" } }, { text: "48.80", options: { align: "right" } },
   { text: "49.30", options: { bold: true } }],
  [{ text: "BLEU", options: { bold: true, color: NAVY } }, "", "", "", "", "", ""],
  ["English to Ekegusii", "Real PSAs", { text: "570", options: { align: "right" } }, { text: "1.63", options: { align: "right" } }, { text: "2.54", options: { align: "right" } }, { text: "7.21", options: { align: "right" } },
   { text: "12.33", options: { bold: true } }],
  ["English to Ekegusii", "Scripture", { text: "1,459", options: { align: "right" } }, { text: "0.83", options: { align: "right" } }, { text: "19.75", options: { align: "right" } }, { text: "19.33", options: { align: "right" } },
   { text: "19.91", options: { bold: true } }],
  ["English to Ekegusii", "Everyday prose", { text: "200", options: { align: "right" } }, { text: "0.07", options: { align: "right" } }, { text: "1.27", options: { align: "right" } }, { text: "1.21", options: { align: "right" } },
   { text: "1.59", options: { bold: true } }],
  ["Kiswahili to Ekegusii", "Real PSAs", { text: "371", options: { align: "right" } }, { text: "1.01", options: { align: "right" } }, { text: "1.77", options: { align: "right" } }, { text: "6.87", options: { align: "right" } },
   { text: "10.66", options: { bold: true } }],
  ["Kiswahili to Ekegusii", "Scripture", { text: "1,463", options: { align: "right" } }, { text: "1.20", options: { align: "right" } }, { text: "19.79", options: { align: "right" } }, { text: "19.79", options: { align: "right" } },
   { text: "19.97", options: { bold: true } }],
], {
  x: MG, y: 1.75, w: CW, colW: [2.55, 1.85, 0.75, 1.15, 1.15, 1.3, 1.15],
  fontFace: BODY, fontSize: 10.5, color: INK,
  border: { type: "solid", color: LINE, pt: 0.5 },
  fill: { color: WHITE }, rowH: 0.29, valign: "middle",
});
card(s, { x: MG, y: 6.0, w: CW, h: 0.95,
  head: "Stock NLLB cannot produce Ekegusii and was asked for Kikuyu, so its row is a floor rather than a baseline. Everyday prose has no Kiswahili row: the Lughayangu corpus is English to Ekegusii only.",
  body: "" });
footer(s, 11);
s.addNotes("Do not read this table aloud. It is here so the panel can check any number they " +
  "want, and so the report can quote it. Point at the released column and move on.");

/* ===================================================================== 11 */
s = pres.addSlide();
s.background = { color: NAVY_D };
s.addText("THE FINDING", {
  x: MG, y: 0.72, w: CW, h: 0.3, fontFace: BODY, fontSize: 12, bold: true,
  color: GOLD, charSpacing: 2.4, margin: 0,
});
s.addText("The curriculum lost to its own control", {
  x: MG, y: 1.12, w: 10.5, h: 0.75, fontFace: H1, fontSize: 32, bold: true,
  color: WHITE, margin: 0,
});
s.addText([
  { text: "−6.04", options: { fontSize: 56, bold: true, color: GOLD } },
  { text: "  chrF2++", options: { fontSize: 20, bold: true, color: "C9D2F0" } },
], { x: MG, y: 2.15, w: 5.4, h: 0.9, fontFace: BODY, margin: 0, valign: "middle" });
s.addText("The two-stage curriculum scored 34.93 on real PSAs. Training on the same data in a single pass scored 40.97.",
  { x: MG, y: 3.1, w: 5.4, h: 1.1, fontFace: BODY, fontSize: 14, color: "C9D2F0",
    margin: 0, lineSpacingMultiple: 1.2 });

const kills = [
  ["Not a data advantage", "The single pass sees every unique example the curriculum sees. The replay slice is duplicated scripture, not new material."],
  ["Not a compute advantage", "At equal epochs the two-stage run takes about 9% more gradient updates, because replayed rows are trained on twice. It had more compute and still lost."],
  ["A plausible mechanism", "Stage 2's dev loss bottomed at 1.357 and drifted back to 1.417 — it overfitted a small, PSA-heavy set. The single pass sees PSAs interleaved with scripture throughout, which regularises it."],
];
kills.forEach(([h, b], i) => {
  const y = 2.15 + i * 1.5;
  s.addShape(pres.ShapeType.roundRect, {
    x: MG + 5.9, y, w: CW - 5.9, h: 1.32, fill: { color: "22326F" },
    rectRadius: 0.08, line: { color: "22326F" },
  });
  s.addText(h, { x: MG + 6.15, y: y + 0.16, w: CW - 6.4, h: 0.32, fontFace: BODY,
    fontSize: 14, bold: true, color: WHITE, margin: 0 });
  s.addText(b, { x: MG + 6.15, y: y + 0.5, w: CW - 6.4, h: 0.72, fontFace: BODY,
    fontSize: 11.5, color: "B9C4E8", margin: 0, lineSpacingMultiple: 1.1 });
});
s.addText("A control that falsifies your hypothesis is a result, not a failure. It is also the reason the number can be trusted.",
  { x: MG, y: 6.35, w: CW, h: 0.5, fontFace: BODY, fontSize: 14, color: GOLD,
    italic: true, margin: 0 });
footer(s, 12);
s.addNotes("Do not bury this. It is the most defensible finding in the project, because it is the " +
  "one the experiment was designed to be able to disprove. The released model is the control.");

/* ===================================================================== 12 */
s = pres.addSlide();
titleSlide(s, "Results in context", "What the released model actually achieves");
stat(s, { x: MG, y: 1.85, w: 2.9, value: "40.97", label: "chrF2++ on real Kenyan PSAs, English → Ekegusii" });
stat(s, { x: MG + 3.15, y: 1.85, w: 2.9, value: "+181%", label: "relative gain over the no-Ekegusii floor" });
stat(s, { x: MG + 6.3, y: 1.85, w: 2.9, value: "+17.0", label: "over stage 1 — the value of PSA adaptation" });
stat(s, { x: MG + 9.45, y: 1.85, w: 2.4, value: "0", label: "measurable forgetting of general Ekegusii", color: "1A7F52" });
card(s, { x: MG, y: 3.65, w: CW / 2 - 0.15, h: 1.5,
  head: "It does not forget",
  body: "The released model scores 49.81 on held-out scripture — higher than the model trained on " +
        "scripture alone. Joint training simply does not create the forgetting problem that replay was " +
        "designed to solve." });
card(s, { x: MG + CW / 2 + 0.15, y: 3.65, w: CW / 2 - 0.15, h: 1.5,
  head: "It generalises beyond both domains",
  body: "32.98 on contemporary everyday prose, against 28.97 for the scripture-only model — so the " +
        "gain is not simply memorising PSA phrasing.", tint: TINT_G });
card(s, { x: MG, y: 5.3, w: CW, h: 1.15, disc: "◷",
  head: "Three checkpoints published, each verified by weight fingerprint",
  body: "Baseline, curriculum and released model are all on Hugging Face, so the comparison can be " +
        "reproduced rather than taken on trust." });
footer(s, 13);
s.addNotes("If asked about the Kiswahili direction: 39.61 released against 34.50 curriculum, same story.");

/* ===================================================================== 13 */
s = pres.addSlide();
titleSlide(s, "Limitations", "What this project cannot claim");
const lims = [
  ["No human evaluation", "Every number here is an automatic metric against a single reference. No fluency, adequacy or cultural-accuracy ratings have been collected from Ekegusii speakers. This is the largest outstanding gap."],
  ["Unverified references", "The PSA Ekegusii was supplied by the supervisor. The translator and quality-assurance process are unknown and no inter-annotator agreement exists."],
  ["Scripture-heavy mixture", "About 57,000 of the 62,669 unique parallel pairs are Bible verses, so the model may still lean formal on unfamiliar input."],
  ["Institutional vocabulary", "The 53.6% content-word gap was measured, not solved. HELB, KUCCPS and iTax remain the weakest cases — and the confidence score does not catch them."],
];
lims.forEach(([h, b], i) => {
  card(s, { x: MG + (i % 2) * (CW / 2 + 0.15), y: 1.85 + Math.floor(i / 2) * 1.72,
            w: CW / 2 - 0.15, h: 1.55, disc: "·", head: h, body: b });
});
s.addText("Not suitable for medical, legal or emergency communication without review by a fluent Ekegusii speaker.",
  { x: MG, y: 5.5, w: CW, h: 0.5, fontFace: BODY, fontSize: 13.5, color: "B23B3A",
    bold: true, margin: 0 });
footer(s, 14);
s.addNotes("Say the human-evaluation gap out loud before anyone asks. The demo's correction form is " +
  "the mechanism for closing it.");

/* ===================================================================== 14 */
s = pres.addSlide();
titleSlide(s, "Deployment", "A working translator, and a way to collect corrections");
card(s, { x: MG, y: 1.8, w: CW / 3 - 0.2, h: 2.2, disc: "▸",
  head: "Live demo",
  body: "Pick a direction — English or Kiswahili into Ekegusii — type an announcement, get a " +
        "translation with a confidence label. Runs on GPU at about a second per sentence." });
card(s, { x: MG + CW / 3 + 0.1, y: 1.8, w: CW / 3 - 0.2, h: 2.2, disc: "✎",
  head: "Correction form",
  body: "Any Ekegusii speaker can rate a translation and type what it should have said. Every " +
        "correction is both the missing human evaluation and a training pair for the next round.",
  tint: TINT_G, discColor: GOLD });
card(s, { x: MG + 2 * (CW / 3) + 0.2, y: 1.8, w: CW / 3 - 0.2, h: 2.2, disc: "⚑",
  head: "Honest confidence",
  body: "The label reports the model's own certainty, not a probability of being correct. It is " +
        "uncalibrated, and the interface says so rather than implying reliability it cannot offer." });
card(s, { x: MG, y: 4.25, w: CW, h: 1.35, disc: "→",
  head: "Next: collect 100+ rated sentences from Ekegusii speakers",
  body: "That converts the largest limitation into a result, and turns the deployment from a " +
        "demonstration into a data-collection instrument." });
s.addText("github.com/SamAbr/PSA-MT   ·   huggingface.co/samuelabrha",
  { x: MG, y: 5.9, w: CW, h: 0.4, fontFace: BODY, fontSize: 13, color: NAVY,
    bold: true, margin: 0 });
footer(s, 15);
s.addNotes("Have the demo already open in a browser tab before this slide. Do not start it live.");

/* ===================================================================== 15 */
s = pres.addSlide();
s.background = { color: NAVY };
s.addText("What this project shows", {
  x: MG, y: 1.5, w: 10.5, h: 0.6, fontFace: H1, fontSize: 30, bold: true,
  color: WHITE, margin: 0,
});
const takeaways = [
  "Transfer learning can add a language a large multilingual model has never seen. A few tens of thousands of parallel sentences and a well-chosen embedding seed are enough.",
  "Domain adaptation matters as much as language coverage: scripture teaches Ekegusii, but not the voice of a public notice.",
  "The control mattered more than the hypothesis. Testing the curriculum properly is what makes the released number worth believing.",
];
takeaways.forEach((t, i) => {
  const y = 2.5 + i * 1.15;
  s.addShape(pres.ShapeType.ellipse, {
    x: MG, y: y + 0.06, w: 0.34, h: 0.34, fill: { color: GOLD }, line: { color: GOLD },
  });
  s.addText(String(i + 1), { x: MG, y: y + 0.06, w: 0.34, h: 0.34, align: "center",
    valign: "middle", fontFace: BODY, fontSize: 12, bold: true, color: INK, margin: 0 });
  // valign top, or a single-line takeaway centres in its 0.95" box and the
  // numbered disc ends up floating above the text it belongs to.
  s.addText(t, { x: MG + 0.62, y: y - 0.02, w: 11.3, h: 0.95, fontFace: BODY,
    fontSize: 15, color: "D5DCF4", margin: 0, valign: "top",
    lineSpacingMultiple: 1.18 });
});
s.addText(TEAM_LINE, {
  x: MG, y: 6.28, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5,
  color: "C9D2F0", margin: 0,
});
s.addText("Supervisor: " + SUPERVISOR + "  ·  USIU-Africa, 2026", {
  x: MG, y: 6.6, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5,
  color: "9AA8D8", margin: 0,
});
s.addNotes("Close on the control point. It is the methodological contribution and the thing that " +
  "distinguishes this from a fine-tuning exercise.");

pres.writeFile({ fileName: "Ekegusii_NMT_presentation.pptx" })
  .then(f => console.log("wrote", f));
