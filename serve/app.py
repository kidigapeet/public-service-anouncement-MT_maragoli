"""
app.py - a four-system Ekegusii translation service.

Serves the same four systems notebook 04 evaluates, so a visitor can type a
sentence and watch the adaptation gain happen instead of reading it off a table:

    stock   facebook/nllb-200-distilled-600M, asked for kik_Latn (Kikuyu)
            because Ekegusii is not in its vocabulary at all. This is the
            floor, not a fair comparison - see NOTE_STOCK below.
    stage1  fine-tuned on Bible + storybooks. The model knows Ekegusii.
    stage2  stage 1, then adapted on Kenyan PSAs with Bible replay.
    mixed   everything in one pass, no curriculum. THE RELEASED MODEL.

The naming is a fossil worth explaining: "mixed" was built as the *control* for
a two-stage curriculum, and it beat the curriculum on every test set - by 6.04
chrF2++ on real PSAs, while also scoring higher than stage 1 on scripture. The
ablation falsified the hypothesis it was designed to test, so the control is
what gets shipped and stage 2 is kept only as evidence for that claim.

The three fine-tuned checkpoints live in *private* Hugging Face repositories,
so the process needs HF_TOKEN in its environment. Nothing here writes to the
Hub; a read-scoped token is enough and is what you should use.

Run it:
    pip install -r requirements.txt
    export HF_TOKEN=hf_...
    uvicorn app:app --host 0.0.0.0 --port 8000

Or, without any weights at all, to work on the front end:
    MOCK_MODE=1 uvicorn app:app --port 8000
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

log = logging.getLogger("ekegusii")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
STARTED = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ---------------------------------------------------------------------------
# CONFIGURATION - everything tunable is an environment variable, because this
# runs in a container and a container's only interface is its environment.
# ---------------------------------------------------------------------------


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _flag(name: str, default: bool = False) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


MOCK_MODE = _flag("MOCK_MODE")

BASE_MODEL = _env("BASE_MODEL", "facebook/nllb-200-distilled-600M")
HF_STAGE1 = _env("HF_STAGE1", "samuelabrha/nllb-200-600M-ekegusii-stage1")
HF_STAGE2 = _env("HF_STAGE2", "samuelabrha/nllb-200-600M-ekegusii-psa")
HF_MIXED = _env("HF_MIXED", "samuelabrha/nllb-200-600M-ekegusii-mixed")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# How many models may sit in memory at once. Each NLLB-600M is ~2.4 GB in fp32
# and ~1.2 GB in fp16, so four of them is ~9.6 GB of RAM on CPU or ~4.8 GB of
# VRAM. On a 16 GB card or a 16 GB container, keep all four resident. On
# anything smaller, drop this to 2 and accept a reload pause when a visitor
# switches systems - correctness is identical either way.
MAX_RESIDENT = int(_env("MAX_RESIDENT", "4"))
PRELOAD = _flag("PRELOAD", True)

BEAMS = int(_env("BEAMS", "4"))
MAX_NEW_TOKENS = int(_env("MAX_NEW_TOKENS", "128"))
MAX_CHARS = int(_env("MAX_CHARS", "4000"))
MAX_SEGMENTS = int(_env("MAX_SEGMENTS", "40"))

# Which systems this instance serves. The public demo runs ONE model - four of
# them on a 2 vCPU box would make every visitor wait a minute for an answer, and
# a stranger who wants an Ekegusii translation does not care about the ablation.
# Set ENABLED_SYSTEMS=stock,stage1,stage2,mixed for the internal comparison view.
ENABLED_SYSTEMS = [s.strip() for s in _env("ENABLED_SYSTEMS", "mixed").split(",") if s.strip()]

# Where corrections go. A dataset repo id, e.g. "samuelabrha/ekegusii-psa-feedback".
# Unset, feedback is appended to a local file and lost when the container dies -
# fine for a laptop, useless on a Space, so set it before sharing the link.
FEEDBACK_REPO = _env("FEEDBACK_REPO", "")
FEEDBACK_FILE = HERE / "feedback.jsonl"

# Requests per minute per client IP. This matters the moment the service is
# reachable from the internet: it is an unauthenticated endpoint in front of a
# GPU, and a single loop can occupy the card indefinitely. 0 disables it.
RATE_LIMIT_PER_MIN = int(_env("RATE_LIMIT_PER_MIN", "30"))

ENG, SWH, GUZ = "eng_Latn", "swh_Latn", "guz_Latn"

# NOTE_STOCK -------------------------------------------------------------
# Stock NLLB-200 has no guz_Latn token; asking it for Ekegusii is not
# possible. Notebook 04 asks it for kik_Latn (Kikuyu) - the nearest Kenyan
# Bantu language it does support - to establish what "no Ekegusii support"
# looks like numerically. The UI must label this honestly. It is a floor,
# not a baseline, and presenting it as a comparison would be misleading.
STOCK_TARGET = _env("STOCK_TARGET", "kik_Latn")

SYSTEMS: "OrderedDict[str, dict]" = OrderedDict([
    ("stock", {
        "repo": BASE_MODEL,
        "label": "Stock NLLB-200",
        "target": STOCK_TARGET,
        "note": "No Ekegusii in the vocabulary. Asked for Kikuyu (kik_Latn), "
                "the nearest language it supports. A floor, not a fair comparison.",
        "role": "floor",
        "private": False,
    }),
    ("stage1", {
        "repo": HF_STAGE1,
        "label": "Stage 1 · general",
        "target": GUZ,
        "note": "Fine-tuned on the Bible and storybook corpus. Knows Ekegusii, "
                "has never seen a public service announcement.",
        "role": "baseline",
        "private": True,
    }),
    ("stage2", {
        "repo": HF_STAGE2,
        "label": "Stage 2 · curriculum",
        "target": GUZ,
        "note": "Stage 1, then adapted on Kenyan PSAs with Bible replay. The "
                "two-stage curriculum — which lost to the single pass by 6.04 "
                "chrF2++ and is kept as the evidence for that.",
        "role": "ablation",
        "private": True,
    }),
    ("mixed", {
        "repo": HF_MIXED,
        "label": "Single pass",
        "target": GUZ,
        "note": "All the data in one pass, no curriculum. Best on every test "
                "set, including scripture. The released model.",
        "role": "headline",
        "private": True,
    }),
])

unknown_systems = [s for s in ENABLED_SYSTEMS if s not in SYSTEMS]
if unknown_systems:
    raise SystemExit(f"ENABLED_SYSTEMS names systems that do not exist: {unknown_systems}")
SYSTEMS = OrderedDict((k, v) for k, v in SYSTEMS.items() if k in ENABLED_SYSTEMS)

SOURCE_LANGUAGES = [
    {"code": ENG, "label": "English", "direction": "English → Ekegusii"},
    {"code": SWH, "label": "Kiswahili", "direction": "Kiswahili → Ekegusii"},
]

# ---------------------------------------------------------------------------
# SEGMENTATION
# ---------------------------------------------------------------------------
# NLLB is a sentence-level model. Handing it a whole paragraph makes it drop
# clauses, so split, translate each piece, and rejoin. Blank lines are
# preserved as paragraph breaks because a PSA's shape carries meaning.

_SENT_END = re.compile(r"(?<=[.!?:;])\s+(?=[A-Z0-9\"'“])")


def segment(text: str) -> list:
    """Split into translatable units, remembering blank lines."""
    units = []
    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if not stripped:
            units.append(None)          # a paragraph break, not a segment
            continue
        parts = [p.strip() for p in _SENT_END.split(stripped) if p.strip()]
        units.extend(parts or [stripped])
    while units and units[0] is None:
        units.pop(0)
    while units and units[-1] is None:
        units.pop()
    return units


def rejoin(units: list, translations: list) -> str:
    """Put translated segments back where their sources were."""
    out, it = [], iter(translations)
    for u in units:
        out.append("\n\n" if u is None else next(it, ""))
    text = ""
    for piece in out:
        if piece == "\n\n":
            text = text.rstrip() + "\n\n"
        else:
            text += piece + " "
    return text.strip()


# ---------------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------------


class Registry:
    """
    Least-recently-used cache of loaded models.

    Generation holds a lock. One 600M model saturates a GPU on its own, so
    letting two requests decode concurrently would not make either finish
    sooner - it would just double the peak memory and risk an OOM mid-demo.
    """

    def __init__(self):
        self._loaded: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = asyncio.Lock()
        self.device = "cpu"
        self.dtype = None
        self.errors: dict = {}

    # -- lifecycle ---------------------------------------------------------

    def describe(self) -> dict:
        return {"device": self.device, "dtype": str(self.dtype), "mock": MOCK_MODE,
                "resident": list(self._loaded), "max_resident": MAX_RESIDENT}

    def _load(self, name: str):
        """Blocking. Called in a worker thread."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        spec = SYSTEMS[name]
        repo = spec["repo"]
        kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
        log.info("loading %s from %s ...", name, repo)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(repo, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            repo, dtype=self.dtype, **kwargs).to(self.device).eval()

        target = spec["target"]
        if tok.convert_tokens_to_ids(target) == tok.unk_token_id:
            raise RuntimeError(
                f"{repo} has no {target!r} token. For the fine-tuned systems this "
                f"means the tokenizer was not uploaded alongside the weights - "
                f"push_to_hub must be called on the tokenizer too.")
        log.info("loaded %s in %.1fs (vocab %d)", name, time.time() - t0, len(tok))
        return tok, model

    async def get(self, name: str):
        if name in self._loaded:
            self._loaded.move_to_end(name)
            return self._loaded[name]
        pair = await asyncio.to_thread(self._load, name)
        self._loaded[name] = pair
        self._loaded.move_to_end(name)
        while len(self._loaded) > MAX_RESIDENT:
            evicted, _ = self._loaded.popitem(last=False)
            log.info("evicted %s to stay under MAX_RESIDENT=%d", evicted, MAX_RESIDENT)
            self._free()
        return pair

    @staticmethod
    def _free():
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- inference ---------------------------------------------------------

    def _generate(self, tok, model, texts: list, src_lang: str, tgt_lang: str,
                  beams: int, max_new_tokens: int):
        """
        Blocking. Builds NLLB's input format by hand - [src_lang] tokens [eos] -
        which is exactly what notebook 04 does, so a number quoted in the paper
        and a string shown in this UI came out of the same code path.

        Returns (texts, confidences), where a confidence is the geometric-mean
        per-token probability of the chosen output. See NOTE_CONFIDENCE.
        """
        import torch

        eos, pad = tok.eos_token_id, tok.pad_token_id
        tgt_id = tok.convert_tokens_to_ids(tgt_lang)
        enc = [[tok.convert_tokens_to_ids(src_lang)] +
               tok(t, add_special_tokens=False, truncation=True,
                   max_length=max_new_tokens - 2)["input_ids"] + [eos] for t in texts]
        width = max(len(e) for e in enc)
        # Left-pad: the encoder is bidirectional so the side does not change the
        # result, but matching notebook 04 keeps demo and evaluation identical.
        ids = torch.tensor([[pad] * (width - len(e)) + e for e in enc]).to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=(ids != pad).long(),
                                 forced_bos_token_id=tgt_id,
                                 max_new_tokens=max_new_tokens, num_beams=beams,
                                 return_dict_in_generate=True, output_scores=True)

        seqs = out.sequences
        scores = getattr(out, "sequences_scores", None)
        if scores is None:
            # Greedy decoding does not populate sequences_scores; derive the same
            # quantity from per-step logits instead.
            trans = model.compute_transition_scores(
                seqs, out.scores, normalize_logits=True)
            finite = torch.isfinite(trans)
            scores = (trans.masked_fill(~finite, 0.0).sum(-1)
                      / finite.sum(-1).clamp(min=1))
        conf = scores.exp().clamp(0.0, 1.0).tolist()
        return tok.batch_decode(seqs, skip_special_tokens=True), conf

    async def translate(self, name: str, texts: list, src_lang: str,
                        beams: int, max_new_tokens: int):
        tgt_lang = SYSTEMS[name]["target"]
        if MOCK_MODE:
            await asyncio.sleep(0.25)
            tag = {"stock": "[kikuyu-ish]", "stage1": "[scriptural]",
                   "stage2": "[psa-register]", "mixed": "[one-pass]"}[name]
            return [f"{tag} {t}" for t in texts], [0.72] * len(texts)
        tok, model = await self.get(name)
        async with self._lock:
            return await asyncio.to_thread(
                self._generate, tok, model, texts, src_lang, tgt_lang,
                beams, max_new_tokens)


registry = Registry()


# NOTE_CONFIDENCE ------------------------------------------------------------
# What the number is: the geometric-mean per-token probability the model assigned
# to the output it chose. For beam search that is exp(sequences_scores), the
# length-normalised sum of log-probabilities.
#
# What it is NOT: a probability that the translation is correct. Neural MT is
# routinely confident and wrong, and this model has a documented vocabulary gap -
# 53.6% of PSA content-word types never appeared in training. A fluent, confident
# guess at "KUCCPS" is exactly the failure this score will not catch.
#
# The band thresholds below are eyeballed, not calibrated against accuracy. They
# separate "the model was unsure" from "the model was not", nothing more, and the
# UI says so. Calibrating them properly needs the human evaluation that has not
# been done yet - do not present them as reliability estimates until it has.
CONFIDENCE_BANDS = [(0.65, "high"), (0.45, "moderate"), (0.0, "low")]


def band(conf: float) -> str:
    return next(name for floor, name in CONFIDENCE_BANDS if conf >= floor)


# ---------------------------------------------------------------------------
# METRICS - the chrF numbers notebook 04 produced, shown beside the live output
# ---------------------------------------------------------------------------

METRIC_CANDIDATES = [
    HERE / "metrics" / "evaluation_results.csv",
    HERE.parent / "artifacts" / "data" / "evaluation_results.csv",
]


def load_metrics() -> dict:
    for path in METRIC_CANDIDATES:
        if not path.exists():
            continue
        with open(path, encoding="utf-8-sig", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("direction")]
        systems = [c.split(" ")[0] for c in (rows[0] if rows else {})
                   if c.endswith(" chrF")]
        return {"source": path.name, "systems": systems, "rows": rows,
                "complete": "mixed" in systems}
    return {"source": None, "systems": [], "rows": [], "complete": False}


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MOCK_MODE:
        import torch
        registry.device = "cuda" if torch.cuda.is_available() else "cpu"
        registry.dtype = torch.float16 if registry.device == "cuda" else torch.float32
        log.info("device=%s dtype=%s", registry.device, registry.dtype)
        if registry.device == "cpu":
            log.warning("no GPU visible - expect several seconds per sentence per system")
        # Only warn when a token is actually needed. A repo id that resolves to a
        # directory on this machine needs no authentication, and warning anyway
        # sends people hunting for a token problem that does not exist - which is
        # exactly what happened the first time this ran on the GPU node.
        needs_token = [k for k, v in SYSTEMS.items()
                       if v["private"] and not Path(v["repo"]).is_dir()]
        if needs_token and not HF_TOKEN:
            log.warning("HF_TOKEN is unset and these are private Hub repos, so "
                        "they will 401: %s", ", ".join(needs_token))
        for k, v in SYSTEMS.items():
            if Path(v["repo"]).is_dir():
                log.info("%s loads from local disk (%s) - no token needed", k, v["repo"])
        if PRELOAD:
            for name in SYSTEMS:
                try:
                    await registry.get(name)
                except Exception as exc:      # a missing model must not kill the app
                    registry.errors[name] = str(exc)
                    log.error("could not preload %s: %s", name, exc)
    else:
        log.warning("MOCK_MODE - no weights loaded, outputs are placeholders")
    yield


app = FastAPI(title="Ekegusii PSA translation", lifespan=lifespan)

# ---------------------------------------------------------------------------
# RATE LIMIT
# ---------------------------------------------------------------------------
# A fixed window per IP, held in memory. Not distributed, not exact at window
# boundaries, and it resets on restart - all fine, because the job is to stop
# one script from monopolising the GPU, not to bill anyone.

_hits: dict = {}


def client_ip(request) -> str:
    """
    Behind a tunnel or proxy, request.client.host is 127.0.0.1 for everyone,
    which would rate-limit the whole internet as a single client. Cloudflare
    sets CF-Connecting-IP; most other proxies set X-Forwarded-For.
    """
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limit(request, call_next):
    if RATE_LIMIT_PER_MIN <= 0 or not request.url.path.startswith("/api/"):
        return await call_next(request)
    if request.url.path in {"/api/health", "/api/systems", "/api/metrics"}:
        return await call_next(request)

    window = int(time.time() // 60)
    ip = client_ip(request)
    key = (ip, window)
    _hits[key] = _hits.get(key, 0) + 1
    if len(_hits) > 4096:                      # drop stale windows
        for k in [k for k in _hits if k[1] < window]:
            _hits.pop(k, None)
    if _hits[key] > RATE_LIMIT_PER_MIN:
        return JSONResponse(
            {"detail": f"Rate limit: {RATE_LIMIT_PER_MIN} requests per minute. "
                       f"This demo runs on a single shared GPU."}, status_code=429)
    return await call_next(request)


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    src_lang: str = ENG
    systems: list = Field(default_factory=lambda: list(SYSTEMS))
    beams: int = BEAMS
    max_new_tokens: int = MAX_NEW_TOKENS


@app.get("/api/health")
async def health():
    return {
        "ok": True, **registry.describe(), "errors": registry.errors,
        "pid": os.getpid(),
        "started": STARTED,
        # Exposed so a restart can be verified from outside. Without this, a
        # stale process left holding the port looks identical to a fresh one.
        "enabled_systems": list(SYSTEMS),
        "feedback_repo": FEEDBACK_REPO or None,
        "feedback_durable": bool(FEEDBACK_REPO),
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
    }


@app.get("/api/systems")
async def systems():
    metrics = load_metrics()
    by_system = {}
    for row in metrics["rows"]:
        for s in metrics["systems"]:
            col = f"{s} chrF"
            if row.get(col):
                by_system.setdefault(s, {})[f"{row['direction']} · {row['test set']}"] = \
                    float(row[col])
    return {
        "systems": [
            {"id": k, **{f: v[f] for f in ("label", "note", "role", "repo", "target")},
             "available": k not in registry.errors,
             "error": registry.errors.get(k),
             "chrf": by_system.get(k, {})}
            for k, v in SYSTEMS.items()
        ],
        "source_languages": SOURCE_LANGUAGES,
        "metrics": metrics,
        "runtime": registry.describe(),
        "defaults": {"beams": BEAMS, "max_new_tokens": MAX_NEW_TOKENS,
                     "max_chars": MAX_CHARS},
    }


@app.get("/api/metrics")
async def metrics():
    return load_metrics()


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    if len(req.text) > MAX_CHARS:
        raise HTTPException(413, f"text is longer than MAX_CHARS ({MAX_CHARS})")
    unknown = [s for s in req.systems if s not in SYSTEMS]
    if unknown:
        raise HTTPException(400, f"unknown system(s): {unknown}")
    if req.src_lang not in {l["code"] for l in SOURCE_LANGUAGES}:
        raise HTTPException(400, f"unsupported source language {req.src_lang!r}")

    units = segment(req.text)
    texts = [u for u in units if u is not None]
    if not texts:
        raise HTTPException(400, "nothing to translate")
    if len(texts) > MAX_SEGMENTS:
        raise HTTPException(413, f"{len(texts)} segments exceeds MAX_SEGMENTS "
                                 f"({MAX_SEGMENTS}); send it in smaller pieces")

    beams = max(1, min(req.beams, 8))
    max_new = max(16, min(req.max_new_tokens, 256))

    results = []
    for name in req.systems:
        t0 = time.perf_counter()
        try:
            pieces, confs = await registry.translate(
                name, texts, req.src_lang, beams, max_new)
            # Report the weakest segment, not the average. A paragraph whose
            # first sentence is solid and whose third is a guess should not read
            # as uniformly fine - the guess is the part someone must check.
            worst = min(confs) if confs else 0.0
            results.append({
                "system": name,
                "label": SYSTEMS[name]["label"],
                "target": SYSTEMS[name]["target"],
                "output": rejoin(units, pieces),
                "segments": len(texts),
                "confidence": round(worst, 4),
                "confidence_band": band(worst),
                "per_segment": [round(c, 4) for c in confs],
                "ms": round((time.perf_counter() - t0) * 1000),
                "ok": True,
            })
        except Exception as exc:
            log.exception("%s failed", name)
            results.append({"system": name, "label": SYSTEMS[name]["label"],
                            "ok": False, "error": str(exc)})
    return {"src_lang": req.src_lang, "segments": len(texts), "results": results}


class Feedback(BaseModel):
    src_lang: str
    source: str = Field(min_length=1, max_length=MAX_CHARS)
    machine: str = Field(default="", max_length=MAX_CHARS)
    correction: str = Field(default="", max_length=MAX_CHARS)
    rating: str = ""           # "good" | "usable" | "wrong"
    note: str = Field(default="", max_length=2000)
    system: str = ""


@app.post("/api/feedback")
async def feedback(item: Feedback):
    """
    Record a native speaker's correction.

    This is the most valuable thing the demo produces. Automatic metrics cannot
    tell you whether the Ekegusii reads as a public notice rather than as
    scripture; a Kisii speaker typing the right sentence can, and every
    correction is a new training pair for the next round.
    """
    record = item.model_dump()
    record["received"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = json.dumps(record, ensure_ascii=False)

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

    if not FEEDBACK_REPO:
        # Say so plainly rather than implying it was saved somewhere durable.
        log.warning("FEEDBACK_REPO unset - correction kept only on this container's "
                    "disk, which is wiped on restart")
        return {"ok": True, "stored": "local", "durable": False}

    try:
        from huggingface_hub import upload_file
        await asyncio.to_thread(
            upload_file,
            path_or_fileobj=str(FEEDBACK_FILE), path_in_repo="feedback.jsonl",
            repo_id=FEEDBACK_REPO, repo_type="dataset", token=HF_TOKEN)
        return {"ok": True, "stored": FEEDBACK_REPO, "durable": True}
    except Exception as exc:
        log.exception("could not upload feedback")
        # The correction is still on local disk; do not tell the user it is lost.
        return {"ok": True, "stored": "local", "durable": False, "error": str(exc)}


@app.get("/")
async def index():
    path = STATIC / "index.html"
    if not path.exists():
        return JSONResponse({"error": "static/index.html is missing"}, 500)
    return FileResponse(path)


if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
