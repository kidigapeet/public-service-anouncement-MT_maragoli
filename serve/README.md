# `serve/` — the translation service

One FastAPI app that runs in two shapes: a **public demo** serving the released
model, and an **internal comparison** running all four systems side by side.

| System | What it is |
|---|---|
| `stock` | `facebook/nllb-200-distilled-600M`, asked for `kik_Latn` |
| `stage1` | fine-tuned on the Bible and storybook corpus |
| `stage2` | stage 1, then adapted on Kenyan PSAs with Bible replay — the curriculum, which **lost** |
| `mixed` | all the data in one pass — **the released model**, best on every test set |

The comparison view exists because the paper's finding is a negative one: the
two-stage curriculum was the hypothesis, the single pass was its control, and the
control won by 6.04 chrF2++ on real PSAs. That is a table until someone types
*"HELB loan applications close on 30 September"* and reads four outputs next to
each other.

**On `stock`.** Ekegusii has no token in stock NLLB-200, so it *cannot* be asked
for Ekegusii. Following notebook 04, it is asked for Kikuyu (`kik_Latn`), the
nearest Kenyan Bantu language it does support. That establishes what "no
Ekegusii support" looks like numerically. It is a floor, not a fair comparison,
and the UI says so on the card. Do not let a slide turn it into one.

---

## Two modes

| `ENABLED_SYSTEMS` | What you get |
|---|---|
| `mixed` (default) | **Public demo.** One model, a direction dropdown, a confidence label and a correction form. This is what gets shared. |
| `stock,stage1,stage2,mixed` | **Internal comparison.** Four models side by side plus the chrF chart, for your own use and for the write-up. |

## Publishing the public demo

```bash
python serve/deploy_space.py --space samuelabrha/ekegusii-psa-translator
```

Creates a Docker Space, uploads everything, and prints what is left to do. Two
settings must be added by hand in the Space's **Settings** tab, because neither
belongs on a command line:

- **`HF_TOKEN`** (secret, read-scoped) — without it the Space cannot read the
  private model repository and every translation returns a 401.
- **`FEEDBACK_REPO`** (secret, optional) — a dataset repo id such as
  `samuelabrha/ekegusii-feedback`. Without it, corrections are written to the
  container's disk, which a free Space wipes on every restart.

Free CPU Basic is 2 vCPU / 16 GB. One model is 2.4 GB, so it fits comfortably;
expect a few seconds per sentence, and a slower first request after the Space
has been idle, because the weights are re-downloaded.

### Publishing from the training node instead

```bash
bash serve/run_public_demo.sh
```

Starts the API and opens a Cloudflare quick tunnel, printing a public
`https://<random>.trycloudflare.com` URL (also written to `serve/PUBLIC_URL.txt`).
No Cloudflare account, no root, nothing installed system-wide — it fetches a
single static `cloudflared` binary into `serve/`.

On the training node it finds `artifacts/nllb600m-mixed-control` and loads from
disk, so it starts in seconds, needs no Hugging Face token, and runs on the GPU
at roughly a second per sentence.

**The URL is temporary.** A quick tunnel's hostname is random, changes on every
run, and stops working when the process or the node stops. That satisfies "live
demo"; it does not satisfy "publicly accessible demo" as a lasting deliverable —
for that the service has to outlive the session, which currently means a paid
Space or a machine you control.

`--protocol http2` is set deliberately: the default QUIC path uses UDP 7844,
which university and cloud networks routinely block, and the failure looks like
an unexplained timeout.

Because this exposes an unauthenticated GPU endpoint to the internet,
`RATE_LIMIT_PER_MIN` (default 30) caps requests per client IP. Behind the tunnel
every request arrives from 127.0.0.1, so the limiter reads `CF-Connecting-IP` —
without that it would rate-limit the entire internet as one client.

## Run it locally


### 1. A read token

The three fine-tuned repositories are private. Create a **read**-scoped token at
<https://huggingface.co/settings/tokens>. Not a write token — this service never
uploads anything, and a leaked write token can delete your models.

```bash
cp .env.example .env
# put the token in .env; .env is gitignored, keep it that way
```

### 2. Docker

```bash
docker compose --profile gpu up --build     # a box with an NVIDIA card
docker compose --profile cpu up --build     # anything else
```

Then open <http://localhost:8000>.

The first start downloads ~7 GB of weights. They land in the `hf-cache` volume,
so every later start is fast — don't delete that volume casually.

### 3. Without Docker

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Front-end work with no weights at all

```bash
MOCK_MODE=1 uvicorn app:app --port 8000
```

Every system returns a tagged placeholder. Nothing is downloaded, nothing needs
a GPU, and the whole UI — cards, chart, table view, both themes — is exercised.

---

## What it needs from the machine

| | GPU | CPU |
|---|---|---|
| Memory, one model | ~1.2 GB VRAM (fp16) | ~2.4 GB RAM (fp32) |
| Memory, all four | ~4.8 GB VRAM | ~9.6 GB RAM |
| Per sentence, per system, beam 4 | ~0.3 s | a few seconds |

CPU latency is an estimate until measured on the host you actually deploy to —
check `ms` in the response rather than trusting this row.

If the host has less memory than that, set `MAX_RESIDENT=2`. Models are then
evicted least-recently-used and reloaded on demand: identical output, a pause
when a visitor switches systems. `BEAMS=1` roughly halves CPU latency at a real
cost in quality — only worth it on a CPU-only host.

`BEAMS=4` is the default because it is what notebook 04 evaluated with, so a
number quoted in the paper and a string shown on screen came out of the same
settings. Change it and they no longer correspond.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | the UI |
| `GET` | `/api/health` | liveness, device, which models are resident |
| `GET` | `/api/systems` | system metadata + chrF scores + runtime info |
| `GET` | `/api/metrics` | the parsed `evaluation_results.csv` |
| `POST` | `/api/translate` | `{text, src_lang, systems[], beams, max_new_tokens}` |
| `POST` | `/api/feedback` | `{src_lang, source, machine, correction, rating, note}` |

The front end sends **one request per system, in parallel**, so each card
resolves the moment its own model finishes instead of every card waiting for the
slowest.

```bash
curl -s localhost:8000/api/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Report suspected cholera cases to the nearest health facility.",
       "src_lang":"eng_Latn","systems":["stage1","stage2"]}' | python -m json.tool
```

### Paragraphs

NLLB is a sentence-level model; handing it a whole paragraph makes it drop
clauses. `segment()` splits on line breaks and sentence boundaries, translates
each piece, and rejoins — preserving blank lines, because a PSA's shape carries
meaning. `MAX_SEGMENTS` (default 40) caps how much one request may ask for.

---

## Keeping the chart honest

The chart reads `metrics/evaluation_results.csv`, falling back to
`../artifacts/data/evaluation_results.csv`. Whichever columns exist are the
columns it draws — so until `04_evaluate.ipynb` has been re-run with all three
checkpoints present, the mixed control is absent from the chart and the UI says
so in a notice rather than quietly showing a three-bar chart as if it were
complete.

**After re-running notebook 04, copy the new CSV over `metrics/` and rebuild.**

---

## Before this is public

The weights derive from the Ekegusii Revised Bible (© Bible Society of Kenya)
and PSA data of unconfirmed provenance. The repositories are private for that
reason. Clear publication with Prof. Ombui before putting this on a public
hostname, and keep the footer's post-editing warning: an untouched machine
translation of a health or safety notice is not something to publish.

---

## The confidence label

`/api/translate` returns the geometric-mean per-token probability of the chosen
output — `exp(sequences_scores)` for beam search. The UI bands it into
low / moderate / high.

**It is the model's certainty, not a probability of being correct.** Neural MT is
routinely fluent, confident and wrong, and this model has a 53.6% content-word
vocabulary gap on PSA text, so a confident rendering of `KUCCPS` is exactly the
case the number will not catch. The band thresholds are eyeballed, not calibrated
against accuracy — calibrating them needs the human evaluation that has not been
done. Do not present them as reliability estimates until it has.

## Feedback

`POST /api/feedback` appends to `feedback.jsonl` and, when `FEEDBACK_REPO` is
set, uploads that file to a Hugging Face dataset repo. The response says
`durable: false` when it only reached local disk, and the UI wording changes to
match — a container that is about to be wiped should not tell someone their
correction was saved.

Corrections from Ekegusii speakers are the highest-value output of the whole
deployment: they are the human evaluation the project is missing, and they are
training pairs for the next round.
