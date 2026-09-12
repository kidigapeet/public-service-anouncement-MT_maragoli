#!/usr/bin/env python3
"""
gen_banner.py - build the printed project poster.

Format: A1 LANDSCAPE, 841 x 594 mm. The ISO paper proportion, 1.41:1, so a
print shop takes it without rescaling and it files alongside other posters.

LAYOUT: one narrative, read left to right then top to bottom.

    band 1   who we are, what we did          |  what we trained it on
    band 2   THE RESULT (tinted, full bleed, the heaviest block on the page)
    band 3   experimental setup and limitations, one section, two columns
    band 4   credits, a solid navy bar, not small grey text on white

An earlier draft split the corpus into a table AND a donut side by side, which
said the same thing twice. They are now one object: the donut is the focal
point and its legend carries the counts, so there is a single place to look.

Everything is dimensioned in millimetres and the PDF page is real A1, so what
a printer receives needs no interpretation.

Outputs:
    banner.html   the master
    banner.pdf    A1, vector text - GIVE THE PRINTER THIS
    banner.png    preview raster

Brand colours are sampled from the official USIU-Africa logo, not guessed:
blue #293d94, gold #ffca08.

DELIBERATE OVERRIDE: the dataviz palette validator fails the USIU pair. Its
scope is categorical colours for DATA SERIES, which must sit in a lightness
band and clear 3:1 against the surface. These are brand constants on a large
printed surface, a different problem, and a university's colours are not mine
to "fix". The bars use an emphasis pattern, one highlighted and one grey,
rather than categorical hues, because the story is one comparison repeated.

NO EM DASHES. The script exits non-zero if one reaches the output.
"""
import base64
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
LOGO = base64.b64encode((HERE / "usiu_logo.png").read_bytes()).decode()

PAGE_W, PAGE_H = "841mm", "594mm"
PREVIEW_SCALE = 2

BLUE, GOLD = "#293d94", "#ffca08"
INK, INK2, INK3 = "#111827", "#4b5563", "#9aa1ad"
LINE, TINT = "#e6e8ec", "#f4f6fb"

# Verified by count_records.py, which replays the notebook split logic against
# the real CSVs. These three add to 62,669 exactly. African Storybook rows are
# absent on purpose: merged into the corpus AFTER training, so the released
# model never saw them.
SOURCES = [
    ("Ekegusii Bible", 56866, BLUE),
    ("Kenyan public service announcements", 5692, GOLD),
    ("Everyday sentences", 111, "#7d8598"),
]
TOTAL = sum(n for _, n, _ in SOURCES)
TOTAL_PAIRS = f"{TOTAL:,}"

RESULTS = [
    ("English into Ekegusii",   14.56, 40.97),
    ("Kiswahili into Ekegusii", 14.13, 39.61),
]
SCALE_MAX = max(r[2] for r in RESULTS)

TEAM = ["Samuel Abrha", "Weldesenbet Zeray", "Hetal Kumbharana",
        "Halima Mohammed", "Peter Kidiga", "Mitchelle Moraa"]
SUPERVISOR = "Prof. Edward Ombui"
REPO = "github.com/SamAbr/PSA-MT"

team_html = " &middot; ".join(TEAM)
team_label = "Researchers" if len(TEAM) > 1 else "Researcher"

# ---------------------------------------------------------------- the donut
# Drawn with stroke-dasharray on concentric circles rather than arc paths: one
# number per segment and no trigonometry to get wrong. The everyday-sentences
# slice is 0.2% and is therefore a hairline. That is the honest picture, the
# corpus IS the Bible, so it is not inflated to a minimum visible width.
R = 76
CIRC = 2 * 3.14159265 * R
_off, _segs = 0.0, []
for _name, _n, _c in SOURCES:
    _frac = _n / TOTAL
    _segs.append(f'<circle class="seg" r="{R}" stroke="{_c}" '
                 f'stroke-dasharray="{_frac * CIRC:.2f} {CIRC:.2f}" '
                 f'stroke-dashoffset="{-_off * CIRC:.2f}"></circle>')
    _off += _frac
donut_segs = "\n            ".join(_segs)

legend_html = "".join(
    f'\n            <div class="lg"><i style="background:{c}"></i>'
    f'<span class="nm">{name}</span>'
    f'<b>{n:,}</b><em>{100 * n / TOTAL:.1f}%</em></div>'
    for name, n, c in SOURCES)


def _bar(name, value, colour):
    pct = 100 * value / SCALE_MAX
    return f"""
              <div class="bar">
                <div class="lab">{name}</div>
                <div class="track">
                  <div class="fill" style="width:{pct:.1f}%;background:{colour}"></div>
                  <div class="val" style="left:{pct:.1f}%">{value:.1f}</div>
                </div>
              </div>"""


panels_html = "".join(f"""
          <div class="rpanel">
            <div class="rnum">{rel:.2f}<span> chrF2++</span></div>
            <div class="rlab">{label}</div>
            <div class="rdel">+{rel - stock:.1f} over stock NLLB-200,
              a {100 * (rel - stock) / stock:.0f}% relative gain</div>
            {_bar("Stock NLLB-200", stock, "#d0d4de")}
            {_bar("Our model", rel, BLUE)}
          </div>""" for label, stock, rel in RESULTS)

HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  @page {{ size: {PAGE_W} {PAGE_H}; margin: 0; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: {PAGE_W}; height: {PAGE_H}; }}
  body {{ font-family: Poppins, "DejaVu Sans", sans-serif;
          background: #fff; color: {INK}; display: flex; overflow: hidden; }}

  .rail {{ width: 9mm; flex: none; display: flex; flex-direction: column; }}
  .rail .a {{ flex: 3; background: {BLUE}; }}
  .rail .b {{ flex: 2; background: {GOLD}; }}

  .body {{ flex: 1; display: flex; flex-direction: column; }}
  .pad {{ padding-left: 24mm; padding-right: 26mm; }}

  .k {{ color: {INK3}; font-size: 14.5pt; letter-spacing: .1em;
        text-transform: uppercase; display: block; font-weight: 500; }}

  /* ======================= band 1: what we did ======================= */
  .head {{ display: flex; gap: 24mm; align-items: flex-start;
           padding-top: 24mm; padding-bottom: 18mm; flex: 1.2; }}
  .head .left {{ flex: 1.9; }}
  .head .right {{ flex: 1; }}
  .logo {{ height: 30mm; width: auto; display: block; margin-bottom: 13mm; }}
  h1 {{ font-size: 52pt; line-height: 1.1; font-weight: 700;
        letter-spacing: -.022em; color: {BLUE}; }}
  h1 em {{ font-style: normal; color: {INK}; }}
  .sub {{ margin-top: 10mm; font-size: 25pt; line-height: 1.45; color: {INK2}; }}
  .sub b {{ font-weight: 600; color: {INK}; }}
  code {{ font-family: "DejaVu Sans Mono", monospace; font-size: .86em;
          background: {TINT}; padding: 0.5mm 2mm; border-radius: 1.5mm; }}

  /* corpus: ONE object. Donut is the focal point, legend carries the counts. */
  .data {{ border: 1.2px solid {LINE}; border-radius: 4mm; padding: 11mm 12mm 12mm; }}
  .data .k {{ margin-bottom: 8mm; }}
  .donut {{ display: flex; justify-content: center; }}
  .donut svg {{ width: 96mm; height: 96mm; }}
  .seg {{ cx: 100px; cy: 100px; fill: none; stroke-width: 27;
          transform: rotate(-90deg); transform-origin: 100px 100px; }}
  .ctr {{ text-anchor: middle; font-family: Poppins, sans-serif; }}
  .ctr .n {{ font-size: 28px; font-weight: 700; fill: {INK}; }}
  .ctr .t {{ font-size: 13.5px; fill: {INK3}; }}
  .lg {{ display: flex; align-items: baseline; gap: 3.5mm; font-size: 16pt;
         color: {INK2}; margin-top: 6mm; }}
  .lg i {{ width: 5mm; height: 5mm; border-radius: 1.2mm; flex: none;
           align-self: center; }}
  .lg .nm {{ flex: 1; }}
  .lg b {{ color: {INK}; font-weight: 700; font-variant-numeric: tabular-nums; }}
  .lg em {{ font-style: normal; color: {INK3}; width: 17mm; text-align: right;
            font-variant-numeric: tabular-nums; }}

  /* ==================== band 2: the result, the climax ==================== */
  .band {{ background: {TINT}; border-top: 1.2px solid #dfe3ee;
           border-bottom: 1.2px solid #dfe3ee;
           padding-top: 19mm; padding-bottom: 21mm; }}
  .band .k {{ margin-bottom: 12mm; }}
  .results {{ display: flex; gap: 26mm; }}
  .rpanel {{ flex: 1; }}
  .rnum {{ font-size: 92pt; font-weight: 700; letter-spacing: -.045em;
           line-height: 1; color: {BLUE}; }}
  .rnum span {{ font-size: 27pt; font-weight: 600; letter-spacing: -.01em; }}
  .rlab {{ margin-top: 5mm; font-size: 25pt; font-weight: 600; color: {INK}; }}
  .rdel {{ margin-top: 3mm; font-size: 19pt; color: {INK2}; line-height: 1.4; }}
  .bar {{ display: flex; align-items: center; gap: 6mm; margin-top: 8mm; }}
  .bar .lab {{ width: 52mm; font-size: 18pt; color: {INK2};
               text-align: right; flex: none; }}
  .bar .track {{ flex: 1; height: 16mm; position: relative; margin-right: 24mm; }}
  .bar .fill {{ height: 100%; border-radius: 0 1.5mm 1.5mm 0; }}
  .bar .val {{ position: absolute; top: 50%; transform: translateY(-50%);
               font-size: 24pt; font-weight: 700; color: {INK};
               padding-left: 5mm; }}

  /* ============ band 3: setup and limitations, one section ============ */
  .setup {{ padding-top: 16mm; padding-bottom: 14mm; flex: 1;
           display: flex; flex-direction: column; justify-content: center; }}
  .setup .k {{ margin-bottom: 9mm; }}
  .cols {{ display: flex; gap: 26mm; }}
  .col {{ flex: 1; font-size: 22pt; line-height: 1.5; color: {INK2}; }}
  .col b {{ color: {INK}; font-weight: 600; display: block; margin-bottom: 4mm;
            font-size: 24pt; }}

  /* ==================== band 4: one credit bar ==================== */
  .strip {{ background: {BLUE}; color: #fff; display: flex; gap: 20mm;
            align-items: flex-start; padding-top: 11mm; padding-bottom: 11mm;
            font-size: 18pt; line-height: 1.4; }}
  .strip .kk {{ color: {GOLD}; font-size: 13pt; letter-spacing: .1em;
                text-transform: uppercase; display: block; margin-bottom: 3mm;
                font-weight: 500; }}
  .strip b {{ font-weight: 600; }}
  .strip .who {{ flex: 2.3; }}
  .strip .one {{ flex: none; }}
  .strip .repo {{ flex: 1; text-align: right; color: #cfd7f2; font-size: 16.5pt; }}
</style></head>
<body>
  <div class="rail"><div class="a"></div><div class="b"></div></div>
  <div class="body">

    <div class="head pad">
      <div class="left">
        <img class="logo" src="data:image/png;base64,{LOGO}" alt="USIU-Africa">
        <h1>Teaching a Translation Model<br>a Language It Never Knew:<br>
            <em>Ekegusii for Kenyan Public Service Announcements</em></h1>
        <p class="sub">
          NLLB-200 supports 200 languages. <b>Ekegusii is not one of them.</b>
          Using <b>transfer learning</b>, we added <code>guz_Latn</code> to the
          model and fine-tuned it on <b>62,669 parallel sentence pairs</b>, so
          that Kenyan public service announcements can reach 2.7 million more
          speakers.
        </p>
      </div>
      <div class="right">
        <div class="data">
          <span class="k">What it learned from</span>
          <div class="donut">
            <svg viewBox="0 0 200 200">
            {donut_segs}
              <g class="ctr">
                <text class="n" x="100" y="96">{TOTAL_PAIRS}</text>
                <text class="t" x="100" y="117">sentence pairs</text>
              </g>
            </svg>
          </div>{legend_html}
        </div>
      </div>
    </div>

    <div class="band">
      <div class="pad">
        <span class="k">Result: chrF2++ on held out public service announcements</span>
        <div class="results">{panels_html}
        </div>
      </div>
    </div>

    <div class="setup pad">
      <span class="k">Experimental setup and limitations</span>
      <div class="cols">
        <div class="col"><b>How it was measured</b>
          chrF2++ on 570 English and 371 Kiswahili held out announcements.
          Stock NLLB-200 cannot produce Ekegusii and was asked for the nearest
          language it supports, so its bar is a floor rather than a baseline.</div>
        <div class="col"><b>What was held back</b>
          2,993 scripture pairs, 200 everyday sentences and 944 public service
          announcements were never trained on. A further 27,575 pairs from a
          fourth source were discarded as duplicates of text already held.</div>
      </div>
    </div>

    <div class="strip pad">
      <div class="who"><span class="kk">{team_label}</span><b>{team_html}</b></div>
      <div class="one"><span class="kk">Supervisor</span><b>{SUPERVISOR}</b></div>
      <div class="one"><span class="kk">Programme</span>Natural Language Processing<br>
           School of Science &amp; Technology, 2026</div>
      <div class="repo"><span class="kk">Code and model</span>{REPO}</div>
    </div>

  </div>
</body></html>"""

(HERE / "banner.html").write_text(HTML, encoding="utf-8")
print(f"banner.html  {len(HTML):,} bytes")

body = HTML.split("<body>")[1]
bad = {c: body.count(c) for c in "—–−" if body.count(c)}
if bad:
    sys.exit(f"dash characters reached the output: {bad}")

RENDER = f"""
import asyncio
from playwright.async_api import async_playwright

MM_PX = 96 / 25.4          # CSS px per mm
W = round(841 * MM_PX)
H = round(594 * MM_PX)

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={{"width": W, "height": H}},
                              device_scale_factor={PREVIEW_SCALE})
        await pg.goto("file://{HERE}/banner.html", wait_until="networkidle")
        await pg.screenshot(path="{HERE}/banner.png")
        await pg.pdf(path="{HERE}/banner.pdf", width="{PAGE_W}", height="{PAGE_H}",
                     print_background=True,
                     margin={{"top":"0","bottom":"0","left":"0","right":"0"}})
        await b.close()
        print(f"viewport {{W}} x {{H}} css px")

asyncio.run(main())
"""
subprocess.run([sys.executable, "-c", RENDER], check=True)
for f in ("banner.pdf", "banner.png"):
    p = HERE / f
    print(f"{f:<12} {p.stat().st_size / 1024:>8.0f} KB")
print("page size    A1 landscape, 841 x 594 mm (ISO, 1.41:1)")
