"""Offline correctness tests for build_trilingual_corpus.py (no network)."""
import io, os, re, sys, zipfile, shutil, subprocess
import pandas as pd
sys.path.insert(0, "ekegusii")
import build_trilingual_corpus as B

FAILS = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"   -> {detail}"))
    if not cond: FAILS.append(name)

print("=== 1. clean_usfm ===")
c = B.clean_usfm
check("footnote span removed with contents",
      "Adam" not in c(r"In the beginning\f + \fr 1:1 \ft Or Adam\f* God created."),
      c(r"In the beginning\f + \fr 1:1 \ft Or Adam\f* God created."))
check("cross-ref span removed",
      c(r"Text here\x - \xo 1:1 \xt Mat 1:1\x* more.") == "Text here more.",
      c(r"Text here\x - \xo 1:1 \xt Mat 1:1\x* more."))
check("word markup unwrapped",
      c(r'\w God|strong="H430"\w* created') == "God created",
      c(r'\w God|strong="H430"\w* created'))
check("strongs attribute residue gone",
      "strong" not in c(r'The \w LORD|strong="H3068"\w* spoke.'),
      c(r'The \w LORD|strong="H3068"\w* spoke.'))
check("nested char markers unwrapped",
      c(r"the \nd LORD\nd* said") == "the LORD said", c(r"the \nd LORD\nd* said"))
check("pilcrow + whitespace normalised",
      c("¶ Now   the  earth was") == "Now the earth was",
      c("¶ Now   the  earth was"))
check("unterminated footnote truncated",
      c(r"Real text\f + \fr 9:9 dangling") == "Real text",
      c(r"Real text\f + \fr 9:9 dangling"))
check("empty in empty out", c("") == "" and c(None) == "")

print("\n=== 2. parse_usfm_zip ===")
def mkzip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, t in files.items(): z.writestr(n, t)
    buf.seek(0)
    return zipfile.ZipFile(buf)

GEN = """\\id GEN
\\c 1
\\v 1 In the beginning God created the heavens and the earth.
\\p
\\v 2 Now the earth was formless and void,
and darkness covered the deep.
\\v 3-4 And God said, Let there be light. And there was light.
\\c 2
\\v 1 Thus the heavens and the earth were completed.
"""
v, s = B.parse_usfm_zip(mkzip({"01GEN.usfm": GEN}), "test")
check("verse count", len(v) == 4, list(v))
check("keys formed correctly", set(v) == {"GEN_1_1","GEN_1_2","GEN_1_3","GEN_2_1"}, set(v))
check("multi-line verse joined",
      v["GEN_1_2"] == "Now the earth was formless and void, and darkness covered the deep.",
      v["GEN_1_2"])
check("range span recorded", s["GEN_1_3"] == (3, 4), s.get("GEN_1_3"))
check("single verse span", s["GEN_1_1"] == (1, 1), s.get("GEN_1_1"))
check("chapter boundary respected", v["GEN_2_1"].startswith("Thus"), v.get("GEN_2_1"))
check("structural \\p not leaked", "\\p" not in v["GEN_1_1"] + v["GEN_1_2"])

print("\n=== 3. align_three_way: span mismatch protection ===")
EN_RANGE = "\\id GEN\n\\c 1\n\\v 1 Alpha one.\n\\v 2-3 Beta and gamma together here.\n"
XX_SPLIT = "\\id GEN\n\\c 1\n\\v 1 Alpha one.\n\\v 2 Beta only.\n\\v 3 Gamma only.\n"
parsed = {
    "english":  B.parse_usfm_zip(mkzip({"a.usfm": EN_RANGE}), "en"),
    "ekegusii": B.parse_usfm_zip(mkzip({"a.usfm": XX_SPLIT}), "guz"),
    "swahili":  B.parse_usfm_zip(mkzip({"a.usfm": XX_SPLIT}), "swh"),
}
st = {}
df = B.align_three_way(parsed, st)
check("mismatched span dropped", st["dropped_span_mismatch"] == 1, st)
check("only clean verse survives", list(df["ref"]) == ["GEN 1:1"], list(df["ref"]))

print("\n=== 4. align_three_way: happy path ===")
def bible(texts):
    body = "\\id GEN\n\\c 1\n" + "".join(f"\\v {i+1} {t}\n" for i, t in enumerate(texts))
    return B.parse_usfm_zip(mkzip({"a.usfm": body}), "x")
parsed = {
    "english":  bible(["In the beginning God created the heavens and the earth.",
                       "The earth was formless and empty and dark.",
                       "And God said let there be light in the world."]),
    "ekegusii": bible(["Ase ritang'ani Nyasae akonya igoro n'ense yonsi.",
                       "Ense yare tereri na bwomo na omonyoro one.",
                       "Nyasae akaga ekero omosana ogocha ase ense."]),
    "swahili":  bible(["Hapo mwanzo Mungu aliumba mbingu na dunia yote.",
                       "Nayo dunia ilikuwa tupu na giza lilikuwa juu.",
                       "Mungu akasema iwe nuru katika dunia hii yote."]),
}
st = {}
df = B.align_three_way(parsed, st)
df = B.apply_filters(df, st, ["english", "ekegusii", "swahili"])
check("three triples survive", len(df) == 3, len(df))
check("columns present",
      {"ref","book","chapter","verse","english","ekegusii","swahili","source"} <= set(df.columns),
      list(df.columns))
check("word-count columns added", "eng_words" in df.columns and df["eng_words"].iloc[0] == 10,
      df.get("eng_words"))

print("\n=== 5. apply_filters: length ratio catches misalignment ===")
bad = pd.DataFrame([{
    "ref":"GEN 1:1","book":"GEN","chapter":1,"verse":1,
    "english":"A very long English verse that goes on and on and on for many many words indeed truly",
    "ekegusii":"Ase ritang'ani.",       # far too short vs English -> misaligned
    "swahili":"Hapo mwanzo Mungu aliumba mbingu na dunia yote kwa uweza wake mkuu sana leo",
    "source":"bible_ebible"}])
st = {}
out = B.apply_filters(bad, st, ["english","ekegusii","swahili"])
check("ratio outlier removed", len(out) == 0 and st["filters"]["length_ratio"] == 1, st["filters"])

print("\n=== 6. apply_filters: noise / short / dup ===")
rows = [
    {"ref":"a","book":"GEN","chapter":1,"verse":1,"english":"God created the heavens and earth.",
     "ekegusii":"- 1:1 Matayo 3:4","swahili":"Mungu aliumba mbingu na dunia.","source":"x"},   # noise
    {"ref":"b","book":"GEN","chapter":1,"verse":2,"english":"He said.",
     "ekegusii":"Akaga.","swahili":"Akasema.","source":"x"},                                    # too short
    {"ref":"c","book":"GEN","chapter":1,"verse":3,"english":"Let there be light in the world.",
     "ekegusii":"Omosana ogocha ase ense yonsi.","swahili":"Iwe nuru katika dunia hii.","source":"x"},
    {"ref":"d","book":"GEN","chapter":1,"verse":4,"english":"Let there be light in the world.",
     "ekegusii":"Omosana ogocha ase ense yonsi.","swahili":"Iwe nuru katika dunia hii.","source":"x"},  # dup
]
st = {}
out = B.apply_filters(pd.DataFrame(rows), st, ["english","ekegusii","swahili"])
check("noise line dropped", st["filters"]["noise_lines"] == 1, st["filters"])
check("short row dropped", st["filters"]["too_short"] == 1, st["filters"])
check("duplicate dropped", st["filters"]["duplicate_rows"] == 1, st["filters"])
check("one row survives", len(out) == 1, len(out))

print("\n=== 7. harvest_storybooks against real ASP data ===")
src = "/tmp/asp-source"
if not os.path.isdir(src):
    subprocess.run(["git","clone","--depth","1","-q",
                    "https://github.com/global-asp/asp-source.git", src], check=True)
zpath = "/tmp/asp-source-master.zip"
if os.path.exists(zpath): os.remove(zpath)
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(src):
        if ".git" in root: continue
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, src)
            z.write(full, f"asp-source-master/{rel}")

_orig = B.fetch_zip
B.fetch_zip = lambda url, timeout=120: zipfile.ZipFile(zpath)
books = B.harvest_storybooks()
B.fetch_zip = _orig

check("storybook triples produced", len(books) > 0, len(books))
if len(books):
    check("all three languages non-empty",
          bool((books[["english","ekegusii","swahili"]].apply(lambda c: c.str.strip() != "").all().all())))
    check("flagged for review", bool(books["needs_review"].all()))
    check("no markdown images leaked", not books["ekegusii"].str.contains(r"!\[").any())
    check("no page markers leaked", not books["english"].str.contains("##").any())
    r = books.iloc[0]
    print(f"\n  sample page [{r['ref']}]")
    print(f"    EN : {r['english'][:90]}")
    print(f"    GUZ: {r['ekegusii'][:90]}")
    print(f"    SWH: {r['swahili'][:90]}")
    print(f"\n  stories covered: {books['story_id'].nunique()}, pages: {len(books)}")

print("\n" + "="*50)
print("FAILED: " + (", ".join(FAILS) if FAILS else "none"))
sys.exit(1 if FAILS else 0)
