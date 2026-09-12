"""
Offline tests for scrape_lughayangu.py.

The page cache below is reconstructed from the REAL scraper output the user
pasted back - every string here actually appeared on lughayangu.com. That makes
this a regression test against the exact failure that produced garbage rows.
"""
import csv, sys, os
sys.path.insert(0, "ekegusii")
import scrape_lughayangu as S

FAILS = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"   -> {detail}"))
    if not cond: FAILS.append(name)

UI = [
    "Define a ekegusii word", "Searching...", "Your voice can preserve this word",
    "Already lending your voice? Log in", "Free, and it takes seconds.",
    "TUMIA KISWAHILI", "Login Via Google", "Need a Ekegusii translation? Get a quote",
]
def page(headword, extra):
    tmpl = (f"Help the community learn how {headword} is really pronounced, and your "
            f"recording will be played for everyone learning Ekegusii, always credited to you.")
    return {"headword": headword, "contributor": "Purity", "date": "September 22, 2022",
            "blocks": UI[:3] + [tmpl] + extra + UI[3:]}

CACHE = {
 "https://lughayangu.com/ekegusii/amanagu": page("Amanagu",
   ["black nightshade", "amanagu ne ching'eni chingiya"]),
 "https://lughayangu.com/ekegusii/chinsaga": page("Chinsaga",
   ["spider plant", "obama agorete chinsaga chia emerongo kianda"]),
 "https://lughayangu.com/ekegusii/chinsiaga": page("Chinsiaga",
   ["napier grass", "omokori egasi atema chinsiaga ekori paka ankio"]),
 "https://lughayangu.com/ekegusii/enderema": page("Enderema",
   ["vine spinach", "enderema ekoiyekwa ne risosa"]),
 "https://lughayangu.com/ekegusii/rinoe": page("Rinoe",
   ["kidney beans", "rinoe riama ange ne ritoke"]),
 "https://lughayangu.com/ekegusii/egesare": page("Egesare",
   ["Cowpeas leaves", "sokoro aanchete egasare",
    "Grandfather loves eating cowpeas leaves"]),
 "https://lughayangu.com/ekegusii/rise": page("Rise",
   ["stinging nettle", "atebetigwe anywe rise risibi amanyinga",
    "She was instructed to an infusion of ground stinging nettle to cleanse her blood",
    "babaete rise buna chingeni", "they gave them stinging nettle as vegatables"]),
 "https://lughayangu.com/ekegusii/risosaamasosa": page("Risosa",
   ["pumpkin leaves", "Synonyms: amasosa", "babaganete risosa",
    "They shared the pumpkin leaves amongt themselves"]),
 "https://lughayangu.com/ekegusii/engano": page("Engano",
   ["wheat", "mama agorete amasi y'engano atato"]),
 "https://lughayangu.com/ekegusii/omote": page("Omote",
   ["Baiyereti omote korosi chibao", "Butora Omote oyo."]),
}

print("=== training LID on the real Bible corpus ===")
lid = S.train_lid("data/bible_en_guz_swh.csv")

print("\n=== boilerplate detection ===")
bp = S.find_boilerplate(CACHE, min_pages=4)
def is_bp(t, hw=""): return S.normalise_for_df(t, hw) in bp
for ui in UI:
    check(f"UI dropped: {ui[:40]}", is_bp(ui), "NOT detected")
check("templated pronounce-copy dropped (headword masked)",
      is_bp("Help the community learn how Amanagu is really pronounced, and your recording "
            "will be played for everyone learning Ekegusii, always credited to you.", "Amanagu"))
check("real content NOT dropped: black nightshade", not is_bp("black nightshade", "Amanagu"))
check("real content NOT dropped: example sentence",
      not is_bp("amanagu ne ching'eni chingiya", "Amanagu"))

print("\n=== extraction ===")
sents, lex, unp = S.extract(CACHE, lid, bp)
print(f"  sentences={len(sents)}  lexicon={len(lex)}  unpaired={len(unp)}")

all_text = [r["english"] for r in sents] + [r["ekegusii"] for r in sents] + \
           [r["english"] for r in lex] + [r["ekegusii"] for r in lex]
for ui in UI:
    check(f"no UI in output: {ui[:34]}", ui not in all_text, "LEAKED")
check("no 'Help the community' anywhere",
      not any("Help the community" in t for t in all_text))
check("no 'Synonyms:' metadata", not any(t.startswith("Synonyms") for t in all_text))
check("no Kiswahili leaked", "TUMIA KISWAHILI" not in all_text)

print("\n  --- lexicon ---")
lexmap = {r["ekegusii"]: r["english"] for r in lex}
for k, v in lexmap.items(): print(f"    {k:<12} = {v}")
for hw, gloss in [("Amanagu","black nightshade"), ("Chinsaga","spider plant"),
                  ("Chinsiaga","napier grass"), ("Enderema","vine spinach"),
                  ("Rinoe","kidney beans"), ("Engano","wheat")]:
    check(f"gloss {hw} = {gloss}", lexmap.get(hw) == gloss, lexmap.get(hw))
check("gloss side is English, not Ekegusii",
      all(lid.predict(r["english"])[0] == "en" for r in lex),
      [r["english"] for r in lex if lid.predict(r["english"])[0] != "en"])

print("\n  --- sentence pairs ---")
for r in sents:
    print(f"    EN : {r['english'][:78]}")
    print(f"    GUZ: {r['ekegusii'][:78]}")
pairs = {(r["ekegusii"], r["english"]) for r in sents}
check("Egesare pair correct",
      ("sokoro aanchete egasare", "Grandfather loves eating cowpeas leaves") in pairs)
check("Risosa pair correct",
      ("babaganete risosa", "They shared the pumpkin leaves amongt themselves") in pairs)
check("Rise pair 1 correct",
      ("atebetigwe anywe rise risibi amanyinga",
       "She was instructed to an infusion of ground stinging nettle to cleanse her blood") in pairs)
check("Rise pair 2 correct",
      ("babaete rise buna chingeni", "they gave them stinging nettle as vegatables") in pairs)
check("every pair has Ekegusii on the guz side",
      all(lid.predict(r["ekegusii"])[0] == "guz" for r in sents),
      [r["ekegusii"] for r in sents if lid.predict(r["ekegusii"])[0] != "guz"])
check("every pair has English on the en side",
      all(lid.predict(r["english"])[0] == "en" for r in sents),
      [r["english"] for r in sents if lid.predict(r["english"])[0] != "en"])

print("\n  --- unpaired (correctly refused, not guessed) ---")
for r in unp[:6]:
    side = r["ekegusii"] or r["english"]
    print(f"    [{r['reason']}] {side[:70]}")
check("Amanagu example refused (no translation on page)",
      any(r["ekegusii"] == "amanagu ne ching'eni chingiya" for r in unp))
check("Omote's two examples refused (0 english)",
      sum(1 for r in unp if r["headword"] == "Omote") == 2,
      [r["ekegusii"] for r in unp if r["headword"] == "Omote"])
check("nothing fabricated: unpaired rows have exactly one side filled",
      all(bool(r["ekegusii"]) != bool(r["english"]) for r in unp))

print("\n" + "=" * 52)
print("FAILED: " + (", ".join(FAILS) if FAILS else "none"))
sys.exit(1 if FAILS else 0)
