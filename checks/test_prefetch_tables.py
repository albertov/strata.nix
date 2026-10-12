"""Offline checks of prefetch.py's download tables (no network, no download).

The expected URLs, names, sizes and hashes below come from the Hugging Face API at the revisions
setup.py pins - not from prefetch.py itself:

    GET /api/models/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee.../<quant>
    GET /api/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f920.../<quant>

and from upstream's own tables (setup.py FAMILIES/MODELS/UNSLOTH_SHARDS) and docs
(docs/UNSLOTH_Q4.md, docs/UNSLOTH_Q6.md). Run it through `nix flake check`
(checks/prefetch-tables.nix), which sets STRATA_SHARE to the strata package's share dir.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import prefetch

REV_UNSLOTH = "38bb39ee97821de2c9009abb7e93950eec396e66"
REV_QWEN = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
REV_SWIFT = "b22d729eae29b5796f76fb70f91aef549b9fc52c"
REV_CODER = "5348543e0147355ac9cbcb031184a3546350988e"

# quant -> (pinned folder on huggingface.co, byte total, shard file names)
UNSLOTH = {
    "UD-Q4_K_XL": (
        f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV_UNSLOTH}/UD-Q4_K_XL/",
        111_334_654_784,
        [f"Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf" for i in (1, 2, 3, 4)],
    ),
    "UD-IQ4_XS": (
        f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV_UNSLOTH}/UD-IQ4_XS/",
        93_682_584_224,
        [f"Qwen3.8-Flash-Next-UD-IQ4_XS-0000{i}-of-00003.gguf" for i in (1, 2, 3)],
    ),
    "UD-Q6_K_XL": (
        f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV_UNSLOTH}/UD-Q6_K_XL/",
        169_165_382_688,
        [f"Qwen3.8-Flash-Next-UD-Q6_K_XL-0000{i}-of-00006.gguf" for i in range(1, 7)],
    ),
}

# one shard per ordinary family (the two-shard layout): the {q}/ subfolder (qwen, coder) and Swift's flat layout
ORDINARY = {
    ("qwen", "IQ2_XS"): (
        f"https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/{REV_QWEN}/IQ2_XS/",
        2, "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
    ),
    ("swift", "IQ2_XS"): (
        "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/"
        f"resolve/{REV_SWIFT}/",
        2, "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
    ),
    ("coder", "IQ1_M"): (
        f"https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/{REV_CODER}/IQ1_M/",
        2, "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
    ),
}

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        fails.append(msg)


def url_of(family: str, quant: str, name: str) -> str:
    return prefetch.shard_url(prefetch.FAMILIES[family], quant, name)


def names_of(family: str, quant: str) -> list[str]:
    return prefetch.shard_names(prefetch.FAMILIES[family], quant)


# ---- the Unsloth files: every shard's URL, name, size and SHA-256 ------------------------------
for quant, (folder, total, names) in UNSLOTH.items():
    check(quant in prefetch.QUANTS, f"{quant}: not in QUANTS")
    if quant not in prefetch.QUANTS:
        continue
    q = prefetch.QUANTS[quant]
    check(q.get("families") == ("unsloth",), f"{quant}: families should be ('unsloth',), got {q.get('families')}")
    check(q.get("shards") == len(names), f"{quant}: shards = {q.get('shards')}, the repo has {len(names)}")
    check(q.get("budget"), f"{quant}: needs the RAM budget (--resident-budget-gib), mark it budget")

    check(names_of("unsloth", quant) == names, f"{quant}: shard names {names_of('unsloth', quant)} != {names}")
    for n in names:
        got = url_of("unsloth", quant, n)
        want = folder + n
        check(got == want, f"{quant}: bad URL\n     got  {got}\n     want {want}")
        check(n in prefetch.SHA256, f"{quant}: {n} has no SHA-256 entry")
        if n in prefetch.SHA256:
            size, sha = prefetch.SHA256[n]
            check(re.fullmatch(r"[0-9a-f]{64}", sha) is not None, f"{n}: SHA-256 {sha!r} is not 64 hex")
            check(size > 0, f"{n}: size {size} is not a byte count")

    if all(n in prefetch.SHA256 for n in names):
        got = sum(prefetch.SHA256[n][0] for n in names)
        check(got == total, f"{quant}: shard bytes sum to {got:,}, the repo says {total:,}")
    check("--compat-bf16" in prefetch.pack_args(prefetch.FAMILIES["unsloth"]),
          f"{quant}: Unsloth packs need --compat-bf16 (iq_pack refuses Q8_0 projections without it)")

# ---- the ordinary families: pinned revisions, {q}/ subfolder, two shards -----------------------
for (family, quant), (folder, count, name) in ORDINARY.items():
    got_names = names_of(family, quant)
    check(len(got_names) == count and got_names[0] == name,
          f"{family}/{quant}: {len(got_names)} shards {got_names}, want {count} starting with {name}")
    check(url_of(family, quant, name) == folder + name,
          f"{family}/{quant}: bad URL\n     got  {url_of(family, quant, name)}\n     want {folder + name}")

# every quant belongs to at least one family that exists, and every family has a pinned revision
for quant, q in prefetch.QUANTS.items():
    fams = q.get("families") or tuple(prefetch.FAMILIES)
    check(bool(fams) and all(f in prefetch.FAMILIES for f in fams), f"{quant}: families {fams} unknown")
for family, fam in prefetch.FAMILIES.items():
    check(fam.get("repo") in prefetch.HF_REVISIONS, f"{family}: {fam.get('repo')!r} has no pinned revision")

# ---- the RAM budget for the Unsloth files (docs/UNSLOTH_Q4.md: 64 GB -> 40 GiB) ----------------
check(prefetch.resident_budget_gib(prefetch.QUANTS["UD-Q4_K_XL"], 64) == 40,
      "UD-Q4_K_XL on 64 GB should budget 40 GiB (the measured setting)")
check(prefetch.resident_budget_gib(prefetch.QUANTS["UD-Q4_K_XL"], 96) == 71,
      "UD-Q4_K_XL on 96 GB: at most all 71 GiB of its experts")
check(prefetch.resident_budget_gib(prefetch.QUANTS["UD-Q6_K_XL"], 64) == 40,
      "UD-Q6_K_XL on 64 GB: the Q4 numbers are the starting point")
check(prefetch.resident_budget_gib(prefetch.QUANTS["UD-Q6_K_XL"], 188) == 101,
      "UD-Q6_K_XL on 188 GB: the budget is capped at all 101 GiB of its experts (every one in RAM)")

# ---- images: the family's encoder (mmproj), its pinned URL, size and SHA-256 --------------------
# From setup.py (FAMILIES[...]["mmproj_hf"] and ["mmproj"]) and from the Hub's own answers at those
# revisions: HEAD /resolve/<rev>/<name> replies with x-linked-size and x-linked-etag.
ENCODERS = {
    "qwen": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
             907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
    "swift": ("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf",
              907543520, "cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60"),
    "coder": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF", "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
              907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
    # Unsloth only requantizes the original model, so it reads images with the original's encoder
    "unsloth": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
}

for family, (repo, name, size, sha) in ENCODERS.items():
    rev = {"ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": REV_QWEN,
           "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": REV_SWIFT,
           "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": REV_CODER}[repo]
    fam = prefetch.FAMILIES[family]
    check(tuple(fam.get("encoder", ())) == (repo, name),
          f"{family}: encoder is {fam.get('encoder')!r}, want {(repo, name)}")
    check(prefetch.encoder_url(fam) == f"https://huggingface.co/{repo}/resolve/{rev}/{name}",
          f"{family}: bad encoder URL {prefetch.encoder_url(fam)}")
    check(prefetch.SHA256.get(name) == (size, sha),
          f"{family}: SHA256[{name}] is {prefetch.SHA256.get(name)!r}, the Hub has {(size, sha)}")

# Swift is a different model with its own mmproj; picking the original's would encode images into a
# embedding space the pack's tokenizer and projections do not speak
check(ENCODERS["swift"][1] != ENCODERS["qwen"][1], "Swift ships its own mmproj; the original's is not it")

# ---- the engine's --vision and serve's "vision" section (setup.py's VISION["gpu"] / cfg["vision"]) -
check(prefetch.vision_args() == ["--vision", "--vram-reserve-mib", "700"],
      f"vision args are {prefetch.vision_args()}, want the encoder asked for and its 700 MiB kept free")

check(prefetch.vision_section("/nix/store/x/bin/strata-vision", "/var/lib/strata/models/m.gguf",
                             "/var/lib/strata/models/s1.gguf")
      == {"exe": "/nix/store/x/bin/strata-vision", "mmproj": "/var/lib/strata/models/m.gguf",
          "model": "/var/lib/strata/models/s1.gguf", "gpu": True, "max_tokens": 1024},
      f"the serve config's vision section is {prefetch.vision_section('a', 'b', 'c')}")

# ---- extraEngineArgs: the flag form the module writes is the form the parser reads ---------------
# The two halves live in different languages, so both are pinned here.  A bare `--extra-arg --flag`
# is what the service on albertows3 was started with on 2026-10-12, and argparse answers "expected
# one argument": a value that starts with -- reads as another flag, so systemd looped on a failed
# preStart until the option was commented out.
mod = Path(os.environ["STRATA_MODULE"]).read_text(encoding="utf-8")
check(re.search(r'--extra-arg=\$\{', mod) is not None,
      "nix/module.nix must write each extra engine argument as --extra-arg=${...}, not --extra-arg ${...}")

extra = prefetch.parser().parse_args(["--family", "qwen", "--model", "IQ2_XS", "--data-dir", "/tmp/x",
                                      "--extra-arg=--kv", "--extra-arg=q4_0",
                                      "--extra-arg=--conversation-cache-mib", "--extra-arg=16384"]).extra_arg
check(extra == ["--kv", "q4_0", "--conversation-cache-mib", "16384"],
      f"the module's --extra-arg= form arrives at the parser as {extra}")

if fails:
    print(f"prefetch tables: {len(fails)} FAILED")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("prefetch tables: OK")
