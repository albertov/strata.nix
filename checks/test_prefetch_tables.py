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
    "UD-Q5_K_XL": (
        f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV_UNSLOTH}/UD-Q5_K_XL/",
        158_286_406_650,
        [f"Qwen3.8-Flash-Next-UD-Q5_K_XL-0000{i}-of-00006.gguf" for i in range(1, 7)],
    ),
    "UD-Q6_K_XL": (
        f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV_UNSLOTH}/UD-Q6_K_XL/",
        169_165_382_688,
        [f"Qwen3.8-Flash-Next-UD-Q6_K_XL-0000{i}-of-00006.gguf" for i in range(1, 7)],
    ),
    "Q8_0": (
        f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV_UNSLOTH}/Q8_0/",
        188_225_033_248,
        [f"Qwen3.8-Flash-Next-Q8_0-0000{i}-of-00006.gguf" for i in range(1, 7)],
    ),
}

# The NVFP4 family: ready-made packs, not GGUF shards.  model -> (repo, revision, {path: (role, bytes, sha256)}).
# From the fork's setup.py NVFP4_REPOS, checked against the Hub's LFS pointers at those revisions.
REV_NVFP4 = "640a1db97446ddd5e89d2b70ecd0297987192c93"
REV_NVFP4_NVIDIA = "cca8f7fee5ce0af932097340a75e06df25c4e57d"
_NVFP4_TOKENIZER = {
    "pack/tokenizer/chat_template.jinja": ("pack", 8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    "pack/tokenizer/merges.txt": ("pack", 3600844, "080cf95432173729d160346b7882bd2a644b07f7a3813a3eb73f4e66787bfde9"),
    "pack/tokenizer/token_type.json": ("pack", 744960, "5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b"),
    "pack/tokenizer/tokenizer.json": ("pack", 554, "87be2ac47d8bc7393b7435df6ccd276740019a393941a2585d5edce59f450c54"),
    "pack/tokenizer/vocab.json": ("pack", 5737005, "4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4"),
}
_NVFP4_SHARED = {
    "ple-fp8.gguf": ("ple", 51_200_246_176, "37f89b8ce5a3425932763257c40229ec02723ad2ab14b4bbfc26a486f4725c9c"),
    "token-embd-bf16.gguf": ("embd", 1_271_398_656, "386203af9070c7b43acb24ded8922876721d97bbad1a709b25cb49d694a7f51d"),
}
READY = {
    "NVFP4": ("Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata", REV_NVFP4, {
        "pack/dense.bin": ("pack", 1_538_035_200, "974df6dcefcae85e3fd2c19fcc12eeaacaadde8261ab732d5af5af6ce2e25a7a"),
        "pack/experts.bin": ("pack", 67_948_118_016, "325e5e0d85c262a8d8278daf73b20696ec877d9a3ff584e57391fa69ad10bc64"),
        "pack/index.txt": ("pack", 93663, "f62dc71e1c74a72794db54d3917bc71131cc6eae6f3dcdc3fdada48ff70f93ab"),
        "pack/native_experts.txt": ("pack", 1599, "615f655fb28bf60b243664860b135710bf4177b111c6be89073e780a35de1018"),
        **_NVFP4_TOKENIZER,
        "qwen-nvfp4-gptq-dense.gguf": ("dense", 5_991_569_472, "abdf74336b53d8f7ca3d99947f1eac8f19b8a544966750127cb61e561b127e3b"),
        **_NVFP4_SHARED,
    }),
    "NVFP4-NVIDIA": ("Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata", REV_NVFP4_NVIDIA, {
        "pack/dense.bin": ("pack", 1_538_625_024, "1f1c2422069de292673ae6cee7d69f7090732dec69f3f125612a0de456279714"),
        "pack/experts.bin": ("pack", 67_948_118_016, "bb42080bb7585505b64d091d5a734c05376bbd6171406fa9688a46fbc261b957"),
        "pack/index.txt": ("pack", 117490, "17793d1c5b187a73d3c1dc1677af4d4c9b23845394dd7e88bfa53396b41b77e3"),
        "pack/native_experts.txt": ("pack", 1590, "d7f8b4372bb122f3d711030e5d522bff1c8746ac7d133a59a2cc7277283e59b6"),
        **_NVFP4_TOKENIZER,
        "qwen-nvidia-nvfp4-dense.gguf": ("dense", 5_992_176_896, "0f795ca0cbb908e0cf2b894b30103e21455b5a05346a7ef79da208540cfd2be3"),
        **_NVFP4_SHARED,
    }),
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

# ---- the NVFP4 family: ready-made packs (no GGUF shards, nothing packed locally) ---------------
for model, (repo, revision, files) in READY.items():
    check(model in prefetch.QUANTS, f"{model}: not in QUANTS")
    if model not in prefetch.QUANTS:
        continue
    q = prefetch.QUANTS[model]
    check(q.get("families") == ("nvfp4",), f"{model}: families should be ('nvfp4',), got {q.get('families')}")
    check(q.get("ready") is True, f"{model}: it is a ready-made pack, mark it ready")
    check(q.get("budget") is True, f"{model}: its 68 GB of experts need the RAM budget, mark it budget")
    check(q.get("arena_gb") == 68.0, f"{model}: arena_gb = {q.get('arena_gb')}, upstream says 68.0")
    check(q.get("ram_gb") == 80, f"{model}: ram_gb = {q.get('ram_gb')}, upstream says 80")

    ready_table = getattr(prefetch, "READY", {}).get(model)
    check(ready_table is not None, f"{model}: no READY entry")
    if ready_table is not None:
        check(ready_table.get("repo") == repo, f"{model}: repo {ready_table.get('repo')!r}, want {repo!r}")
        check(ready_table.get("revision") == revision,
              f"{model}: revision {ready_table.get('revision')!r}, want {revision!r}")

    ready_files = getattr(prefetch, "ready_files", None)
    if ready_files is None:
        check(False, f"{model}: prefetch.ready_files is missing")
        continue
    got = {f[3]: (f[0], f[4], f[5]) for f in ready_files(model)}
    check(got == files, f"{model}: files differ\n     got  {got}\n     want {files}")
    for f in ready_files(model):
        check(f[1] == repo and f[2] == revision, f"{model}/{f[3]}: repo/revision {f[1]}@{f[2]}")
        check(prefetch.ready_url(f) == f"https://huggingface.co/{repo}/resolve/{revision}/{f[3]}",
              f"{model}/{f[3]}: bad URL {prefetch.ready_url(f)}")

    ready_paths = getattr(prefetch, "ready_paths", None)
    if ready_paths is None:
        check(False, f"{model}: prefetch.ready_paths is missing")
        continue
    paths = ready_paths(model, Path("/models"))
    root = Path("/models") / repo.replace("/", "--")
    dense = next(p for p, (role, _, _) in files.items() if role == "dense")
    check(paths.get("pack") == root / "pack", f"{model}: pack is {paths.get('pack')}, want {root / 'pack'}")
    check(paths.get("dense") == root / dense, f"{model}: dense is {paths.get('dense')}, want {root / dense}")
    check(paths.get("ple") == root / "ple-fp8.gguf", f"{model}: ple is {paths.get('ple')}")
    check(paths.get("embd") == root / "token-embd-bf16.gguf", f"{model}: embd is {paths.get('embd')}")

    ready_model_args = getattr(prefetch, "ready_model_args", None)
    if ready_model_args is None:
        check(False, f"{model}: prefetch.ready_model_args is missing")
    else:
        check(ready_model_args(paths) == [
            "--pack", str(root / "pack"),
            "--native", str(root / dense), "--native-dense-gguf", str(root / dense),
            "--ple-gguf", str(root / "ple-fp8.gguf"),
            "--embd-gguf", str(root / "token-embd-bf16.gguf"),
        ], f"{model}: bad engine arguments {ready_model_args(paths)}")

# the PLE table and the token embedding are the same files in both repositories: a second install links them
for name in ("ple-fp8.gguf", "token-embd-bf16.gguf"):
    shas = {files[name][2] for (_, _, files) in READY.values()}
    check(len(shas) == 1, f"{name}: the NVFP4 repositories should publish the same file, got {shas}")

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
    if fam.get("repo") is not None:      # ready-made families pin one revision per model in READY
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


def budget(quant: str, ram: float):
    """The budget, or None when the quant is not in the table (so the failure names it)."""
    q = prefetch.QUANTS.get(quant)
    return None if q is None else prefetch.resident_budget_gib(q, ram)


check(budget("UD-Q5_K_XL", 128) == 91,
      "UD-Q5_K_XL on 128 GB: at most all 91 GiB of its experts")
check(budget("Q8_0", 128) == 104,
      "Q8_0 on 128 GB: 104 of its 119 GiB of experts in RAM, the rest read from the SSD")
check(budget("Q8_0", 192) == 119,
      "Q8_0 on 192 GB: the budget is capped at all 119 GiB of its experts (every one in RAM)")
check(budget("NVFP4", 80) == 56,
      "NVFP4 on 80 GB: 56 of its 63 GiB of experts in RAM while the budget mode applies")
check(budget("NVFP4", 96) == 63,
      "NVFP4 on 96 GB: the budget is capped at all 63 GiB of its experts")


def budget_needed(quant: str, ram: float):
    """Upstream's rule: all experts in RAM when they and 16 GB more fit; else a RAM budget."""
    fn = getattr(prefetch, "budget_needed", None)
    q = prefetch.QUANTS.get(quant)
    return None if fn is None or q is None else fn(q, ram)


check(budget_needed("UD-Q4_K_XL", 64) is True, "UD-Q4_K_XL on 64 GB: its 71 GiB of experts do not fit with 16 GB spare")
check(budget_needed("UD-Q4_K_XL", 96) is False, "UD-Q4_K_XL on 96 GB: every expert fits with 16 GB spare")
check(budget_needed("Q8_0", 128) is True, "Q8_0 on 128 GB: its 119 GiB of experts need the budget")
check(budget_needed("Q8_0", 136) is False, "Q8_0 on 136 GB: every expert fits with 16 GB spare, as upstream says")
check(budget_needed("NVFP4", 64) is True, "NVFP4 on 64 GB: its 63 GiB of experts need the budget")
check(budget_needed("NVFP4", 80) is False, "NVFP4 on 80 GB: every expert fits with 16 GB spare, as upstream says")

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
    # the NVFP4 packs keep the vision tower unquantized and read pictures with the original's encoder
    "nvfp4": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
              907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
}

for family, (repo, name, size, sha) in ENCODERS.items():
    rev = {"ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": REV_QWEN,
           "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": REV_SWIFT,
           "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": REV_CODER}[repo]
    fam = prefetch.FAMILIES.get(family)
    check(fam is not None, f"{family}: not in FAMILIES")
    if fam is None:
        continue
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

# ---- how big a picture the encoder may see (setup.py's --vision-tokens, its #625) ----------------
# One image token is patch_size 16 x spatial_merge_size 2 = 32 x 32 px in this mmproj (read from its
# GGUF header), so max_tokens is the picture's pixel budget: 1024 is 1.05 Mpx, which shrinks a
# 2032x824 screenshot (1.67 Mpx) 1.26x and makes a line of 20 px text unreadable.  Upstream recommends
# 1024 and caps nothing ("more than the encoder's default needs more VRAM and context per picture"),
# and the encoder warms up AT the cap - the VRAM is spent whether or not a picture is ever sent, and
# the engine, which starts after it, sizes its expert cache from what is left.
check(prefetch.vision_section("a", "b", "c")["max_tokens"] == prefetch.VISION_IMAGE_TOKENS,
      "with no cap asked for, serve gets the device default")
check(prefetch.vision_section("a", "b", "c", 2048)["max_tokens"] == 2048,
      f"the cap must reach serve: vision_section(a, b, c, 2048) is {prefetch.vision_section('a', 'b', 'c', 2048)}")


def parses(argv: list):
    """The parser, but a rejected flag is a failed check instead of a SystemExit out of the harness."""
    try:
        return prefetch.parser().parse_args(argv)
    except SystemExit:
        return None


BASE = ["--family", "unsloth", "--model", "UD-IQ4_XS", "--data-dir", "/tmp/x"]
asked = parses(BASE + ["--vision", "--vision-exe", "/nix/store/x/bin/strata-vision",
                       "--vision-tokens", "2048"])
check(asked is not None and getattr(asked, "vision_tokens", None) == 2048,
      "--vision-tokens N is how the module says how big a picture the encoder may see")
plain = parses(BASE)
check(plain is not None and getattr(plain, "vision_tokens", "MISSING") is None,
      "no --vision-tokens: prefetch's own default, so an unchanged service keeps upstream's 1024")
nv = parses(["--family", "nvfp4", "--model", "NVFP4", "--data-dir", "/tmp/x"])
check(nv is not None and nv.family == "nvfp4", "--family nvfp4 --model NVFP4 must parse")
nv2 = parses(["--family", "nvfp4", "--model", "NVFP4-NVIDIA", "--data-dir", "/tmp/x"])
check(nv2 is not None and nv2.model == "NVFP4-NVIDIA", "--family nvfp4 --model NVFP4-NVIDIA must parse")


# ---- extraEngineArgs: the flag form the module writes is the form the parser reads ---------------
# The two halves live in different languages, so both are pinned here.  A bare `--extra-arg --flag`
# is what the service on albertows3 was started with on 2026-10-12, and argparse answers "expected
# one argument": a value that starts with -- reads as another flag, so systemd looped on a failed
# preStart until the option was commented out.
mod = Path(os.environ["STRATA_MODULE"]).read_text(encoding="utf-8")

# ---- every Unsloth file the prefetch ships is selectable in the NixOS module --------------------
model_enum = re.search(r"model = lib\.mkOption \{\s*type = lib\.types\.enum \[(.*?)\];", mod, re.S)
check(model_enum is not None, "nix/module.nix: model = lib.mkOption with a types.enum list not found")
unsloth_models = re.search(r"unslothModels = \[(.*?)\];", mod, re.S)
check(unsloth_models is not None, "nix/module.nix: unslothModels list not found")
nvfp4_models = re.search(r"nvfp4Models = \[(.*?)\];", mod, re.S)
check(nvfp4_models is not None, "nix/module.nix: nvfp4Models list not found")
for quant in UNSLOTH:
    if model_enum is not None:
        check(re.search(rf'"{re.escape(quant)}"', model_enum.group(1)) is not None,
              f"{quant}: nix/module.nix's model enum must offer it")
    if unsloth_models is not None:
        check(re.search(rf'"{re.escape(quant)}"', unsloth_models.group(1)) is not None,
              f"{quant}: nix/module.nix's unslothModels list must include it")
for quant in READY:
    if model_enum is not None:
        check(re.search(rf'"{re.escape(quant)}"', model_enum.group(1)) is not None,
              f"{quant}: nix/module.nix's model enum must offer it")
    if nvfp4_models is not None:
        check(re.search(rf'"{re.escape(quant)}"', nvfp4_models.group(1)) is not None,
              f"{quant}: nix/module.nix's nvfp4Models list must include it")

check(re.search(r'--extra-arg=\$\{', mod) is not None,
      "nix/module.nix must write each extra engine argument as --extra-arg=${...}, not --extra-arg ${...}")

# the same for the encoder's picture cap: the option is Nix, the flag is argparse, and only this file
# reads both, so a typo on one side is caught here rather than on a machine that boots the service
check(re.search(r'--vision-tokens \$\{toString cfg\.imageTokens\}', mod) is not None,
      "nix/module.nix must pass imageTokens on as --vision-tokens ${toString cfg.imageTokens}")
check(re.search(r'\(cfg\.images && cfg\.imageTokens != null\)', mod) is not None,
      "imageTokens goes over only with images on: a cap without --vision would be a silent no-op")

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
