#!/usr/bin/env python3
"""Strata prefetch for NixOS: download, verify and prepare the model files, then emit a run config.

    strata-prefetch --family swift --model IQ2_XS --context 32768 --data-dir /var/lib/strata \
        --emit-config /var/lib/strata/config.json --engine /run/current-system/.../bin/strata

Steps (each skipped when already done):
  1. download the model's shards from Hugging Face (resumable, size- and hash-verified)
  2. build the pack with upstream tools (iq_pack.py: tensors + tokenizer)
  3. fetch and pack the MTP draft layer (~5 GB, from the original Qwen checkpoint)
  4. write prepared.json and, with --emit-config, the engine run config for serve/server.py

No model weights are installed by nix: they live in --data-dir.  A fully prepared
data dir needs no network (the .done marks short-circuit every download).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SHARE = Path(os.environ.get("STRATA_SHARE", "")) if os.environ.get("STRATA_SHARE") else None
if SHARE is None:
    sys.exit("strata-prefetch: STRATA_SHARE is not set (run it through the strata package's wrapper)")

TOOLS = SHARE / "tools"
sys.path.insert(0, str(TOOLS))

HF_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# The revisions upstream setup.py pins (its HF_REVISIONS).  Every file is fetched at its pinned
# revision, where its published SHA-256 (the SHA256 table below) is the file's fingerprint: a repo
# that gets re-uploaded cannot silently change a 50 GB shard under us.
HF_REVISIONS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",        # 2026-09-29
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "b22d729eae29b5796f76fb70f91aef549b9fc52c",   # 2026-09-24
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": "5348543e0147355ac9cbcb031184a3546350988e",  # 2026-09-29
    "unsloth/Qwen3.8-Flash-Next-GGUF": "38bb39ee97821de2c9009abb7e93950eec396e66",                    # 2026-09-30
}

# file names: {q} the quant, {i} the shard number, {n} the shard count (QUANTS["shards"])
FAMILIES = {
    "qwen": {
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        # the image encoder (mmproj): llama.cpp's mtmd file for this model, at the revision this table pins
        "encoder": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
        "sub": "{q}/",
        "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-0000{n}.gguf",
        "shards": 2,
        "tag": "",
        "name": "qwen3.8-flash-next",
        "profile": "expert-profile.bin",
    },
    "swift": {
        "repo": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "encoder": ("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf"),
        "sub": "",
        "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-0000{n}.gguf",
        "shards": 2,
        "tag": "swift-",
        "name": "swift-1.5",
        "profile": "expert-profile.bin",
    },
    "coder": {
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
        "encoder": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF", "mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
        "sub": "{q}/",
        "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-0000{n}.gguf",
        "shards": 2,
        "tag": "coder-",
        "name": "qwen3.8-flash-next-coder",
        "profile": "expert-profile-coder.bin",
    },
    # Unsloth's quantizations of the original model (docs/UNSLOTH_Q4.md, docs/UNSLOTH_Q6.md): one
    # subfolder per quant, and its small Q8_0 projections have to be dequantized for the pack
    # (--compat-bf16, as upstream setup.py's "pack_args").  Its experts stay in the GGUF files: no
    # experts.bin, so --low-ram does not apply; the RAM that is short is covered by the budget below.
    "unsloth": {
        "repo": "unsloth/Qwen3.8-Flash-Next-GGUF",
        # only a requantization of the original model, so it reads images with the original's encoder
        "encoder": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
        "sub": "{q}/",
        "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-0000{n}.gguf",
        "pack_args": ["--compat-bf16"],
        "tag": "unsloth-",
        "name": "qwen3.8-flash-next-ud",
        "profile": "expert-profile.bin",
    }
}

# arena_gb: the routed experts' bytes (decimal GB, as upstream's MODELS table); ram_gb: the RAM the
# size wants.  budget: the experts do not all fit in RAM, so the engine keeps a RAM budget of them
# (--resident-budget-gib) and reads the rest from the SSD; shards: the file count when it is not 2.
QUANTS = {
    "Q2_0": {"arena_gb": 34.0, "ram_gb": 48},
    "IQ2_XS": {"arena_gb": 35.5, "ram_gb": 48},
    "IQ3_XXS": {"arena_gb": 42.9, "ram_gb": 60},
    "IQ3_S": {"arena_gb": 50.3, "ram_gb": 62, "families": ("qwen",)},
    "IQ1_M": {"arena_gb": 23.4, "ram_gb": 32, "families": ("coder",)},
    # Unsloth's 4-bit (experimental upstream): 111.3 GB, 71.7 GiB of experts, 7-8.5 tokens/s on a
    # 64 GB PC with a 12 GB card, where the budget is 40 GiB (docs/UNSLOTH_Q4.md)
    "UD-Q4_K_XL": {"arena_gb": 77.0, "ram_gb": 48, "families": ("unsloth",), "budget": True, "shards": 4,
                   "name": "qwen3.8-flash-next-ud-q4_k_xl"},
    # ~4-bit i-quants (a regular choice upstream since 0.1.39): 93.7 GB, 59.5 GB of experts, all of
    # them in RAM from ~80 GB of RAM on
    "UD-IQ4_XS": {"arena_gb": 59.5, "ram_gb": 48, "families": ("unsloth",), "budget": True, "shards": 3,
                  "name": "qwen3.8-flash-next-ud-iq4_xs"},
    # 6-bit (manual upstream, not in its setup's menu): 169.2 GB in six shards, 101.75 GiB of experts
    # (47 layers Q6_K gate/up + Q8_0 down, layer 2 Q8_0 gate/up) and a 54.4 GB Q8_0 PLE table in its
    # own shard 3.  Measured 78.3 tokens/s with every expert in RAM (docs/UNSLOTH_Q6.md); on a smaller
    # machine the Q4 numbers are the starting point, and the experts are Q6_K: 45% more bytes per
    # element read than Q4_K, and 170-byte PLE rows against IQ4_NL's 90, so the SSD reads are slower
    "UD-Q6_K_XL": {"arena_gb": 109.2, "ram_gb": 48, "families": ("unsloth",), "budget": True, "shards": 6,
                   "name": "qwen3.8-flash-next-ud-q6_k_xl"},
}

# The Unsloth shards' published sizes and SHA-256 at the pinned revision (the Hub's LFS pointers, as
# upstream setup.py's UNSLOTH_SHARDS and docs/UNSLOTH_Q4.md, docs/UNSLOTH_Q6.md).  Shard 1 is 10,946,624
# bytes in all three quantizations, so a size check alone cannot tell them apart.
SHA256 = {
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf": (10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf": (49859583136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf": (49376141504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf": (12087983520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf": (10946624, "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf": (49835229856, "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf": (43836407744, "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf": (10946624, "8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf": (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf": (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf": (49814108512, "0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf": (49419317344, "9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf": (14838313984, "e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f"),
    # The image encoders (setup.py's MMPROJ / FAMILIES[...]["mmproj"]) at the pinned revisions of the
    # repositories that publish them: qwen, coder and unsloth share one file (byte for byte, as the Hub
    # hashes show) and Swift has its own.  The coder's and unsloth's rows live in the same table under
    # this name on purpose: the same file from any of the three repositories must hash the same.
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf": (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
    "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf": (907543520, "cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60"),
}

UNSLOTH_RAM_LEFT_GB = 24   # RAM beside the budget: the OS, the engine, and the file cache the rest reads through


def shard_count(fam: dict, quant: str) -> int:
    return QUANTS[quant].get("shards", fam.get("shards", 2))


def shard_names(fam: dict, quant: str) -> list:
    n = shard_count(fam, quant)
    pattern = QUANTS[quant].get("file", fam["file"])
    return [pattern.format(q=quant, i=i, n=n) for i in range(1, n + 1)]


def shard_url(fam: dict, quant: str, name: str) -> str:
    repo = fam["repo"]
    return f"https://huggingface.co/{repo}/resolve/{HF_REVISIONS[repo]}/{fam['sub'].format(q=quant)}{name}"


def pack_args(fam: dict) -> list:
    return list(fam.get("pack_args", []))


def encoder_url(fam: dict) -> str:
    """The family's image encoder (its mmproj file) at the pinned revision of the repository that
    publishes it - for the Unsloth files and the Coder, the original model's repository."""
    repo, name = fam["encoder"]
    return f"https://huggingface.co/{repo}/resolve/{HF_REVISIONS[repo]}/{name}"


# The image encoder on the GPU (setup.py's VISION["gpu"]): it warms up before the engine - about 1.2 GB
# at 1024 image tokens - so the engine sizes its expert slots around the 700 MiB kept free for it.
VISION_IMAGE_TOKENS = 1024
VISION_RESERVE_MIB = 700


def vision_args() -> list:
    """The engine's arguments for images: accept GENI requests, and keep the encoder's VRAM free."""
    return ["--vision", "--vram-reserve-mib", str(VISION_RESERVE_MIB)]


def vision_section(exe: str, mmproj: str, model: str, max_tokens: int = VISION_IMAGE_TOKENS) -> dict:
    """The serve config's "vision" section (setup.py's cfg["vision"] for a GPU encoder): serve keeps
    one `strata-vision` process alive, sends it `ENC <image> <out>` lines and hands the embeddings it
    writes to the engine as GENI.  `model` is shard 1: the encoder reads the vision start/end tokens
    and the image size from the model itself.  `max_tokens` is the most image tokens one picture
    becomes (serve passes it to strata-vision as --max-tokens)."""
    return {"exe": str(exe), "mmproj": str(mmproj), "model": str(model),
            "gpu": True, "max_tokens": VISION_IMAGE_TOKENS}   # STUB: the argument is not used yet


def resident_budget_gib(quant: dict, ram: float) -> int:
    """The GiB of experts the engine keeps in RAM (--resident-budget-gib): the RAM less 24 GB for the OS,
    the engine and the file cache the other experts are read through, at most all of them, at least 8.
    64 GB: 40, the measured setting for UD-Q4_K_XL (docs/UNSLOTH_Q4.md); the cap (= every expert in RAM)
    is what a big-RAM machine gets, where docs/UNSLOTH_Q6.md leaves the flag out - the same state."""
    all_experts = int(quant["arena_gb"] / 1.073741824)
    return max(8, min(round(ram) - UNSLOTH_RAM_LEFT_GB, all_experts))


def say(msg=""):
    print(msg, flush=True)


def fail(msg, hint=None):
    say(f"[strata-prefetch] ERROR: {msg}")
    if hint:
        say(f"  {hint}")
    sys.exit(1)


def ok(msg):
    say(f"[strata-prefetch] ok: {msg}")


def done(path: Path) -> bool:
    return path.with_name(path.name + ".done").exists()


def mark(path: Path, note: str = ""):
    path.with_name(path.name + ".done").write_text(time.strftime("%Y-%m-%d %H:%M") + ("\n" + note if note else ""),
                                                   encoding="utf-8")


def token_headers() -> dict:
    tok = ""
    for env in HF_TOKEN_ENV:
        if os.environ.get(env):
            tok = os.environ[env]
    return {"Authorization": f"Bearer {tok}", "User-Agent": "strata-nix"} if tok else {"User-Agent": "strata-nix"}


def download(url: str, dst: Path, what: str, token=None) -> None:
    """Resumable download with a size check, like upstream setup.py (a .done mark short-circuits)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what} already present")
        return
    headers = token_headers()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers=headers)
            total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})", "check the network and run again (everything else is kept)")
            time.sleep(5)
    if dst.exists() and total and dst.stat().st_size == total:
        mark(dst)
        ok(f"{what} already present")
        return
    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        try:
            req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 10:
                        last = time.time()
                        size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total else f"{have / 1e6:7.1f} MB"
                        print(f"\r  {what}: {size}   ", end="", flush=True)
            print()
            if not total or have >= total:
                break
        except OSError as e:
            print()
            say(f"  download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}", "check the network and run again (the download resumes)")
    part.replace(dst)
    mark(dst)
    ok(f"{what} downloaded")


def verify_sha256(shard: Path) -> None:
    """The published hash of a shard the Hub gives one for, once per file: the finish mark keeps it, so
    the hash is not recomputed (all of the Unsloth files together are 90-170 GB, ~15 minutes).  A file
    with the wrong hash is deleted, so the next run downloads it again."""
    want = SHA256.get(shard.name)
    if want is None:
        return
    size, sha = want
    d = shard.with_name(shard.name + ".done")
    if d.exists() and f"sha256 {sha}" in d.read_text(encoding="utf-8", errors="replace"):
        return
    if shard.stat().st_size != size:
        fail(f"{shard.name} is {shard.stat().st_size:,} bytes, the Hub has {size:,}",
             "delete it and run again (or check --gguf-dir)")
    say(f"[strata-prefetch] verifying {shard.name} (sha256, {size / 1e9:.0f} GB, a few minutes) ...")
    h = hashlib.sha256()
    with open(shard, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != sha:
        shard.unlink()
        d.unlink(missing_ok=True)
        fail(f"{shard.name}'s SHA-256 is {h.hexdigest()}, not the published {sha}",
             "the file was deleted: run again to download it once more")
    mark(shard, f"sha256 {sha}")
    ok(f"{shard.name} matches its published hash")


def check_shard(shard: Path) -> None:
    """The shard is present and whole: its header must not name tensors running past the file's end."""
    import gguf_reader as G
    if not shard.exists():
        fail(f"missing {shard}")
    try:
        g = G.GGUFFile(shard)
    except (ValueError, struct.error) as e:
        fail(f"{shard.name} is not a whole GGUF shard ({e})", "delete it and run again")
    need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
    if shard.stat().st_size < need:
        fail(f"{shard.name} is short: {shard.stat().st_size:,} of {need:,} bytes",
             "delete it and run again (or put a whole shard into --gguf-dir)")


def run_tool(script: str, *args, env=None) -> None:
    cmd = [sys.executable, str(TOOLS / script), *(str(a) for a in args)]
    say("  > " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        fail(f"{script} failed (exit {r.returncode})")


def mem_total_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) * 1024 / 2**30
    return 0.0


def parser() -> argparse.ArgumentParser:
    """The command line, apart from reading it: the module writes these flags from Nix, and the form of
    --extra-arg (a value that starts with --) is something both sides have to agree on."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="swift", choices=sorted(FAMILIES))
    ap.add_argument("--model", default="IQ2_XS", choices=sorted(QUANTS))
    ap.add_argument("--context", type=int, default=32768, choices=[8192, 32768, 65536, 131072, 262144])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gguf-dir", help="a directory holding the family's shard files already downloaded")
    ap.add_argument("--low-ram", choices=["on", "off"], help="default: on when the experts would not fit the RAM")
    ap.add_argument("--resident-budget-gib", type=int,
                    help="GiB of experts the engine keeps in RAM (the Unsloth files; default: the RAM less 24 GB)")
    ap.add_argument("--vision", action="store_true",
                    help="images too: fetch the family's mmproj and give serve the encoder")
    ap.add_argument("--vision-exe", help="the strata-vision binary to encode images with (with --vision)")
    ap.add_argument("--emit-config")
    ap.add_argument("--engine")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int)
    ap.add_argument("--gpu")
    ap.add_argument("--api-key-file")
    ap.add_argument("--hf-token-file")
    ap.add_argument("--sampling-json")
    ap.add_argument("--mcp-json")
    # --extra-arg=VALUE: argparse takes the rest of the word as the value, so VALUE may start with --
    ap.add_argument("--extra-arg", action="append", default=[])
    return ap


def main() -> int:
    a = parser().parse_args()

    fam, quant = FAMILIES[a.family], QUANTS[a.model]
    allowed = quant.get("families") or tuple(FAMILIES)
    if a.family not in allowed:
        fail(f"{a.model} is only published for: {', '.join(allowed)} (not {a.family})")
    if a.vision and not a.vision_exe:
        fail("--vision needs --vision-exe: the strata-vision binary reads the pictures",
             "the NixOS module passes the bundle's own (package.passthru.vision)")

    data = Path(a.data_dir)
    models = data / "models"
    pack = data / "packs" / (fam["tag"].lower() + a.model.lower())
    mtp = data / "mtp"
    rt = mtp / "rt"

    names = shard_names(fam, a.model)
    if a.gguf_dir:
        gdir = Path(a.gguf_dir)
        shards = [gdir / n for n in names]
        ok(f"using GGUF files from {gdir}")
    else:
        shards = [models / n for n in names]

    # decided before anything is downloaded: a wrong --low-ram for these files should not cost 170 GB first
    ram = mem_total_gb()
    budget = None
    if quant.get("budget"):
        # the experts stay in the GGUF files and the engine reads them in place: no experts.bin, and the
        # RAM that cannot hold all of them is handled by the budget instead of by low-RAM mode
        if a.low_ram == "on":
            fail(f"{a.model} has no low-RAM mode: its experts are read from the GGUF files in place",
                 "--resident-budget-gib says how much of them the RAM holds")
        if a.resident_budget_gib is not None and a.resident_budget_gib <= 0:
            fail("--resident-budget-gib needs N > 0")
        budget = a.resident_budget_gib if a.resident_budget_gib is not None else resident_budget_gib(quant, ram)
        low_ram = "off"
        all_gib = int(quant["arena_gb"] / 1.073741824)
        ok(f"RAM budget: {budget} of the {all_gib} GiB of experts in RAM"
           + (", the rest read from the SSD" if budget < all_gib else ", every one of them"))
    else:
        if a.resident_budget_gib is not None:
            fail(f"--resident-budget-gib is one of the Unsloth files' settings; {a.model} keeps all of its "
                 "experts in RAM (extra engine flags: --extra-arg)")
        low_ram = (a.low_ram or "auto")
        if low_ram == "auto":
            low_ram = "on" if ram < quant["ram_gb"] else "off"

    token = Path(a.hf_token_file).read_text(encoding="utf-8").strip() if a.hf_token_file else None
    if not a.gguf_dir:
        for i, (n, s) in enumerate(zip(names, shards), 1):
            download(shard_url(fam, a.model, n), s, f"{a.family}/{a.model} shard {i}/{len(names)}", token=token)

    shard1 = shards[0]
    for s in shards:
        check_shard(s)      # whole and readable: cheap, so it comes before the hash
        verify_sha256(s)    # the Hub's published hash, once per file

    # the image encoder: 0.9 GB, hashed like a shard, and only wanted when the service takes pictures
    vision = None
    if a.vision:
        enc = fam["encoder"][1]
        mmproj = (gdir if a.gguf_dir and (gdir / enc).is_file() else models) / enc
        if mmproj.parent == models:
            download(encoder_url(fam), mmproj, "vision encoder", token=token)
        verify_sha256(mmproj)
        vision = vision_section(a.vision_exe, mmproj, shard1)
        ok(f"images: {enc} encoded by {a.vision_exe}")

    # the PLE table's shard: found by scanning the GGUF headers (shard 2 for qwen/coder, 1 for swift,
    # a shard of its own - 3 - for the Unsloth 6-bit file)
    import gguf_reader as G
    ple = next((s for s in shards if any(t.name == "per_layer_token_embd.weight" for t in G.GGUFFile(s).tensors)),
               None)
    if ple is None:
        fail("the model has no per_layer_token_embd tensor (is this a Qwen3.8-Flash-Next GGUF?)")

    if not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
        say("[strata-prefetch] building the pack (iq_pack.py, a few minutes) ...")
        run_tool("iq_pack.py", "--gguf", shard1, "--out", pack, *pack_args(fam))
        ok(f"pack built: {pack}")
    else:
        ok(f"pack already built: {pack}")

    if low_ram == "on":
        if not (pack / "experts.bin").exists():
            say("[strata-prefetch] building the experts.bin file (low-RAM mode, a few minutes) ...")
            run_tool("iq_pack.py", "--gguf", shard1, "--out", pack, "--experts-bin")
        ok("low-RAM mode: the engine maps experts from the pack's experts.bin")

    if not (rt / "experts.bin").exists():
        say("[strata-prefetch] fetching and packing the MTP draft layer (~5 GB, once) ...")
        run_tool("mtp_fetch.py", "fetch", "--out", mtp)
        run_tool("mtp_pack.py", "--src", mtp, "--experts", "q2_0", "--out", mtp / "mtp-q2_0.gguf")
        run_tool("mtp_rt.py", "--gguf", mtp / "mtp-q2_0.gguf", "--out", rt)
        if not (rt / "draft_vocab.bin").exists():
            shutil.copyfile(SHARE / "data" / "draft_vocab.bin", rt / "draft_vocab.bin")
        ok(f"MTP ready: {rt}")
    else:
        if not (rt / "draft_vocab.bin").exists():
            shutil.copyfile(SHARE / "data" / "draft_vocab.bin", rt / "draft_vocab.bin")
        ok(f"MTP already prepared: {rt}")

    args = [
        "--pack", str(pack),
        "--native", str(shard1),
        "--ple-gguf", str(ple),
        "--expert-profile", str(SHARE / "data" / fam["profile"]),
        "--expert-cache", "auto",
        "--prefill", "auto",
        "--spec", "4",
        "--spec-min-p", "0.5",
        "--mtp", str(rt),
        "--max-context", str(a.context),
    ]
    if budget is not None:
        args += ["--resident-budget-gib", str(budget)]
    if a.context >= 65536 and low_ram != "on":
        args += ["--kv-resident", "32768"]
    if low_ram == "on":
        args += ["--mmap-experts"]
    if vision:
        args += vision_args()
    args += a.extra_arg

    prepared = {"args": args, "tokenizer": str(pack / "tokenizer"), "model_name": quant.get("name", fam["name"]),
                "context": a.context, "low_ram": low_ram == "on", "budget_gib": budget}
    if vision:
        prepared["vision"] = vision
    (data / f"prepared-{fam['tag'].lower()}{a.model.lower()}.json").write_text(json.dumps(prepared, indent=1),
                                                                              encoding="utf-8")
    ok(f"prepared: pack={pack} mtp={rt} context={a.context}")

    if a.emit_config:
        if not a.engine:
            fail("--emit-config needs --engine")
        cfg = {"exe": a.engine, "args": args, "tokenizer": prepared["tokenizer"],
               "model_name": prepared["model_name"],
               "log": str(data / f"engine-{fam['tag'].lower()}{a.model.lower()}.log")}
        if vision:
            cfg["vision"] = vision          # serve keeps strata-vision alive and routes image parts through it
        if a.host:
            cfg["host"] = a.host
        if a.gpu:
            cfg["gpu"] = a.gpu
        if a.api_key_file:
            cfg["api_key"] = Path(a.api_key_file).read_text(encoding="utf-8").strip()
        if a.sampling_json:
            cfg["sampling"] = json.loads(Path(a.sampling_json).read_text(encoding="utf-8"))
        if a.mcp_json:
            cfg["mcp_servers"] = json.loads(Path(a.mcp_json).read_text(encoding="utf-8"))
        out = Path(a.emit_config)
        out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        if a.api_key_file:
            os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)
        ok(f"config written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
