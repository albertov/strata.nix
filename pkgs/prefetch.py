#!/usr/bin/env python3
"""Strata prefetch for NixOS: download, verify and prepare the model files, then emit a run config.

    strata-prefetch --family swift --model IQ2_XS --context 32768 --data-dir /var/lib/strata \
        --emit-config /var/lib/strata/config.json --engine /run/current-system/.../bin/strata

Steps (each skipped when already done):
  1. download the model's files from Hugging Face (resumable, size- and hash-verified): GGUF shards,
     or the whole ready-made pack folder of an NVFP4 model
  2. build the pack with upstream tools (iq_pack.py: tensors + tokenizer; ready-made packs ship it)
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
    },
    # NVFP4's ready-made packs (docs/NVFP4.md): no GGUF shards and nothing is packed locally.  Each model
    # has its own repository (READY below); the packs keep the vision tower unquantized, so pictures use
    # the original model's encoder.  Their experts live in the pack's experts.bin, so --low-ram does not
    # apply; a RAM budget covers the machines whose RAM does not hold them.
    "nvfp4": {
        "encoder": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
        "tag": "",
        "name": "qwen3.8-flash-next-nvfp4",
        "profile": "expert-profile.bin",
        "ready": True,
    },
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
    # 5-bit (Unsloth Dynamic): 158.3 GB in six shards, 91.6 GiB of experts (Q5_K gate/up, Q6_K in
    # layer 2, Q8_0 downs), close to Q8_0 in quality (94.55% of its top tokens, docs/QUANTS.md).  Its
    # shards 2 (the output head) and 3 (the PLE table) are byte-identical to UD-Q6_K_XL's and Q8_0's
    "UD-Q5_K_XL": {"arena_gb": 98.4, "ram_gb": 108, "families": ("unsloth",), "budget": True, "shards": 6,
                   "name": "qwen3.8-flash-next-ud-q5_k_xl"},
    # 6-bit (Unsloth Dynamic): 169.2 GB in six shards, 101.75 GiB of experts (47 layers Q6_K gate/up +
    # Q8_0 down, layer 2 Q8_0 gate/up) and a 54.4 GB Q8_0 PLE table in its own shard 3.  Measured 78.3
    # tokens/s with every expert in RAM (docs/UNSLOTH_Q6.md); on a smaller machine the Q4 numbers are
    # the starting point, and the experts are Q6_K: 45% more bytes per element read than Q4_K, and
    # 170-byte PLE rows against IQ4_NL's 90, so the SSD reads are slower
    "UD-Q6_K_XL": {"arena_gb": 109.2, "ram_gb": 118, "families": ("unsloth",), "budget": True, "shards": 6,
                   "name": "qwen3.8-flash-next-ud-q6_k_xl"},
    # 8-bit, the reference quality: 188.2 GB in six shards and 119.5 GiB of experts, every one of them
    # Q8_0.  For a 192 GB PC with every expert in RAM (53-57 tokens/s on the R9700, docs/QUANTS.md);
    # below that the budget keeps the hottest ones and reads the rest from the SSD
    "Q8_0": {"arena_gb": 128.3, "ram_gb": 136, "families": ("unsloth",), "budget": True, "shards": 6,
             "name": "qwen3.8-flash-next-q8_0"},
    # NVFP4's ready-made packs (docs/NVFP4.md): 128 GB downloaded, 68 GB of 4-bit NVIDIA experts and ~80 GB
    # of RAM for all of them.  RTX 50 cards multiply them on their FP4 tensor cores; other NVIDIA cards run
    # them at 8-bit precision.  NVFP4 is the GPTQ pack (best quality of the family); NVFP4-NVIDIA is NVIDIA's
    # own checkpoint, converted with plain rounding (same size and speed)
    "NVFP4": {"arena_gb": 68.0, "ram_gb": 80, "families": ("nvfp4",), "budget": True, "ready": True,
              "name": "qwen3.8-flash-next-nvfp4"},
    "NVFP4-NVIDIA": {"arena_gb": 68.0, "ram_gb": 80, "families": ("nvfp4",), "budget": True, "ready": True,
                     "name": "qwen3.8-flash-next-nvfp4-nvidia"},
}

# The ready-made NVFP4 packs (upstream setup.py's NVFP4_REPOS, checked against the Hub's LFS pointers).
# Files are downloaded into <models>/<owner>--<repo>/<their path>, so the pack folder arrives whole; the
# PLE table and the token embedding are the same files in both repositories and are hard-linked, not
# downloaded twice.  role: pack = the experts pack folder, dense = the dense GGUF (--native and
# --native-dense-gguf), ple = the FP8 n-gram table (--ple-gguf), embd = the BF16 token embedding.
_NVFP4_TOKENIZER = {
    "pack/tokenizer/chat_template.jinja": ("pack", 8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    "pack/tokenizer/merges.txt": ("pack", 3600844, "080cf95432173729d160346b7882bd2a644b07f7a3813a3eb73f4e66787bfde9"),
    "pack/tokenizer/token_type.json": ("pack", 744960, "5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b"),
    "pack/tokenizer/tokenizer.json": ("pack", 554, "87be2ac47d8bc7393b7435df6ccd276740019a393941a2585d5edce59f450c54"),
    "pack/tokenizer/vocab.json": ("pack", 5737005, "4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4"),
}
_NVFP4_SHARED = {
    "ple-fp8.gguf": ("ple", 51200246176, "37f89b8ce5a3425932763257c40229ec02723ad2ab14b4bbfc26a486f4725c9c"),
    "token-embd-bf16.gguf": ("embd", 1271398656, "386203af9070c7b43acb24ded8922876721d97bbad1a709b25cb49d694a7f51d"),
}
READY = {
    "NVFP4": {
        "repo": "Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata",
        "revision": "640a1db97446ddd5e89d2b70ecd0297987192c93",
        "files": {
            "pack/dense.bin": ("pack", 1538035200, "974df6dcefcae85e3fd2c19fcc12eeaacaadde8261ab732d5af5af6ce2e25a7a"),
            "pack/experts.bin": ("pack", 67948118016, "325e5e0d85c262a8d8278daf73b20696ec877d9a3ff584e57391fa69ad10bc64"),
            "pack/index.txt": ("pack", 93663, "f62dc71e1c74a72794db54d3917bc71131cc6eae6f3dcdc3fdada48ff70f93ab"),
            "pack/native_experts.txt": ("pack", 1599, "615f655fb28bf60b243664860b135710bf4177b111c6be89073e780a35de1018"),
            **_NVFP4_TOKENIZER,
            "qwen-nvfp4-gptq-dense.gguf": ("dense", 5991569472, "abdf74336b53d8f7ca3d99947f1eac8f19b8a544966750127cb61e561b127e3b"),
            **_NVFP4_SHARED,
        },
    },
    "NVFP4-NVIDIA": {
        "repo": "Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata",
        "revision": "cca8f7fee5ce0af932097340a75e06df25c4e57d",
        "files": {
            "pack/dense.bin": ("pack", 1538625024, "1f1c2422069de292673ae6cee7d69f7090732dec69f3f125612a0de456279714"),
            "pack/experts.bin": ("pack", 67948118016, "bb42080bb7585505b64d091d5a734c05376bbd6171406fa9688a46fbc261b957"),
            "pack/index.txt": ("pack", 117490, "17793d1c5b187a73d3c1dc1677af4d4c9b23845394dd7e88bfa53396b41b77e3"),
            "pack/native_experts.txt": ("pack", 1590, "d7f8b4372bb122f3d711030e5d522bff1c8746ac7d133a59a2cc7277283e59b6"),
            **_NVFP4_TOKENIZER,
            "qwen-nvidia-nvfp4-dense.gguf": ("dense", 5992176896, "0f795ca0cbb908e0cf2b894b30103e21455b5a05346a7ef79da208540cfd2be3"),
            **_NVFP4_SHARED,
        },
    },
}

# The Unsloth shards' published sizes and SHA-256 at the pinned revision (the Hub's LFS pointers, as
# upstream setup.py's UNSLOTH_SHARDS, UNSLOTH_IQ4_XS_SHARDS and UNSLOTH_BIG_SHARDS, and docs/QUANTS.md).
# Shard 1 is metadata only: 10,946,624 bytes in every file but UD-Q5_K_XL (10,946,618), and shards 2
# (the head) and 3 (the PLE table) are byte-identical across UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 - so a
# size check alone cannot tell the quantizations apart.
SHA256 = {
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf": (10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf": (49859583136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf": (49376141504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf": (12087983520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf": (10946624, "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf": (49835229856, "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf": (43836407744, "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf": (10946618, "610922cdb1afe1094e95d7a3e86b12ee33317ebbf7320ac9344e21b6ae4bb69f"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00006.gguf": (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00003-of-00006.gguf": (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00004-of-00006.gguf": (49990245824, "9ee6bbe462e6830864382d629101490625b1c95afa15492b7ad550c7bed2dd13"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00005-of-00006.gguf": (49882416192, "0b2437f661f584030cb5aeb8fb1bb70f6422518a6746bfcc0304167c7b6b8beb"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00006-of-00006.gguf": (3320101792, "55dd96c5ad0863b06b6393151b9ad0ac727b8d58a62e02a296ca5d5ddbfee4a3"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf": (10946624, "8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf": (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf": (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf": (49814108512, "0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf": (49419317344, "9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf": (14838313984, "e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f"),
    "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf": (10946624, "2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2"),
    "Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf": (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf": (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf": (49446841216, "bfa634025fabbd2658bf7694bc80b90e571699c768723f844c934c7ef06c691a"),
    "Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf": (49668930400, "232a8f14cc0fa4262e7efe8593774b136fe40909e39c7a020342ddaa27259a97"),
    "Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf": (34015618784, "538a93bca918064983409a41187ad4c68640f9aced6f29564da8f551bf86d7a5"),
    # The image encoders (setup.py's MMPROJ / FAMILIES[...]["mmproj"]) at the pinned revisions of the
    # repositories that publish them: qwen, coder and unsloth share one file (byte for byte, as the Hub
    # hashes show) and Swift has its own.  The coder's and unsloth's rows live in the same table under
    # this name on purpose: the same file from any of the three repositories must hash the same.
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf": (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
    "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf": (907543520, "cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60"),
}

UNSLOTH_RAM_LEFT_GB = 24   # RAM beside the budget: the OS, the engine, and the file cache the rest reads through
RESIDENT_BESIDE_GB = 16    # experts and this much more fit the RAM: every expert stays in RAM, no budget flag


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
    publishes it - for the Unsloth files, the Coder and the NVFP4 packs, the original model's repository."""
    repo, name = fam["encoder"]
    return f"https://huggingface.co/{repo}/resolve/{HF_REVISIONS[repo]}/{name}"


def ready_files(model: str) -> list:
    """A ready-made pack's files as (role, repo, revision, path, bytes, sha256), as setup.py's ready_files."""
    r = READY[model]
    return [(role, r["repo"], r["revision"], path, size, sha)
            for path, (role, size, sha) in r["files"].items()]


def ready_url(f: tuple) -> str:
    """One ready-made file at its pinned revision."""
    _, repo, revision, path, _, _ = f
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def ready_where(models: Path, f: tuple) -> Path:
    """Where a ready-made file goes: <models>/<owner>--<repo>/<its path>, so the pack folder is whole."""
    return Path(models) / f[1].replace("/", "--") / f[3]


def ready_paths(model: str, models: Path) -> dict:
    """What the engine is given: the pack folder and the dense, PLE and embedding GGUFs."""
    out = {}
    for f in ready_files(model):
        p = ready_where(models, f)
        if f[0] == "pack" and p.name == "index.txt":
            out["pack"] = p.parent
        elif f[0] in ("dense", "ple", "embd"):
            out[f[0]] = p
    return out


def ready_model_args(paths: dict) -> list:
    """The engine's model files for a ready-made pack (setup.py's ready branch): the dense GGUF is both
    --native and --native-dense-gguf, so --native does not look for a second shard."""
    return ["--pack", str(paths["pack"]),
            "--native", str(paths["dense"]), "--native-dense-gguf", str(paths["dense"]),
            "--ple-gguf", str(paths["ple"]),
            "--embd-gguf", str(paths["embd"])]


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
            "gpu": True, "max_tokens": int(max_tokens)}


def resident_budget_gib(quant: dict, ram: float) -> int:
    """The GiB of experts the engine keeps in RAM (--resident-budget-gib): the RAM less 24 GB for the OS,
    the engine and the file cache the other experts are read through, at most all of them, at least 8.
    64 GB: 40, the measured setting for UD-Q4_K_XL (docs/UNSLOTH_Q4.md)."""
    all_experts = int(quant["arena_gb"] / 1.073741824)
    return max(8, min(round(ram) - UNSLOTH_RAM_LEFT_GB, all_experts))


def budget_needed(quant: dict, ram: float) -> bool:
    """Upstream's rule (setup.py's unsloth_budget): every expert stays in RAM when they and 16 GB more fit
    this PC (no --resident-budget-gib, as ISTA's files); only below that does the RAM budget apply.  An
    explicitly asked-for budget bypasses this."""
    return ram < quant["arena_gb"] / 1.073741824 + RESIDENT_BESIDE_GB


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


def verify_sha256(shard: Path, want: tuple | None = None) -> None:
    """The published hash of a shard the Hub gives one for, once per file: the finish mark keeps it, so
    the hash is not recomputed (all of the Unsloth files together are 90-170 GB, ~15 minutes).  A file
    with the wrong hash is deleted, so the next run downloads it again.  `want` is an explicit
    (bytes, sha256) for the ready-made packs, whose table is per model and not keyed by file name."""
    if want is None:
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


def ready_verified(p: Path, sha: str) -> bool:
    """Downloaded whole and checked: its finish mark holds the pinned SHA-256."""
    m = p.with_name(p.name + ".done")
    try:
        return p.is_file() and f"sha256 {sha}" in m.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def ready_fetch(model: str, models: Path, token=None) -> None:
    """Every file of a ready-made pack: already here and checked; the same file (by SHA-256) checked for
    the other pack, linked instead of downloaded (the PLE table, the token embedding and the tokenizer);
    or downloaded at its pinned revision and checked against its size and SHA-256."""
    others = [(ready_where(models, g), g[5]) for m in READY for g in ready_files(m)]
    for f in ready_files(model):
        dst, size, sha = ready_where(models, f), f[4], f[5]
        if ready_verified(dst, sha):
            ok(f"{f[3]} already downloaded")
            continue
        src = next((p for p, s in others if s == sha and p != dst and ready_verified(p, s)), None)
        if src is not None and not dst.exists():
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(src, dst)
                mark(dst, f"sha256 {sha}")
                ok(f"{f[3]}: the same file as {src} (linked, not downloaded again)")
                continue
            except OSError:
                pass                                    # another drive: downloaded
        download(ready_url(f), dst, f"{f[1].split('/')[-1]}: {f[3]}", token=token)
        verify_sha256(dst, (size, sha))


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
    ap.add_argument("--gguf-dir", help="a directory holding the family's shard files already downloaded (not NVFP4)")
    ap.add_argument("--low-ram", choices=["on", "off"], help="default: on when the experts would not fit the RAM")
    ap.add_argument("--resident-budget-gib", type=int,
                    help="GiB of experts the engine keeps in RAM (the Unsloth and NVFP4 files; default: the RAM less 24 GB)")
    ap.add_argument("--vision", action="store_true",
                    help="images too: fetch the family's mmproj and give serve the encoder")
    ap.add_argument("--vision-exe", help="the strata-vision binary to encode images with (with --vision)")
    ap.add_argument("--vision-tokens", type=int,
                    help="the most image tokens one picture becomes (with --vision; default "
                         f"{VISION_IMAGE_TOKENS}); 1 token is 32x32 px, so it is the picture's pixel budget")
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
    ready = bool(fam.get("ready"))
    allowed = quant.get("families") or tuple(FAMILIES)
    if a.family not in allowed:
        fail(f"{a.model} is only published for: {', '.join(allowed)} (not {a.family})")
    if ready and a.gguf_dir:
        fail(f"{a.family} is a ready-made pack, not GGUF shards: --gguf-dir does not apply",
             "its files go in <data-dir>/models; remove --gguf-dir")
    if a.vision and not a.vision_exe:
        fail("--vision needs --vision-exe: the strata-vision binary reads the pictures",
             "the NixOS module passes the bundle's own (package.passthru.vision)")
    if a.vision_tokens is not None and not a.vision:
        fail("--vision-tokens is one of the image settings: --vision takes pictures in the first place")
    if a.vision_tokens is not None and a.vision_tokens <= 0:
        fail("--vision-tokens needs N > 0 (1 token is 32x32 px of picture)")

    data = Path(a.data_dir)
    models = data / "models"
    pack = None if ready else data / "packs" / (fam["tag"].lower() + a.model.lower())
    mtp = data / "mtp"
    rt = mtp / "rt"

    if ready:
        names, shards = [], []
    else:
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
        # the experts stay in the model files and the engine reads them in place: no low-RAM mode, and the
        # RAM that cannot hold all of them is handled by the budget instead
        if a.low_ram == "on":
            fail(f"{a.model} has no low-RAM mode: its experts are read from the model files in place",
                 "--resident-budget-gib says how much of them the RAM holds")
        if a.resident_budget_gib is not None and a.resident_budget_gib <= 0:
            fail("--resident-budget-gib needs N > 0")
        low_ram = "off"
        all_gib = int(quant["arena_gb"] / 1.073741824)
        if a.resident_budget_gib is not None or budget_needed(quant, ram):
            budget = a.resident_budget_gib if a.resident_budget_gib is not None else resident_budget_gib(quant, ram)
            ok(f"RAM budget: {budget} of the {all_gib} GiB of experts in RAM"
               + (", the rest read from the SSD" if budget < all_gib else ", every one of them"))
        else:
            ok(f"all the {all_gib} GiB of experts in RAM (they and {RESIDENT_BESIDE_GB} GB more fit this "
               f"machine's {ram:.0f} GB)")
    else:
        if a.resident_budget_gib is not None:
            fail(f"--resident-budget-gib is one of the budget files' settings (the Unsloth and NVFP4 files); "
                 f"{a.model} keeps all of its experts in RAM (extra engine flags: --extra-arg)")
        low_ram = (a.low_ram or "auto")
        if low_ram == "auto":
            low_ram = "on" if ram < quant["ram_gb"] else "off"

    token = Path(a.hf_token_file).read_text(encoding="utf-8").strip() if a.hf_token_file else None
    if ready:
        ready_fetch(a.model, models, token=token)
        paths = ready_paths(a.model, models)
        pack, ple, vision_model = paths["pack"], paths["ple"], paths["dense"]
        ok(f"ready-made pack: {pack}")
    else:
        if not a.gguf_dir:
            for i, (n, s) in enumerate(zip(names, shards), 1):
                download(shard_url(fam, a.model, n), s, f"{a.family}/{a.model} shard {i}/{len(names)}", token=token)

        shard1 = shards[0]
        vision_model = shard1
        for s in shards:
            check_shard(s)      # whole and readable: cheap, so it comes before the hash
            verify_sha256(s)    # the Hub's published hash, once per file

    # the image encoder: 0.9 GB, hashed like a shard, and only wanted when the service takes pictures
    vision = None
    if a.vision:
        enc = fam["encoder"][1]
        gdir = Path(a.gguf_dir) if a.gguf_dir else None
        mmproj = (gdir if gdir is not None and (gdir / enc).is_file() else models) / enc
        if mmproj.parent == models:
            download(encoder_url(fam), mmproj, "vision encoder", token=token)
        verify_sha256(mmproj)
        tokens = VISION_IMAGE_TOKENS if a.vision_tokens is None else a.vision_tokens
        vision = vision_section(a.vision_exe, mmproj, vision_model, tokens)
        ok(f"images: {enc} encoded by {a.vision_exe}")
        # the side of the largest square picture that reaches the encoder unshrunk: the number to look
        # at when a line of small text came back unreadable (1 image token = 32x32 px)
        ok(f"images: up to {tokens} image tokens per picture, the cap the encoder warms up at "
           f"(a {int((tokens * 1024) ** 0.5)} px square fits at native size)")

    if ready:
        # a ready-made pack has no GGUF shards to scan: it ships the FP8 n-gram table on its own
        pass
    else:
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

    if ready:
        model_args = ready_model_args(paths)
    else:
        model_args = ["--pack", str(pack), "--native", str(shard1), "--ple-gguf", str(ple)]
    args = [
        *model_args,
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
