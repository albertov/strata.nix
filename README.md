# strata.nix

Run [Strata](https://github.com/Niko1221/Strata) — the engine that serves
**Qwen3.8-Flash-Next** (a 125B-parameter MoE, ~6B active) on a single NVIDIA GPU —
as a native NixOS service, with an OpenAI- and Anthropic-compatible API on localhost.

```
┌─────────────┐   OpenAI/Anthropic    ┌───────────────┐   stdin/stdout   ┌────────────────┐
│ your apps   │ ────  HTTP  API  ───▶ │  strata-serve │ ───────────────▶ │ strata engine  │
│ / web chat  │   http://127.0.0.1    │  (python)     │                  │ (CUDA, built   │
└─────────────┘                       └───────────────┘                  │  from source)  │
                                                                          └────────────────┘
        weights: downloaded once (resumable) into /var/lib/strata, not in the nix store
```

## Requirements

- NixOS on **x86_64-linux** with an NVIDIA RTX 20/30/40/50 GPU (12 GB VRAM or more)
  and the official driver (580+). About 64 GB of RAM for IQ2_XS and below; the Unsloth
  and NVFP4 sizes need more (up to ~136 GB and ~80 GB).
- ~70 GB of free disk space for the model files (downloaded at first service start);
  94-188 GB for the [Unsloth sizes](#the-unsloth-files) and 128 GB for the
  [NVFP4 packs](#nvfp4-packs) — an NVMe SSD there matters.

The engine is compiled for `sm_120` (RTX 50) by default, so `nix build .#strata-engine` works on an
RTX 5070/5090 out of the box; for another card override the architecture
(`strata-engine.override { cudaArch = "89"; }` for RTX 40).

## Quick start

```nix
# flake.nix of your system
{
  inputs.strata-nix.url = "github:Yunor743/strata.nix";
  # ...
  outputs = { self, nixpkgs, strata-nix, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        strata-nix.nixosModules.default
        {
          services.strata.enable = true;
          # defaults: Swift 1.5, IQ2_XS, 32K context, API on 127.0.0.1:8080
        }
      ];
    };
  };
}
```

The service:

1. (first start) downloads the model shards from Hugging Face, builds the pack and
   the MTP draft layer — resumable, so an interrupted start continues where it left off;
2. writes `/var/lib/strata/config.json` (engine arguments, API key if any);
3. starts `strata-serve` on `http://127.0.0.1:8080`.

While the model loads (35-55 GB into RAM) the machine can be slow for 1-3 minutes:
that is normal. API: `http://127.0.0.1:8080/v1/chat/completions` (OpenAI) and
`/v1/messages` (Anthropic). Chat in the browser: `http://127.0.0.1:8080/`.

## Options (`services.strata.*`)

| Option | Default | Meaning |
| --- | --- | --- |
| `enable` | `false` | Enable the service. |
| `package` | `strataPkgs.strata` | The strata-serve bundle (carries the engine). |
| `family` | `swift` | `swift` (Swift 1.5: same accuracy, answers ~1.8x sooner), `qwen` (original), `coder` (half the experts, code-focused, ~32 GB RAM), `unsloth` (Unsloth's 4- to 8-bit files) or `nvfp4` (NVIDIA's 4-bit FP4 experts: NVIDIA only). |
| `model` | `IQ2_XS` | `Q2_0` / `IQ2_XS` / `IQ3_XXS` / `IQ3_S` (qwen only) / `IQ1_M` (coder only) / `UD-IQ4_XS` / `UD-Q4_K_XL` / `UD-Q5_K_XL` / `UD-Q6_K_XL` / `Q8_0` (unsloth only) / `NVFP4` / `NVFP4-NVIDIA` (nvfp4 only). |
| `context` | `32768` | 8192–262144 tokens. |
| `host` / `port` | `127.0.0.1` / `8080` | Listen address. |
| `apiKeyFile` | `null` | Path to a file with the API key (systemd credential). Required when `host` is not loopback. |
| `hfTokenFile` | `null` | Hugging Face token file, only for gated repos. |
| `openFirewall` | `false` | Open the TCP port in the firewall. |
| `gpus` | `[ ]` | CUDA device indices; several = layer split. Empty: the card with the most VRAM. |
| `lowRam` | `null` | Map experts from disk instead of RAM copies (auto when RAM is short). Not for the Unsloth or NVFP4 files. |
| `residentBudgetGiB` | `null` | Unsloth/NVFP4 files: GiB of their experts the engine keeps in RAM; the rest are read from the SSD. Auto: all of them when they fit with 16 GB spare, else RAM − 24 GB. |
| `images` | `false` | Read pictures (OpenAI `image_url`, Anthropic image blocks): fetches the model's mmproj (0.9 GB) and runs `strata-vision` beside the engine. |
| `imageTokens` | `null` | Pictures: the most image tokens one picture becomes (`strata-vision --max-tokens`). Auto: 1024. One token is 32×32 px, so it is a pixel budget the encoder warms up at. |
| `sampling` | `{ }` | Sampling defaults (temperature, top_p, top_k, …). |
| `mcpServers` | `{ }` | MCP servers for the web chat (Claude Desktop format; they run with the service's rights). |
| `extraEngineArgs` | `[ ]` | Extra engine arguments, one list element each (e.g. `[ "--kv" "q4_0" ]` → `--kv q4_0`). |
| `mockEngine` | `false` | Scripted mock instead of the model (tests; no GPU, no download). |

## Images

`images = true` takes pictures: OpenAI's `image_url` parts and Anthropic's image blocks, in the web
chat and through the API. It costs three things, none of them per turn:

- **0.9 GB** more on disk: the family's `mmproj-*.gguf`, fetched at a pinned revision and checked
  against its published SHA-256 like every other file. The Coder and the Unsloth files are
  requantizations of the original model and use its encoder; only Swift 1.5 has one of its own.
- **~1.2 GB of VRAM** at 1024 image tokens: `strata-serve` keeps one `strata-vision` process alive
  beside the engine, and the engine is told to keep 700 MiB free for it (`--vram-reserve-mib`).
  A GPU encoder is what this build makes; on a card near 12 GB it is worth reading prompts without
  it (`--vision cpu` upstream, ~2-3 s per picture there).
- nothing per request for a picture a client resends every turn: the encoder caches by the image's
  hash.

Upstream publishes image runs for the original model, Swift and the Coder, for `UD-IQ4_XS` (a regular
choice), and end-to-end tests for `UD-Q4_K_XL` and `UD-Q5_K_XL`; for `UD-Q6_K_XL` and `Q8_0` nothing is
published either way, so treat those combinations as untested.  The NVFP4 packs keep the vision tower
unquantized and read pictures with the original model's encoder.

**How big a picture.** One image token is 32×32 px in this encoder (`clip.vision.patch_size` 16,
`spatial_merge_size` 2×2), so `imageTokens` is a picture's pixel budget: the default 1024 tokens is
1.05 Mpx, which shrinks a 2032×824 screenshot 1.26× — around where a 20 px line of text stops being
readable — where 2048 is 2.10 Mpx (native for that screenshot, and for 1440p) and 4096 is 4.19 Mpx.
The encoder warms up **at the cap**, so its VRAM is spent whether or not a client ever sends a
picture, and the engine, which starts after it, sizes its expert cache from what is left: on a card
the engine already fills, a bigger cap is paid for in decode speed. Upstream recommends 1024 and caps
nothing — "more than the encoder's default needs more VRAM and context per picture".

## The Unsloth files

`family = "unsloth"` takes Unsloth's own quantizations of the original model, the closest of them to
the full weights. They are big, they keep their experts in the GGUF files (no `experts.bin`, so
`lowRam` does not apply), and they need the engine's opt-in k-quant expert kernels — which
`strata-engine` always builds (`STRATA_MMQ_KQUANTS`, `STRATA_Q6K_EXPERTS`). Their cost is ~1-3
expert slots of VRAM for every other model on the same build, which is why upstream leaves them off.

| Model | Download | Experts | Notes |
| --- | ---: | ---: | --- |
| `UD-IQ4_XS` | 93.7 GB (3 shards) | 59.5 GB | the regular ~4-bit choice: all of them in RAM from ~72 GB of RAM |
| `UD-Q4_K_XL` | 111.3 GB (4 shards) | 77.0 GB | 4-bit; 7-8.5 tokens/s on a 64 GB PC with a 12 GB card |
| `UD-Q5_K_XL` | 158.3 GB (6 shards) | 98.4 GB | 5-bit: Q5_K experts with Q8_0 downs, close to 8-bit (94.6% of Q8_0's top tokens); 57-60 tokens/s with every expert in RAM |
| `UD-Q6_K_XL` | 169.2 GB (6 shards) | 109.2 GB | 6-bit: Q6_K experts with Q8_0 downs; 78 tokens/s measured with every expert in RAM; its PLE table is a Q8_0 54.4 GB shard the engine mmaps |
| `Q8_0` | 188.2 GB (6 shards) | 128.3 GB | 8-bit, the reference quality (every expert Q8_0); all 119 GiB of experts in RAM from ~136 GB of RAM here; 53-57 tokens/s measured with every expert in RAM |

```nix
services.strata = {
  enable = true;
  family = "unsloth";
  model = "UD-Q6_K_XL";
  context = 8192;              # every GB of KV cache is a GB less of cached experts
  gpus = [ 0 1 ];
};
```

What the prefetch does for them: it fetches every shard at the pinned Hugging Face revision
(`resolve/38bb39ee…/<quant>/`), checks each file's size **and published SHA-256** once (shard 1 is
metadata-only and the same size in every file but `UD-Q5_K_XL`, and shards 2-3 are byte-identical
across `UD-Q5_K_XL`, `UD-Q6_K_XL` and `Q8_0`, so a size alone cannot tell them apart), and packs with
`--compat-bf16`.  When the experts do not all fit with 16 GB to spare, it gives the engine
`--resident-budget-gib` — the RAM less 24 GB (64 GB → 40 GiB, the measured setting in
`docs/UNSLOTH_Q4.md`); from ~136 GB every one of `Q8_0`'s 119 GiB fits, so no flag is written and the
engine keeps them all.  Under the cap, the experts that do not fit are read from the SSD for every
token: use an NVMe.

## NVFP4 packs

`family = "nvfp4"` serves the original model with NVIDIA's 4-bit FP4 routed experts, as ready-made
Strata packs: nothing is converted or packed locally.  They need an NVIDIA card (RTX 50 cards use
their FP4 tensor cores; older cards multiply at 8-bit precision), and they read pictures with the
original model's encoder.

| Model | What it is | Download | Experts |
| --- | --- | ---: | ---: |
| `NVFP4` | the GPTQ pack (Strata's own conversion of Qwen's BF16 experts): the best quality of the family | 128 GB | 68 GB |
| `NVFP4-NVIDIA` | NVIDIA's own checkpoint, converted with plain rounding; the same size and speed | 128 GB | 68 GB |

Every expert stays in RAM from ~80 GB of RAM; below that the RAM budget keeps the hottest ones and
reads the rest from the pack's `experts.bin` on the SSD.  The prefetch downloads the pack folder
whole (`index.txt`, `experts.bin`, `dense.bin`, the tokenizer), the dense GGUF, the FP8 n-gram table
and the BF16 token embedding at the repositories' pinned revisions, checks every file's size and
**published SHA-256**, and hard-links the n-gram table and token embedding when the other pack is
already present.  The MTP draft layer comes from the original Qwen checkpoint, as for every family.

```nix
services.strata = {
  enable = true;
  family = "nvfp4";
  model = "NVFP4";
  context = 32768;
};
```

Upstream also has `NVFP4-QAD` (NVIDIA's FP4-trained checkpoint, converted on the PC); it is not
offered here because that conversion needs `torch`/`transformers` and llama.cpp's converter, which
this bundle does not carry.  Licenses: the GPTQ pack is under Qwen's Community License 1.0;
`NVFP4-NVIDIA` is under the NVIDIA Open Model License plus Qwen's Community License 1.0.

Example with the original model, the best quant and a LAN-reachable API:
```nix
services.strata = {
  enable = true;
  family = "qwen";
  model = "IQ3_XXS";
  context = 131072;
  host = "0.0.0.0";
  port = 8080;
  apiKeyFile = "/run/secrets/strata-api-key";
  sampling = { temperature = 1.0; top_p = 0.95; top_k = 20; };
};
```

## Packages

- `strata` — the server bundle: `strata-serve`, `strata-chat` and `strata-prefetch`
  (the download/prepare tool, usable standalone; `strata-prefetch --help`).
- `strata-engine` — the CUDA inference binary, built from source.  Default
  `cudaArch = "120"` (RTX 50); `strata-engine.override { cudaArch = "89"; }` for RTX 40.  It always
  enables the k-quant expert kernels the Unsloth files need (`STRATA_MMQ_KQUANTS`,
  `STRATA_Q6K_EXPERTS`), which upstream keeps opt-in because the extra kernel instances load eagerly
  and cost other models a couple of expert slots of VRAM.
- `strata-vision` — the image encoder (llama.cpp's mtmd at the same pinned llama.cpp, with ggml-cuda),
  for `images = true`. The bundle carries it under `passthru.vision`, so nothing builds it until a
  service asks for images; the build is `STRATA_PORTABLE=ON` on purpose — a `-march=native` encoder
  built on one PC dies with an illegal instruction on another (#411 #412 #419).
- The model weights are **not** packaged: they are downloaded at runtime into the
  service's state directory (the nix store would not survive a 70 GB model well).

## How the engine is built

`strata-engine` compiles the pinned Strata source with nixpkgs' CUDA toolchain
(`backendStdenv`, CUDA 12.9) against the llama.cpp commit pinned in Strata's
`third_party/ggml/VERSION.txt`. If Strata bumps that pin, evaluation fails with the
new commit and an instruction to update the `llama-pinned` input.

The flake input `strata` tracks upstream `main`. The lock file decides what you
actually run — `nix flake lock --update-input strata` pulls new releases
(and new engine features); audit what lands in it before doing so.

## Testing

- `nix flake check` — a serve smoke test (mock engine: HTTP endpoints without a GPU), a NixOS VM
  test of the systemd service with the mock engine, and `prefetch-tables`, which checks the download
  tables offline against the names, sizes and SHA-256 the Hugging Face API gives at the pinned
  revisions (a wrong subfolder or shard count would otherwise cost a 100-188 GB download first). The
  same check covers the image encoders and the NVFP4 packs' file lists.
- `nix build .#strata-engine` — the real compile (20-40 minutes, once, cached after).
- `nix build .#strata-vision` — the image encoder, for `images = true` (it builds ggml-cuda again).

## Licenses

- This flake: MIT.
- Strata (engine, server, tools): MIT, upstream <https://github.com/Niko1221/Strata>.
- The model weights are not part of this flake; each model's license applies
  (Qwen Community 1.0; Swift Open License 1.0; the NVFP4 packs' licenses are listed
  above; see the models' Hugging Face pages).
