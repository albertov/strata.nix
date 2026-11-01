{ strataPkgs }:
{ pkgs, config, lib, ... }:
let
  cfg = config.services.strata;
  unslothModels = [ "UD-IQ4_XS" "UD-Q4_K_XL" "UD-Q5_K_XL" "UD-Q6_K_XL" "Q8_0" ];
  nvfp4Models = [ "NVFP4" "NVFP4-NVIDIA" ];
in
{
  options.services.strata = {
    enable = lib.mkEnableOption "Strata, a local OpenAI/Anthropic-compatible API for Qwen3.8-Flash-Next (125B MoE)";

    package = lib.mkPackageOption strataPkgs "strata" { };

    family = lib.mkOption {
      type = lib.types.enum [ "qwen" "swift" "coder" "unsloth" "nvfp4" ];
      default = "swift";
      description = ''
        Which model: `swift` (Swift 1.5, answers ~1.8x sooner for the same accuracy),
        `qwen` (the original Qwen3.8-Flash-Next), `coder` (half the experts kept, code-focused),
        `unsloth` (Unsloth's 4- to 8-bit files: the closest to the full weights, the biggest
        download) or `nvfp4` (NVIDIA's 4-bit FP4 experts, as ready-made packs: NVIDIA cards only).
      '';
    };

    model = lib.mkOption {
      type = lib.types.enum [
        "Q2_0"
        "IQ2_XS"
        "IQ3_XXS"
        "IQ3_S"
        "IQ1_M"
        "UD-IQ4_XS"
        "UD-Q4_K_XL"
        "UD-Q5_K_XL"
        "UD-Q6_K_XL"
        "Q8_0"
        "NVFP4"
        "NVFP4-NVIDIA"
      ];
      default = "IQ2_XS";
      description = ''
        Quantization size: Q2_0 (fastest), IQ2_XS (recommended), IQ3_XXS (better, slower),
        IQ3_S (best, qwen only, ~62 GB RAM), IQ1_M (coder only, ~32 GB RAM).  With
        family = unsloth: UD-IQ4_XS (93.7 GB, their regular 4-bit), UD-Q4_K_XL (111.3 GB),
        UD-Q5_K_XL (158.3 GB), UD-Q6_K_XL (169.2 GB) or Q8_0 (188.2 GB, every expert at 8
        bits: the reference quality; its 119 GiB of experts all fit in RAM from ~136 GB,
        else part of them are read from the SSD).  UD-Q5_K_XL and UD-Q6_K_XL come closest to
        the full model (78 tokens/s measured for UD-Q6_K_XL with its 101 GiB of experts in
        RAM).  All five keep their experts in the GGUF files - no experts.bin, so lowRam
        does not apply - and want an NVMe SSD.  With family = nvfp4: NVFP4 (the GPTQ pack,
        the best quality of the family) or NVFP4-NVIDIA (NVIDIA's own checkpoint, the same
        size and speed); 128 GB downloaded, 68 GB of experts, all of them in RAM from ~80 GB.
        NVIDIA only: RTX 50 cards use their FP4 tensor cores, older cards run at 8-bit.
      '';
    };

    context = lib.mkOption {
      type = lib.types.enum [ 8192 32768 65536 131072 262144 ];
      default = 32768;
      description = "Context window in tokens. From 64K on, the KV cache streams from RAM.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Listen address. Use 0.0.0.0 to serve the LAN (requires apiKeyFile).";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8080;
    };

    apiKeyFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "File holding the API key (read through a systemd credential, never in the unit).";
    };

    hfTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "File holding a Hugging Face token (only for gated model repos).";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
    };

    gpus = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = ''
        CUDA device numbers (as nvidia-smi numbers them, PCI order).
        Empty: the engine picks the card with the most VRAM.
        Several: layer split across the cards.
      '';
    };

    lowRam = lib.mkOption {
      type = lib.types.nullOr lib.types.bool;
      default = null;
      description = ''
        Map the experts from the pack's experts.bin instead of copying them into RAM
        (needs a big GPU to stay fast). Null: automatic from the model's RAM needs.
        Not available for the Unsloth and NVFP4 files - see residentBudgetGiB.
      '';
    };

    residentBudgetGiB = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      description = ''
        The Unsloth and NVFP4 files only: how many GiB of their experts the engine keeps in RAM
        (--resident-budget-gib); the rest are read from the SSD for every token. Null: every expert
        stays in RAM when they and 16 GB more fit this machine; otherwise the RAM less 24 GB, capped
        at all of them (64 GB: 40, the measured setting for UD-Q4_K_XL in docs/UNSLOTH_Q4.md;
        ~80 GB: all 63 GiB of NVFP4's experts; ~136 GB: all 119 GiB of Q8_0's).
      '';
    };

    sampling = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      example = { temperature = 1.0; top_p = 0.95; top_k = 20; };
      description = "Sampling defaults for requests that leave the fields out.";
    };

    images = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Read images too: OpenAI `image_url` parts and Anthropic image blocks.  The service downloads
        the family's mmproj file (~0.9 GB) into its state directory and keeps one `strata-vision`
        process alive beside the engine (from the package's passthru.vision); at the default 1024
        image tokens that encoder takes about 1.2 GB of VRAM, for which the engine keeps 700 MiB
        free.  See `imageTokens` for how big a picture it is allowed to see.
        The Unsloth files, the Coder and the NVFP4 packs are requantizations of the original model,
        so they read images with the original's encoder.  Encoded pictures are cached by their hash,
        so a client that resends an image every turn pays for it once.
      '';
    };

    imageTokens = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      description = ''
        The most image tokens one picture becomes (`strata-vision --max-tokens`).  Null: 1024, what
        upstream's setup uses.  One image token is 32x32 px of picture in this encoder (patch 16,
        spatial merge 2x2), so the cap is a pixel budget: 1024 tokens is 1.05 Mpx - a 2032x824
        screenshot arrives shrunk 1.26x, which is what makes a line of small text unreadable - and
        2048 is 2.10 Mpx, native for that screenshot and for a 1440p one.  The encoder warms up AT the
        cap, so its VRAM is spent whether or not a client ever sends a picture, and the engine, which
        starts after it, sizes its expert cache from what is left: on a card the engine already fills,
        a bigger cap buys picture detail with decode speed.  Ignored with images = false.
      '';
    };

    mcpServers = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      description = ''
        MCP servers for the web app's chat, in Claude Desktop's format.
        They run with this service's rights; only add ones you trust.
      '';
    };

    extraEngineArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Extra arguments appended to the engine command line (e.g. --kv q4_0).";
    };

    mockEngine = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Serve a scripted mock engine instead of the model (tests; no GPU or download).";
    };

    prefetchOnStart = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Download/prepare missing model files before starting (resumable, skipped when done).";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.model != "IQ3_S" || cfg.family == "qwen";
        message = "services.strata: IQ3_S is only published for family = qwen.";
      }
      {
        assertion = cfg.model != "IQ1_M" || cfg.family == "coder";
        message = "services.strata: IQ1_M is only published for family = coder.";
      }
      {
        assertion = cfg.model != "IQ2_XS" || cfg.family != "coder";
        message = "services.strata: the coder family ships IQ1_M only; pick model = IQ1_M.";
      }
      {
        assertion = !(lib.elem cfg.model unslothModels) || cfg.family == "unsloth";
        message = "services.strata: ${cfg.model} is one of Unsloth's files: family = unsloth.";
      }
      {
        assertion = cfg.family != "unsloth" || lib.elem cfg.model unslothModels;
        message = "services.strata: family = unsloth takes model = UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL or Q8_0.";
      }
      {
        assertion = cfg.family != "unsloth" || cfg.lowRam != true;
        message = ''
          services.strata: the Unsloth files have no low-RAM mode - their experts stay in the GGUF
          files and the engine reads them in place. Set residentBudgetGiB instead.'';
      }
      {
        assertion = !(lib.elem cfg.model nvfp4Models) || cfg.family == "nvfp4";
        message = "services.strata: ${cfg.model} is an NVFP4 pack: family = nvfp4.";
      }
      {
        assertion = cfg.family != "nvfp4" || lib.elem cfg.model nvfp4Models;
        message = "services.strata: family = nvfp4 takes model = NVFP4 or NVFP4-NVIDIA.";
      }
      {
        assertion = cfg.family != "nvfp4" || cfg.lowRam != true;
        message = ''
          services.strata: the NVFP4 packs have no low-RAM mode - their experts live in the pack's
          experts.bin and the engine reads them in place. Set residentBudgetGiB instead.'';
      }
      {
        assertion = cfg.host == "127.0.0.1" || cfg.host == "localhost" || cfg.apiKeyFile != null;
        message = "services.strata: exposing the API beyond loopback requires apiKeyFile.";
      }
      {
        assertion = !cfg.mockEngine || !cfg.prefetchOnStart;
        message = "services.strata: mockEngine requires prefetchOnStart = false.";
      }
      {
        assertion = cfg.mockEngine || (cfg.package ? passthru.engine);
        message = "services.strata: package must be the strata-serve bundle (it carries the engine under passthru.engine).";
      }
      {
        assertion = !cfg.images || cfg.mockEngine || (cfg.package ? passthru.vision);
        message = ''
          services.strata: images need a bundle that carries the image encoder (passthru.vision);
          the flake's `strata` package does, one built without it does not.'';
      }
    ];

    systemd.services.strata = {
      description = "Strata: Qwen3.8-Flash-Next (125B MoE) local inference API";
      documentation = [ "https://github.com/Niko1221/Strata" ];
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      path = [ cfg.package ];

      environment = lib.optionalAttrs (cfg.gpus != [ ]) {
        CUDA_DEVICE_ORDER = "PCI_BUS_ID";
        CUDA_VISIBLE_DEVICES = lib.concatStringsSep "," cfg.gpus;
      };

      # Each extra engine argument goes over as --extra-arg=VALUE, with the equals sign: a bare
      # `--extra-arg --flag` makes argparse stop with "expected one argument", because the value looks
      # like another flag - which is why every argument starting with -- used to fail here.  The comment
      # cannot live inside the command string below: it would land in the generated script.
      preStart = lib.mkIf (cfg.prefetchOnStart && !cfg.mockEngine) (toString (
        pkgs.writeShellScript "strata-prestart" (''
          set -euo pipefail
          state="$STATE_DIRECTORY"
          exec ${cfg.package}/bin/strata-prefetch \
            --family ${lib.escapeShellArg cfg.family} \
            --model ${lib.escapeShellArg cfg.model} \
            --context ${toString cfg.context} \
            --data-dir "$state" \
        '' + lib.optionalString (cfg.lowRam != null) ''
            --low-ram ${if cfg.lowRam then "on" else "off"} \
        '' + lib.optionalString (cfg.residentBudgetGiB != null) ''
            --resident-budget-gib ${toString cfg.residentBudgetGiB} \
        '' + lib.optionalString cfg.images ''
            --vision --vision-exe ${lib.escapeShellArg (toString cfg.package.passthru.vision)}/bin/strata-vision \
        '' + lib.optionalString (cfg.images && cfg.imageTokens != null) ''
            --vision-tokens ${toString cfg.imageTokens} \
        '' + lib.optionalString (cfg.hfTokenFile != null) ''
            --hf-token-file "/run/credentials/strata.service/hf-token" \
        '' + ''
            --emit-config "$state/config.json" \
            --engine ${lib.escapeShellArg (toString cfg.package.passthru.engine)}/bin/strata \
            --host ${lib.escapeShellArg cfg.host} \
        '' + lib.optionalString (cfg.apiKeyFile != null) ''
            --api-key-file "/run/credentials/strata.service/api-key" \
        '' + lib.optionalString (cfg.gpus != [ ]) ''
            --gpu ${lib.escapeShellArg (lib.concatStringsSep "," cfg.gpus)} \
        '' + lib.optionalString (cfg.sampling != { }) ''
            --sampling-json ${pkgs.writeText "strata-sampling.json" (builtins.toJSON cfg.sampling)} \
        '' + lib.optionalString (cfg.mcpServers != { }) ''
            --mcp-json ${pkgs.writeText "strata-mcp.json" (builtins.toJSON cfg.mcpServers)} \
        '' + lib.concatMapStrings (x: ''--extra-arg=${lib.escapeShellArg x} '') cfg.extraEngineArgs
        )
      ));

      serviceConfig = {
        Type = "simple";
        DynamicUser = true;
        StateDirectory = "strata";
        StateDirectoryMode = "0750";
        # First start: preStart downloads the model files - ~70 GB for the ordinary sizes, up to
        # 170 GB for the Unsloth ones - and builds the pack; the default TimeoutStartSec kills it
        # mid-download (the service is resumable, but converge it in one pass). Applies to the
        # start, incl. ExecStartPre.
        TimeoutStartSec = "0";
        WorkingDirectory = "%S/strata";
        UMask = "0077";
        Restart = "on-failure";
        RestartSec = 10;
        LoadCredential =
          (lib.optional (cfg.apiKeyFile != null) "api-key:${cfg.apiKeyFile}")
          ++ (lib.optional (cfg.hfTokenFile != null) "hf-token:${cfg.hfTokenFile}");
        ExecStart =
          if cfg.mockEngine
          then
            "${cfg.package}/bin/strata-serve --engine mock --port ${toString cfg.port} " +
            "--host ${lib.escapeShellArg cfg.host}"
          else
            "${cfg.package}/bin/strata-serve --engine strata --config %S/strata/config.json " +
            "--port ${toString cfg.port}";
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [ "AF_UNIX" "AF_INET" "AF_INET6" ];
        LockPersonality = true;
        MemoryDenyWriteExecute = false;
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
  };
}
