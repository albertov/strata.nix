{
  stdenvNoCC,
  strataPkg,
}:

# The download tables in prefetch.py are worth 100-170 GB of traffic: a wrong subfolder, a missing
# slash or a wrong shard count costs a full download before anything notices.  Checked offline
# against the answers the Hugging Face API gives at the pinned revisions (docs/UNSLOTH_Q4.md,
# docs/UNSLOTH_Q6.md) - no network, no download.
stdenvNoCC.mkDerivation {
  name = "strata-prefetch-tables";

  src = ./test_prefetch_tables.py;

  buildCommand = ''
    export STRATA_SHARE=${strataPkg}/share/strata
    export STRATA_MODULE=${../nix/module.nix}
    export PYTHONPATH=$STRATA_SHARE
    cp $src test_prefetch_tables.py
    ${strataPkg.passthru.py}/bin/python test_prefetch_tables.py
    touch $out
  '';
}
