# Super-resolution external-code security

The DRHT and LDM Stages are adapters to operator-supplied repositories and
checkpoints. These paths are configuration-gated because the model sources and
weights are not distributed with lftomo.

Treat all configured assets as executable, trusted inputs:

- `DRHTStage` temporarily inserts `repo_path` into `sys.path`, imports its
  `util` module, calls `get_arch`, and loads `checkpoint_path` with
  `torch.load`.
- `LDMStage` inserts `repo_path` into `sys.path`, imports the configured module,
  resolves the configured callable, and passes it the config/checkpoint paths.
- Python imports execute module-level code. PyTorch checkpoints may use pickle
  and can execute code during deserialization. File existence is not a trust or
  integrity check.

Operational requirements:

1. Use only repositories and checkpoints from a trusted, access-controlled
   source.
2. Pin repository commits and record checkpoint hashes in run provenance.
3. Do not point these Stages at user uploads or unreviewed shared directories.
4. Run model inference with the least filesystem and network access required.
5. Use the explicit interpolation fallback when trusted model assets are not
   available; fallback mode does not import an external repository or load a
   checkpoint.

Restoring `sys.path` after inference prevents persistent import-path changes; it
does not sandbox code that was imported or deserialize checkpoints safely.
