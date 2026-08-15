# Output directory layout

All generated artifacts live under `outputs/`, which is ignored by Git.

```text
outputs/
├── training/
│   └── <environment>/
│       └── <experiment>/
│           └── <run>/
│               ├── checkpoints/
│               ├── videos/
│               ├── wandb/
│               └── training_rank0.log
└── analysis/
    └── <analysis>/
        └── <run>/
```

`steps_per_epoch`, checkpoints, W&B files, logs, and final-policy videos for a
training run therefore remain together. Relative output roots are resolved
against the repository directory, not the caller's current working directory.

The default run name is:

```text
<YYYYMMDD-HHMMSS>__<policy>__<clip-mode>__seed-<seed>
```

Comparison launchers use the supplied W&B group as `<experiment>` and the
tested condition as `<run>`. Analysis entry points write to
`outputs/analysis/policy-clip/` or `outputs/analysis/value-clip/`.
