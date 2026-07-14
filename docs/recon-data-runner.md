# Recon-data self-hosted runner

The lab-internal Tier-2 and Tier-3 workflows route by capability rather than by
machine name:

```yaml
runs-on: [self-hosted, recon-data]
```

The current repository-scoped runner is `PSH1VS12J3-recon-data`. It runs as the
non-privileged `github-runner` account under systemd and accepts jobs only from
`den-sq/lftomo`.

## Installed layout

| Purpose | Path |
| --- | --- |
| GitHub runner | `/opt/actions-runner/lftomo` |
| Persistent conda environment | `/opt/actions-runner/envs/lftomo` |
| Runner worktree | `/opt/actions-runner/lftomo/_work` |
| Download and application caches | `/mnt/wsl/PHYSICALDRIVE2p1/recon-runner/cache` |
| Temporary reconstruction scratch | `/mnt/wsl/PHYSICALDRIVE2p1/recon-runner/scratch` |
| Local phase-contrast data | `/mnt/wsl/PHYSICALDRIVE2p1/B10/...` |
| Network striped/helical data | `/mnt/z/gpfs/...` |

The runner installation and checkout stay on WSL's Linux filesystem. Large,
replaceable cache and scratch files use the local physical disk. The physical
disk is attached and mounted from elevated Windows PowerShell; `/mnt/z` is a
persistent DrvFS mount.

The systemd service has pre-start checks for representative `scan.yaml` files
on both mounts. If either mount is unavailable, the listener stays offline and
retries every 30 seconds instead of accepting a job that cannot read its data.

## Routine status checks

```bash
systemctl is-system-running
systemctl status actions.runner.den-sq-lftomo.PSH1VS12J3-recon-data.service
findmnt /mnt/wsl/PHYSICALDRIVE2p1
findmnt /mnt/z
df -h /mnt/wsl/PHYSICALDRIVE2p1 /mnt/z
```

From an authenticated operator account, confirm GitHub sees the runner:

```bash
gh api repos/den-sq/lftomo/actions/runners \
  --jq '.runners[] | {name,status,busy,labels:[.labels[].name]}'
```

The expected labels are `self-hosted`, `Linux`, `X64`, and `recon-data`.

## Validation workflow

`Recon-data Tier-2 validation` is manual-only. Dispatch it from the Actions UI
or with:

```bash
gh workflow run "Recon-data Tier-2 validation" --repo den-sq/lftomo --ref <branch>
```

The workflow installs the checkout into the persistent environment without
resolving dependencies, then runs:

```bash
python -m pytest tests/test_tier2_datasets.py --run-tier2
```

All six tests must pass. The workflow token is explicitly limited to
read-only repository contents.

## Environment maintenance

The service PATH begins with `/opt/actions-runner/envs/lftomo/bin`; it does not
inherit personal or Windows executable paths. Before changing the environment,
confirm the runner is idle and stop the service:

```bash
sudo systemctl stop actions.runner.den-sq-lftomo.PSH1VS12J3-recon-data.service
sudo -u github-runner /opt/actions-runner/envs/lftomo/bin/conda env update \
  --prefix /opt/actions-runner/envs/lftomo \
  --file /path/to/lftomo/environment.yml \
  --prune
sudo systemctl start actions.runner.den-sq-lftomo.PSH1VS12J3-recon-data.service
```

Run the hermetic suite and the manual Tier-2 workflow after an environment
refresh. Package downloads persist under the physical-drive cache directory.

## Logs and recovery

```bash
journalctl -u actions.runner.den-sq-lftomo.PSH1VS12J3-recon-data.service -n 200
sudo systemctl restart actions.runner.den-sq-lftomo.PSH1VS12J3-recon-data.service
```

If the service repeatedly fails its pre-start checks, restore the missing data
mount; do not remove the checks. If GitHub reports the runner offline while the
service is active, inspect both the journal and
`/opt/actions-runner/lftomo/_diag/`.

The runner normally self-updates. For replacement or removal, first ensure it
is idle, stop and uninstall the service with `svc.sh`, and remove the runner in
repository Settings. Registration and removal tokens are short-lived and must
never be committed, logged in issues, or retained in files.

## Security boundaries

- The runner account has no sudo access.
- Registration is repository-scoped; add separate service instances for other
  repositories only when they gain recon-data workflows.
- Data workflows use `workflow_dispatch`, not automatic pull-request triggers.
- Do not run untrusted pull-request code on this persistent host.
- Keep secrets out of the runner environment and grant workflow permissions
  only when a concrete job requires them.
