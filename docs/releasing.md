# Releasing lftomo

lftomo uses tag-driven semantic versioning through
[`hatch-vcs`](https://github.com/ofek/hatch-vcs). The git tag is the single
source of truth for the package version; there is no version string to edit in
`pyproject.toml`, which prevents the metadata drift that let public-contract
additions ship under a frozen `0.1.0`.

## Choosing the bump

Decide the bump from the change surface since the previous tag:

- **MAJOR** — a breaking change to a public contract: a removed or renamed
  public symbol, a removed or retyped `ProgressEvent` / `AttemptProgress` field,
  or changed semantics a consumer relies on.
- **MINOR** — a backward-compatible addition to the public contract: a new
  public symbol, a new event/progress field, a new Stage. Consumers floor
  against this. (The `#219`/`#220` execution-span additions were a MINOR that
  never happened.)
- **PATCH** — an internal fix or refactor with no public-contract change.

Pre-1.0 we still bump **MINOR for every public-contract addition** so downstream
repositories have a concrete version to pin.

## Stage parameter migration (#332)

This change removes the public `Locality` enum and `StageContract.locality`.
Bounded execution uses `bounded_views_from` and the `run_sinogram` kernel;
dispatch remains owned by `STAGE_DISPATCH`. Apply the breaking-contract rule
above when releasing this change.

Built-in Stages now own validated, immutable `stage.params` models. Consumers
must read `stage.params.sigma` instead of `stage.sigma`, for example. Construct
Stages with keyword arguments; `LayoutConvertStage(target, reason)` retains
its existing positional input adapter. Existing parameter aliases normalize
into one model field, and unknown parameters now fail validation instead of
being silently ignored. Configuration dictionaries remain the serialized
input format and retain their supplied values for sweep and cache identity.

Stage authors declare nested `Params(StageParams)` models and class-level
contracts. Parameter-dependent contract facts use `contract_for(params)`;
the base constructor resolves the instance contract. `SinogramStage` and
`ProjectionStage` own the respective per-unit loops. See
[Adding a Stage](adding-a-stage.md) for the updated example and schema API.

Removing the contract field changes prefix-cache fingerprints and causes safe
cold misses. Existing cached results need no migration.

## Cutting a release

1. Land the change on `master` and confirm CI is green.
2. Tag and push:

   ```console
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

3. `release.yml` checks out the tag with full history, builds the sdist and
   wheel, verifies the built (hatch-vcs) version equals the tag, and publishes a
   GitHub Release with the artifacts attached and generated notes.

## How consumers pin

Downstream repositories (`hatmehyt`, `lfcapture`) pin an exact release ref:

```
lftomo @ git+ssh://git@github.com/den-sq/lftomo.git@vX.Y.Z
```

Each consumer's runtime capability check remains authoritative for editable or
source installs whose package metadata may not describe their actual API
surface.

## Moving to a package index later

The release already produces standard sdist/wheel artifacts. To enable
`pip install 'lftomo>=X.Y'` resolution, add a publish step (PyPI or a private
index) to `release.yml`; the version source does not change.
