Acknowledged — but before I open ~30–60 issues at once and pollute the project, let me verify access and propose the inventory shape for sign-off. Mass-creating issues is durable + lab-visible, so worth a "this is what I'm about to file" moment first.Two blockers to clear before I can file anything, plus the inventory question I want sign-off on. Findings:

1. My token is missing project scope. gh-tava has gist, read:org, repo, workflow but not read:project / project. Same scope-mismatch lesson I already have in memory from the JOURNAL_PAT and gh-cli-rotation notes. Without it I can't read the project, can't add issues to it, can't query its fields.

Fix (on Yarrkha): gh-tava auth refresh -h github.com -s read:project,project — additive OAuth, doesn't invalidate the existing token. Or if you'd rather, I can file the issues without project linkage and you batch-add them via the project's auto-add filters; lossier but unblocks immediately.

2. lfcapture repo doesn't exist yet (404). Fine — its creation is one of the planned issues — but it means lfcapture-targeted issues need a temporary home. Proposal: file them in den-sq/hatmehyt for now with a migrate-to-lfcapture label, and reassign once the repo exists.

3. Repo write access is good on the three existing repos (lftomo, hatmehyt, dan_scripts — all push + triage). Labels are stock GitHub only; I'd need to create phase:0..4, phase:1.5, plus topic labels (architecture, algorithm, cli, testing, migration, consent-needed).

Proposed issue inventory shape — for sign-off

Rather than 60+ tickets, I want to file at two granularities:

• Detail issues for things actionable today (Phase 0 → Phase 1 + Phase 1.5 + foundational test-data work). These are PR-sized and reference the plan section.
• Epic issues for Phase 2 / 3 / 4, each a checklist that becomes detailed issues as we approach them. Filing all of Phase 2 as detail issues now risks them being wrong before they're started — the Stage ABC shape lands in Phase 1 and informs the Stage tickets.

Draft inventory (~22 detail + 3 epics):

| #  | Title                                                                                                      | Repo                                  | Phase | Notes                                                                          |
| -- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------- | ----- | ------------------------------------------------------------------------------ |
| 1  | Repo housekeeping: pyproject + mkdocs + CI on existing tests                                               | lftomo                                | 1     | foundational                                                                   |
| 2  | Define Stage / PipelineState / Pipeline ABCs (incl. Dask-stub backend)                                     | lftomo                                | 1     | gates everything else                                                          |
| 3  | Define unified config schema (Machine/Session/Scan/Recon pydantic)                                         | lftomo                                | 1     | gates importers                                                                |
| 4  | Pipeline-spec validator: shape-compat at build time                                                        | lftomo                                | 1     | depends on #2                                                                  |
| 5  | Re-express weighted_normalize as Stage (proof of pattern)                                                  | lftomo                                | 1     | depends on #2                                                                  |
| 6  | Re-express phase_retrieval (Paganin) as Stage + regression tests                                           | lftomo                                | 1     |                                                                                |
| 7  | Re-express remove_sort_vo / remove_dead_vo as Stages + tests                                               | lftomo                                | 1     |                                                                                |
| 8  | Re-express remove_stripe_fw as Stage + tests                                                               | lftomo                                | 1     |                                                                                |
| 9  | First-class TIFF stack read Stage (benchmark byteread_helper vs distribute_read)                           | lftomo                                | 1     |                                                                                |
| 10 | First-class HDF5 dxchange read Stage                                                                       | lftomo                                | 1     |                                                                                |
| 11 | TIFF slices write Stage                                                                                    | lftomo                                | 1     |                                                                                |
| 12 | Validate scan.yaml schema against all five surveyed lineages                                               | lftomo                                | 1     | depends on #3                                                                  |
| 13 | hatmehyt recon-cli-pivot branch: strip phenotyping; set up [capture] extras stub                           | hatmehyt                              | 1     | single coherent PR                                                             |
| 14 | Top-level recon command-surface stubs (ingest / prep flats / run / convert / export)                       | hatmehyt                              | 1     | depends on #13                                                                 |
| 15 | recon ingest canonical-layout writer                                                                       | hatmehyt                              | 1     | depends on #3, #12                                                             |
| 16 | Config importers: from-recon-minimal-yaml + from-phase-contrast-ini + from-r254-hardcoded + from-b10-recon | hatmehyt                              | 1     | one issue, four sub-tasks                                                      |
| 17 | Create den-sq/lfcapture empty skeleton + own lfcapture console script                                      | hatmehyt (migrate-to-lfcapture label) | 1     | becomes lfcapture#1 when repo exists                                           |
| 18 | Migrate capture/ subtree from hatmehyt + dan_scripts/capture.py → lfcapture                                | hatmehyt + dan_scripts                | 1     | depends on #17                                                                 |
| 19 | Catalog recon_minimal.py's TomoPy-fix wrappers                                                             | lftomo                                | 1.5   | gates Phase 2 on recon/center/phase Stages                                     |
| 20 | Tier-1 synthetic fixtures (Shepp-Logan + striped + misaligned phantoms)                                    | lftomo                                | 1     |                                                                                |
| 21 | Set up self-hosted GitHub Actions runner on Yarrkha + workflow routing                                     | lftomo                                | 1     | enables Tier-2/3                                                               |
| 22 | Tier-2 datasets — placement on Yarrkha, scan.yaml, reference recons (×3 datasets)                          | lftomo                                | 1     | depends on #21; reference recons generated with current production scripts now |
| E1 | Phase 2 — algorithm composition as Stages                                                                  | lftomo                                | 2     | epic; checklist of Stage classes × algorithm options                           |
| E2 | Phase 3 — lfcapture build-out + hatmehyt capture subcommand                                                | lfcapture / hatmehyt                  | 3     | epic                                                                           |
| E3 | Phase 4 — retire and redirect legacy scripts                                                               | varies                                | 4     | epic                                                                           |

Plus ChengLab consent thread (not an issue per se, but a tracker comment on E1 + future Neuroglancer write Stage):

• Comment / placeholder: ChengLab consent for localCT/recon/super_recon/ (DRHT + LDM) lifts. Doesn't block Phase 1.

What I need from you before filing

1. Token scope refresh. Run on Yarrkha: gh-tava auth refresh -h github.com (http://github.com/) -s read:project,project. Or tell me to proceed without project linkage (issues filed in repos, you batch-add via project auto-rules).
2. Sign-off on the inventory shape above — granularity, two-tier (detail + epic), the temporary-home-in-hatmehyt-for-lfcapture-issues call. Suggest changes / additions / removals.
3. Label strategy: OK with me creating phase:0..4, phase:1.5, plus the topic labels (architecture, algorithm, cli, testing, migration, consent-needed) in each target repo? Or do you have a convention I should match?

Once those are settled, I batch-create labels + issues + (if token refreshed) add to the project, and report back with a link list.