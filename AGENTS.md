# AGENTS.md

Curia Task2 workshop harness for this repository root.

## Startup Workflow

Before writing code:
1. Read this file.
2. Read [`curia_t2/README.md`](./curia_t2/README.md).
3. Run `./init.sh`.
4. Read `feature_list.json` and `progress.md`.

## Scope

- Active path: `curia_t2/` only.
- Legacy `CT-NEXUS/` is read-only reference unless explicitly requested.
- One feature at a time; do not start the next feature before recording verification evidence.

## Working Rules

- Always run verification commands before marking a feature `done`.
- Keep modifications minimal and auditable.
- Persist session state in `progress.md` + `feature_list.json`.
- For model download/training requiring network, use proxy only when needed.

## Verification Commands

Run from repo root:

```bash
./init.sh
python -m py_compile curia_t2/*.py
python curia_t2/train_curia_task2.py --help
python curia_t2/predict_curia_task2.py --help
```

Model cache verification:

```bash
python curia_t2/download_curia.py --help
```

## Required Artifacts

- `feature_list.json`: feature tracker
- `progress.md`: session log and decisions
- `session-handoff.md`: ready-to-resume handoff
- `init.sh`: standardized startup checks

## Definition of Done

A feature is done when:
- [ ] Implementation complete
- [ ] Verification commands pass
- [ ] Evidence captured in `feature_list.json`
- [ ] `progress.md` updated with next step

## End-of-Session Checklist

1. Update `progress.md` (what changed, what failed, next step).
2. Update `feature_list.json` status/evidence.
3. Update `session-handoff.md` with exact restart commands.
4. Ensure repository has a clean restart path.
