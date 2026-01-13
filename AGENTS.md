# Repository Guidelines

## Architecture Overview
This repository holds the project brief and reading references for the **Atari Games** ML project. As implementation is added, keep a clear separation between environment setup, agent definition, training, evaluation, and reporting. Goal: models for CartPole, Space Invaders, and Pacman, plus a short blog post.

## Project Structure & Module Organization
Current structure (source‑of‑truth inputs):
- `atari-games-instructions/`: project brief/metadata.
- `doc-referances/`: background PDFs.
- `README.md`: top‑level description.

Recommended architecture as code is added:
- `src/`: core library (agents, networks, utilities).
- `train/`, `eval/`: entry points and metrics.
- `configs/`: experiment configs.
- `notebooks/`: exploration only.
- `models/`, `reports/`: checkpoints (ignored) and write‑ups.

Document new top‑level directories in `README.md`.

## Build, Test, and Development Commands
No build/test scripts are defined yet. If you add tooling, standardize entry points (e.g., `Makefile` or `scripts/`) and keep commands deterministic. Example patterns:
- `python -m venv .venv && source .venv/bin/activate` — create and activate a virtual environment.
- `pip install -r requirements.txt` — install dependencies.
- `pytest` — run tests (if added).

## Coding Style & Naming Conventions
No formatting rules are enforced yet. For Python, follow PEP 8, use 4‑space indentation, and add a formatter/linter (e.g., `black`, `ruff`) with a short usage note. Prefer explicit, task‑oriented names like `train_cartpole.py`, `dqn_agent.py`, and `eval_space_invaders.py`.

## Testing Guidelines
There are no tests in the repository today. If you add tests, place them in `tests/` and use `test_*.py` naming. Cover critical components and add at least one smoke test per training entry point.

## Commit & Pull Request Guidelines
The Git history does not show a commit message convention yet. Use concise, imperative messages (e.g., “Add DQN baseline for CartPole”). For PRs, include:
- A short summary of changes and which game/model it targets.
- Links to related issues or notes.
- Reproducibility details (seeds, hyperparameters, and environment versions).
- Results/metrics and a brief comparison to prior runs.

## Data, Artifacts, and Reproducibility
Avoid committing large datasets or model checkpoints directly. Prefer external storage and document paths/links in `README.md`. Record environment details (`requirements.txt` or `environment.yml`), fix random seeds, and log training configs.

## Agent-Specific Instructions
When writing or modifying code for this project, consult and incorporate the referenced documents in `doc-referances/` (all PDFs) and the brief in `atari-games-instructions/atari-games.md`. Treat these as required sources of guidance and align implementations, terminology, and citations/notes with them.
