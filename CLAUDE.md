# local-decision: working rules for agents

`decide` runs local decision models (Cloudflare's Clef and Clef-Flash) through the
Jev/SystemOne API. Read `README.md` for usage, then `models.toml`, which holds every
model. Research notes and sources are kept in a private wiki, not in this repo.

## Layout

| path | what |
|---|---|
| `bin/decide` | the CLI: list, gpu, serve, status, ask, smoke, stop, test, fetch (symlinked into `~/.local/bin`) |
| `models.toml` | the registry: paths, engines (command templates), one row per target |
| `tests/smoke.json` | planted controls; `tests/*.png` are the image controls |
| `bench/runs.tsv` | one line per `decide smoke` run: the only evidence for a row's `trust` |
| `vendor/`, `.venv/`, `run/` | gitignored: llama.cpp build, MLX venv, live server state and logs |

Weights live in `~/models/<name>`, outside the repo.

## Rules

1. **`models.toml` is the only place a model is named.** A new decision model is a row
   (engine, weights, repo, revision, ctx, inputs, trust). A new engine is an
   `[engines.<name>]` command template. Never hard-code a model, path or flag in
   `bin/decide`.
2. **Decision models stay out of any chat-model front end.** If the Mac runs llama-swap (or
   another router) for chat models, never add a decision model to it: every model there is
   offered to chat clients, and a Clef server cannot chat.
3. **One GPU job at a time.** `decide serve` asks `m3d gpu` and refuses while anything
   holds the GPU. Don't pass `--force` while another session is rendering or serving.
4. **Never SIGKILL a model server.** It leaks wired Metal memory until reboot. `decide
   stop` sends SIGINT, then SIGTERM, and gives up rather than escalate.
5. **Trust comes from evidence.** Change a row's `trust` to `verified N/N` only on the
   evidence of a `bench/runs.tsv` line from `decide smoke`.
6. **Pin what you download.** Every row carries `repo` and `revision`; only ggml-org
   GGUFs (made with llama.cpp b11371 or later) carry Clef's decision head.
7. **Commit as you go**, one logical change per commit, with the message prefix
   `local-decision:`.
