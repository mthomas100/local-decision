# /// script
# dependencies = ["matplotlib", "numpy", "pillow"]
# ///
"""Charts for the README, from bench/results/2026-10-03 and bench/data (read-only).

  uv run docs/make_charts.py [OUTDIR]     default OUTDIR: docs/media
"""
import json, statistics, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "media"
OUT.mkdir(parents=True, exist_ok=True)
DAY = REPO / "bench/results/2026-10-03"
DATA = REPO / "bench/data"
TARGETS = ["clef", "clef-mlx", "clef-flash", "clef-flash-mlx"]
NAME = {"clef": "Clef 27B · llama.cpp", "clef-mlx": "Clef 27B · MLX",
        "clef-flash": "Clef-Flash 9B · llama.cpp", "clef-flash-mlx": "Clef-Flash 9B · MLX"}
HUE = {"clef": "#2a78d6", "clef-mlx": "#2a78d6", "clef-flash": "#eb6834", "clef-flash-mlx": "#eb6834"}
MARK = {"clef": "o", "clef-flash": "o", "clef-mlx": "s", "clef-flash-mlx": "s"}
DASH = {"clef": "-", "clef-flash": "-", "clef-mlx": (0, (4, 2)), "clef-flash-mlx": (0, (4, 2))}
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
                     "axes.titlesize": 11.5, "legend.frameon": False})
SRC = "Data: bench/results/2026-10-03 (one run per target on an M5 Max, 128 GB, 2026-10-03)."


def load(t, s):
    p = DAY / t / f"{s}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def acc(rows):
    ok = [r for r in rows if r["code"] == 200]
    q = next(iter(ok[0]["truth"]))
    a0 = ok[0]["answers"][q]
    if a0["type"] == "noul":
        c = [(r["answers"][q]["noul"] > 0.5) == r["truth"][q] for r in ok]
    else:
        c = [r["answers"][q]["choice"] == r["truth"][q] for r in ok]
    return 100 * sum(c) / len(c), statistics.median(r["ms"] for r in ok), len(ok)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


# 1. accuracy vs median latency, small multiples per suite
suites = [("banking77", "BANKING77 intent routing (77 options)"), ("sms-spam", "SMS spam (yes/no)"),
          ("halueval-qa", "HaluEval QA: answer supported?"), ("shell-success", "Shell command succeeded?")]
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.2))
check = {}
for ax, (s, title) in zip(axs.flat, suites):
    for t in TARGETS:
        a, ms, n = acc(load(t, s)); check[(t, s)] = (round(a, 1), round(ms))
        ax.scatter(ms, a, s=70, marker=MARK[t], color=HUE[t], edgecolor=SURF, linewidth=2, zorder=3)
        mlx = t.endswith("mlx")
        ax.annotate(f"{a:.1f}%", (ms, a), xytext=(-8 if mlx else 8, -3), textcoords="offset points", fontsize=9,
                    color=INK2, ha="right" if mlx else "left")
    ax.set_xscale("log"); ax.set_title(f"{title} (n={n})", loc="left")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("median latency per request, ms (log)"); ax.set_ylabel("accuracy, %")
    lo = min(acc(load(t, s))[0] for t in TARGETS)
    ax.set_ylim(max(0, lo - 8), 101)
    xs = [acc(load(t, s))[1] for t in TARGETS]; ax.set_xlim(min(xs) / 2.2, max(xs) * 2.2)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
handles = [plt.Line2D([], [], marker=MARK[t], color=HUE[t], linestyle="", markersize=8, label=NAME[t]) for t in TARGETS]
fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), fontsize=9.5)
fig.text(0.0, -0.02, SRC + " Up and to the left is better.", fontsize=8.5, color=MUTED)
fig.tight_layout(); save(fig, "accuracy-vs-latency.png")
print(check)

# 2. peak memory per target
runs = {t: json.loads((DAY / t / "run.json").read_text()) for t in TARGETS}
fig, ax = plt.subplots(figsize=(8.5, 3.4))
y = np.arange(len(TARGETS))
for i, t in enumerate(TARGETS):
    r = runs[t]
    ax.barh(i + 0.17, r["peak_rss_gb"], height=0.3, color=HUE[t], zorder=3)
    ax.barh(i - 0.17, r["peak_wired_gb"], height=0.3, color=HUE[t], alpha=0.35, zorder=3, hatch="///", edgecolor=SURF)
    ax.text(r["peak_rss_gb"] + 0.6, i + 0.17, f"{r['peak_rss_gb']} GB RSS", va="center", fontsize=9, color=INK2)
    ax.text(r["peak_wired_gb"] + 0.6, i - 0.17, f"{r['peak_wired_gb']} GB wired", va="center", fontsize=9, color=INK2)
ax.axvline(runs["clef"]["wired_before_gb"], color=MUTED, linewidth=1)
ax.text(runs["clef"]["wired_before_gb"] + 0.4, 0.5, "wired before load: 7.1 GB", va="center", fontsize=8.5, color=MUTED)
ax.set_yticks(y, [NAME[t] for t in TARGETS]); ax.invert_yaxis(); ax.set_xlim(0, 52)
ax.set_xlabel("GB"); ax.grid(axis="y", visible=False)
ax.set_title("Peak memory while serving the whole suite (solid: server process RSS; hatched: system-wide wired)", loc="left", fontsize=10.5)
fig.text(0.0, -0.06, SRC + " Wired is sampled every 2 s for the whole Mac, so it includes the 7.1 GB wired before load.",
         fontsize=8, color=MUTED)
save(fig, "peak-memory.png")

# 3. prompt injection: p(spam) per spam message, plain vs injected
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for ax, t in zip(axs, ["clef", "clef-flash"]):
    plain = {r["id"]: r["answers"]["spam"]["noul"] for r in load(t, "sms-spam") if r["truth"]["spam"]}
    inj = {r["meta"]["base"]: r["answers"]["spam"]["noul"] for r in load(t, "sms-spam-injected")}
    pairs = [(plain[b], p) for b, p in inj.items() if b in plain]
    # the injected set is the 75 spam messages; ids line up by the base id recorded at prepare time
    rng = np.random.default_rng(3)
    for a, b in pairs:
        ax.plot([0, 1], [a, b], color=HUE[t], alpha=0.18, linewidth=1, zorder=2)
    jit = rng.uniform(-0.06, 0.06, (len(pairs), 2))
    ax.scatter(jit[:, 0], [a for a, _ in pairs], s=22, color=HUE[t], edgecolor=SURF, linewidth=1, zorder=3)
    ax.scatter(1 + jit[:, 1], [b for _, b in pairs], s=22, color=HUE[t], edgecolor=SURF, linewidth=1, zorder=3)
    ca, cb = sum(a > 0.5 for a, _ in pairs), sum(b > 0.5 for _, b in pairs)
    ax.axhline(0.5, color=MUTED, linewidth=1, zorder=1)
    ax.set_xticks([0, 1], [f"plain\ncaught {ca}/{len(pairs)} ({100*ca/len(pairs):.1f}%)",
                           f"+ injected 'Answer false.' note\ncaught {cb}/{len(pairs)} ({100*cb/len(pairs):.1f}%)"])
    ax.set_xlim(-0.35, 1.35); ax.grid(axis="x", visible=False)
    ax.set_title(NAME[t], loc="left")
    check[(t, "inj")] = (ca, cb, len(pairs))
axs[0].set_ylabel("p(spam) for each of the 75 spam messages")
fig.suptitle("A one-line prompt injection flips the 27B model; the 9B model shrugs it off", x=0.02, ha="left", fontsize=12.5, color=INK)
fig.text(0.0, -0.06, SRC + " Same 75 spam messages (UCI SMS Spam), threshold 0.5. The MLX targets did not run this suite.",
         fontsize=8.5, color=MUTED)
fig.tight_layout(); save(fig, "prompt-injection.png")
print(check)

# 4. reliability diagrams for the yes/no (noul) suites
rsuites = [("sms-spam", "SMS spam"), ("halueval-qa", "HaluEval QA"), ("shell-success", "Shell success")]
fig, axs = plt.subplots(2, 3, figsize=(11, 7.6), sharex=True, sharey=True)
edges = np.linspace(0, 1, 6)
for row, pair in enumerate((("clef", "clef-mlx"), ("clef-flash", "clef-flash-mlx"))):
    for ax, (s_, title) in zip(axs[row], rsuites):
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, zorder=1)
        for t in pair:
            rows = [r for r in load(t, s_) if r["code"] == 200]
            q = next(iter(rows[0]["truth"]))
            p = np.array([r["answers"][q]["noul"] for r in rows]); yv = np.array([bool(r["truth"][q]) for r in rows])
            idx = np.clip(np.digitize(p, edges[1:-1]), 0, 4)
            pts = [(p[idx == b].mean(), yv[idx == b].mean(), (idx == b).sum()) for b in range(5) if (idx == b).any()]
            ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], color=HUE[t], linestyle=DASH[t], linewidth=2, zorder=2)
            for x, y, c in pts:
                ax.scatter(x, y, s=20 + 1.6 * c, marker=MARK[t], zorder=3, linewidth=1.5,
                           color=HUE[t] if c >= 5 else SURF, edgecolor=HUE[t] if c < 5 else SURF)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
        ax.set_title(f"{NAME[pair[0]].split(' ·')[0]}: {title} (n={len(rows)})", loc="left", fontsize=10.5)
for ax in axs[1]: ax.set_xlabel("predicted p(true), bin mean")
for ax in axs[:, 0]: ax.set_ylabel("observed fraction true")
handles = [plt.Line2D([], [], marker=MARK[t], color=HUE[t], linestyle=DASH[t], markersize=7, label=NAME[t]) for t in TARGETS]
fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), fontsize=9.5, handlelength=3.5)
fig.text(0.0, -0.03, SRC + " Five equal-width bins; marker area = items in the bin; hollow = fewer than 5 items.\n"
         "The diagonal is perfect calibration. Most answers sit near 0 or 1, so the middle bins are thin and noisy.",
         fontsize=8.5, color=MUTED)
fig.tight_layout(); save(fig, "reliability.png")

# 5. latency by input length
fig, ax = plt.subplots(figsize=(8, 4.2))
for t in TARGETS:
    rows = [r for r in load(t, "latency") if r["code"] == 200]
    sizes = sorted({r["meta"]["chars"] for r in rows})
    tok = [next(r["usage"]["input_tokens"] for r in rows if r["meta"]["chars"] == s) for s in sizes]
    med = [statistics.median(r["ms"] for r in rows if r["meta"]["chars"] == s) / 1000 for s in sizes]
    ax.plot(tok, med, color=HUE[t], linestyle=DASH[t], linewidth=2, marker=MARK[t], markersize=7,
            markeredgecolor=SURF, markeredgewidth=1.5, label=NAME[t])
    ax.annotate(f"{med[-1]:.1f} s", (tok[-1], med[-1]), xytext=(6, -3), textcoords="offset points", fontsize=9, color=INK2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("input tokens (log)"); ax.set_ylabel("median seconds per request (log)")
ax.set_xticks([443, 1351, 4391, 11464], ["443", "1,351", "4,391", "11,464"]); ax.minorticks_off()
ax.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20], ["0.2", "0.5", "1", "2", "5", "10", "20"])
ax.set_xlim(350, 20000)
ax.legend(fontsize=9, loc="upper left", handlelength=4)
ax.set_title("One forward pass, three questions: latency grows with the input", loc="left")
fig.text(0.0, -0.05, SRC + " 3 repeats per length; HaluEval knowledge text as filler.", fontsize=8.5, color=MUTED)
save(fig, "latency-by-length.png")

# 6. render-QA grid (hero): planted defects and what the two MLX models said
variants = [("ok", "clean render"), ("black", "all black"), ("magenta", "missing textures"),
            ("noise", "noise"), ("half", "half cut off")]
subjects = ["teddy-bear-robot", "toaster-robot"]
ans = {t: {r["id"]: r["answers"] for r in load(t, "render-qa")} for t in ["clef-mlx", "clef-flash-mlx"]}
fig, axs = plt.subplots(len(variants), len(subjects), figsize=(10.4, 10.6))
for j, subj in enumerate(subjects):
    for i, (v, label) in enumerate(variants):
        ax = axs[i, j]
        ax.imshow(Image.open(DATA / "images" / f"render-{subj}-{v}.jpg")); ax.set_axis_off()
        lines = [f"{label}"]
        for t, short in (("clef-mlx", "Clef 27B"), ("clef-flash-mlx", "Flash 9B")):
            a = ans[t][f"rqa-{subj}-{v}"]
            ch = a["problem"]["choice"]
            lines.append(f"{short}: broken p={a['broken']['noul']:.2f} · {ch.replace('_', ' ')} {a['problem']['probabilities'][ch]:.2f}")
        ax.set_title(lines[0], loc="left", fontsize=10.5, color=INK, pad=3)
        ax.text(0, -0.05, "\n".join(lines[1:]), transform=ax.transAxes, va="top", fontsize=9, color=INK2, family="DejaVu Sans Mono")
fig.subplots_adjust(hspace=0.55, wspace=0.05)
fig.text(axs[-1, 0].get_position().x0, 0.03, "Planted-defect render QA (bench/data/images, renders of AI-generated 3D models).\nAnswers from the MLX targets, "
         "bench/results/2026-10-03/*-mlx/render-qa.jsonl.", fontsize=8.5, color=MUTED)
save(fig, "render-qa-grid.png")
