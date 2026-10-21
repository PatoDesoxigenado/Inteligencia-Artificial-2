"""Gera os gráficos (PNG) usados na apresentação de resultados.

Os gráficos são desenhados com a MESMA paleta e tipografia do PPTX, para que a
apresentação e os gráficos do repositório falem a mesma língua visual.

Uso:  python gerar_graficos_slide.py --saida resultados/slide
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# paleta do slide (RGB)
CORES = {
    "lstm": "#2563EB",
    "transformer": "#7C3AED",
    "tgnn": "#0891B2",
    "gru": "#F59E0B",
    "ekf": "#64748B",
    "rf": "#DC2626",
    "base_movimento_constante": "#94A3B8",
}
NOMES = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "tgnn": "T-GNN",
    "gru": "GRU",
    "ekf": "EKF",
    "rf": "Random Forest",
    "base_movimento_constante": "Mov. constante",
}
ORDEM = ["lstm", "transformer", "tgnn", "gru", "ekf", "base_movimento_constante", "rf"]
INK = "#0F172A"
SUAVE = "#64748B"
FUNDO = "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Noto Sans"],
    "axes.edgecolor": "#E2E8F0",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SUAVE,
    "ytick.color": SUAVE,
    "figure.facecolor": FUNDO,
    "axes.facecolor": FUNDO,
})


def ler_resultados(pasta):
    with open(Path(pasta) / "comparacao_modelos.csv", newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    geral = {l["modelo"]: l for l in linhas if not l.get("dominio")}
    dom = {}
    for l in linhas:
        if l.get("dominio"):
            dom.setdefault(l["dominio"], {})[l["modelo"]] = l
    return geral, dom


def grafico_erro_horizontal(geral, destino):
    """Ranking do erro de pose a 0,5 s (barras horizontais, destaque no melhor)."""
    modelos = [m for m in ORDEM if m in geral]
    vals = [float(geral[m]["mae_0_5s"]) for m in modelos]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    y = np.arange(len(modelos))[::-1]
    melhor = int(np.argmin(vals))
    for i, (m, v) in enumerate(zip(modelos, vals)):
        ax.barh(y[i], v, height=0.62, color=CORES[m], alpha=1.0 if i == melhor else 0.55,
                edgecolor="none", zorder=3)
        ax.text(v + 0.004, y[i], f"{v:.3f}", va="center", ha="left", fontsize=10.5,
                color=INK, fontweight="bold" if i == melhor else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([NOMES[m] for m in modelos], fontsize=10.5, color=INK)
    ax.set_xlabel("erro médio da pose  (alturas do corpo)  ↓ menor é melhor",
                  fontsize=10, color=SUAVE, labelpad=8)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(destino, bbox_inches="tight", facecolor=FUNDO)
    plt.close(fig)


def grafico_oclusao(geral, destino):
    """Erro sob oclusão x à vista (2 s) - barras agrupadas."""
    modelos = [m for m in ORDEM if m in geral]
    vis = [float(geral[m].get("mae_visivel_2_0s", "nan")) for m in modelos]
    ocl = [float(geral[m].get("mae_ocluido_2_0s", "nan")) for m in modelos]
    x = np.arange(len(modelos))
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    ax.bar(x - 0.21, vis, 0.40, label="à vista", color="#BFDBFE", edgecolor="none", zorder=3)
    ax.bar(x + 0.21, ocl, 0.40, label="sob oclusão", color="#F97316", edgecolor="none", zorder=3)
    for i in range(len(modelos)):
        ax.text(x[i] - 0.21, vis[i] + 0.012, f"{vis[i]:.2f}", ha="center", fontsize=8.2, color=SUAVE)
        ax.text(x[i] + 0.21, ocl[i] + 0.012, f"{ocl[i]:.2f}", ha="center", fontsize=8.2,
                color="#9A3412", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([NOMES[m] for m in modelos], fontsize=9.2, color=INK, rotation=18,
                       ha="right")
    ax.set_ylabel("erro a 2 s  (alturas do corpo)", fontsize=10, color=SUAVE)
    ax.set_ylim(0, max(ocl + vis) * 1.16)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(destino, bbox_inches="tight", facecolor=FUNDO)
    plt.close(fig)


def grafico_curva(pasta, destino):
    """Erro vs. duração da oclusão."""
    with open(Path(pasta) / "erro_vs_oclusao.csv", newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    faixas = []
    for l in linhas:
        if l["faixa"] not in faixas:
            faixas.append(l["faixa"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    # o Random Forest fica de fora: o erro dele (2,2) achataria a escala e a leitura
    # da tendência das redes. Ele já aparece no gráfico de barras ao lado.
    for m in [x for x in ORDEM if x != "rf"]:
        pts = [(l["faixa"], float(l["mae"]) if l["mae"] else None) for l in linhas if l["modelo"] == m]
        pts = [(f, v) for f, v in pts if v is not None]
        if not pts:
            continue
        ys = [v for _, v in pts]
        ax.plot(range(len(pts)), ys, label=NOMES[m], color=CORES[m], marker="o",
                markersize=6 if m == "lstm" else 4.5,
                linewidth=2.4 if m == "lstm" else 1.7,
                zorder=5 if m == "lstm" else 3)
    ax.set_xticks(range(len(faixas)))
    ax.set_xticklabels([f"{f.split('-')[0]}–{f.split('-')[1]} s" for f in faixas], fontsize=9.5)
    ax.set_xlabel("há quanto tempo a pessoa está escondida", fontsize=10, color=SUAVE, labelpad=6)
    ax.set_ylabel("erro da pose  (alturas do corpo)", fontsize=10, color=SUAVE)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(frameon=False, fontsize=8.6, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(destino, bbox_inches="tight", facecolor=FUNDO)
    plt.close(fig)


def grafico_horizontes(geral, destino):
    """Erro por horizonte (0,5 / 1,0 / 2,0 s) - linhas."""
    modelos = [m for m in ORDEM if m in geral]
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    for m in modelos:
        ys = [float(geral[m][f"mae_{h}"]) for h in ("0_5s", "1_0s", "2_0s")]
        ax.plot(range(3), ys, marker="o", markersize=6 if m == "lstm" else 4.5,
                linewidth=2.6 if m == "lstm" else 1.8, color=CORES[m],
                label=NOMES[m], zorder=5 if m == "lstm" else 3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["0,5 s", "1,0 s", "2,0 s"], fontsize=10.5)
    ax.set_ylabel("erro da pose  (alturas do corpo)", fontsize=10, color=SUAVE)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(frameon=False, fontsize=8.6, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(destino, bbox_inches="tight", facecolor=FUNDO)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resultados", default="resultados")
    p.add_argument("--saida", default="resultados/slide")
    args = p.parse_args()
    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    geral, dom = ler_resultados(args.resultados)
    grafico_erro_horizontal(geral, out / "erro_ranking.png")
    grafico_oclusao(geral, out / "erro_oclusao.png")
    grafico_curva(args.resultados, out / "erro_curva.png")
    grafico_horizontes(geral, out / "erro_horizontes.png")
    print(f"Gráficos do slide salvos em {out}")


if __name__ == "__main__":
    main()
