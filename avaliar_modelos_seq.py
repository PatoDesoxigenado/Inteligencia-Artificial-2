"""Avalia e COMPARA os modelos temporais (LSTM, GRU, EKF, Transformer, T-GNN + RF).

O que é medido (no conjunto de TESTE):
  1. Erro de pose por horizonte: MAE e RMSE em unidades de altura do corpo
     (comparável entre vídeo real e sintético) e em pixels, para 0,5 s, 1,0 s e 2,0 s.
  2. Erro SOB OCLUSÃO: quanto o modelo erra quando a pessoa está escondida
     (medido no sintético, que tem a verdade absoluta da pose oculta).
  3. Recuperação pós-oclusão: erro logo depois de a pessoa reaparecer.
  4. Comparação com a base de movimento constante (sem nenhum modelo).

Saídas (pasta resultados/):
  comparacao_modelos.csv   tabela com todas as métricas por modelo
  comparacao_modelos.png   gráfico de erro por horizonte
  erro_vs_oclusao.png      gráfico de erro vs. duração da oclusão
  avaliacao_modelos.txt    relatório em texto

Uso:
  python avaliar_modelos_seq.py --dataset dados/dataset.npz --modelos resultados/modelos
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from modelos_temporais import (EKF, GRUPose, LSTMPose, N_ARTICULACOES, RFPose,
                                TGNNPose, TransformerPose, forma_relativa)

# horizontes de avaliação (em segundos) e o quadro correspondente a 25 fps
FPS = 25.0


def carregar_modelos(pasta, T, H):
    import torch
    pasta = Path(pasta)
    saida = {}
    fabricas = {
        "lstm": LSTMPose, "gru": GRUPose,
        "transformer": TransformerPose, "tgnn": TGNNPose,
    }
    for nome, cls in fabricas.items():
        arq = pasta / f"{nome}.pt"
        if not arq.exists():
            continue
        d = torch.load(arq, map_location="cpu", weights_only=False)
        m = cls(d.get("T", T), d.get("H", H))
        m.load_state_dict(d["estado"])
        m.eval()
        saida[nome] = m
    arq = pasta / "rf.joblib"
    if arq.exists():
        import joblib
        saida["rf"] = joblib.load(arq)
    return saida


def prever_ekf(entradas, base, H, T):
    """Roda o EKF quadro a quadro sobre a sequência de entrada de cada amostra e
    devolve a pose prevista em todos os horizontes."""
    saida = np.zeros((len(entradas), H, N_ARTICULACOES, 2), np.float32)
    for n, ent in enumerate(entradas):
        ekf = EKF()
        for t in range(len(ent)):
            pose = ent[t, :, :2]
            mask = ent[t, :, 2]
            pres = ent[t, 3, 0]
            if pres > 0:
                ekf.passo(pose, mask, dt=1.0)
            else:
                ekf.passo(None, None, dt=1.0)
        for h in range(1, H + 1):
            p = ekf.prever_pose(h)
            if p is not None:
                saida[n, h - 1] = p
            else:
                saida[n, h - 1] = base[n, h - 1]
    return saida


def prever_base(base):
    """Base de movimento constante: a própria base do dataset."""
    return base.copy()


def metricas(pred, alvo, sup, vis, ocl, escala_janela, horizonte_quadros):
    """Erro de pose: MAE/RMSE em unidades de altura do corpo e em pixels.

    sup  -> máscara de supervisão (alvo confiável)
    vis  -> a articulação foi vista no quadro alvo
    ocl  -> a pessoa está escondida nesse quadro alvo
    """
    res = {}
    d = pred - alvo
    sq = (d ** 2).sum(axis=-1)          # (N, H, J) erro quadrático por articulação
    dist = np.sqrt(sq)

    for nome, idx in horizonte_quadros.items():
        m = sup[:, idx] > 0
        if m.sum() == 0:
            continue
        d_idx = dist[:, idx]                      # (N, J)
        e = d_idx[m]
        esc = np.repeat(escala_janela[:, None], d_idx.shape[1], axis=1)[m]
        res[f"mae_{nome}"] = float(e.mean())
        res[f"rmse_{nome}"] = float(np.sqrt((e ** 2).mean()))
        res[f"mae_px_{nome}"] = float((e * esc).mean())
        res[f"mediana_{nome}"] = float(np.median(e))
        # só nos alvos VISTOS (a câmera realmente mediu) e só nos ESCONDIDOS
        mv = m & (vis[:, idx] > 0)
        if mv.sum():
            res[f"mae_visivel_{nome}"] = float(dist[:, idx][mv].mean())
        mo = m & (ocl[:, idx] > 0)
        if mo.sum():
            res[f"mae_ocluido_{nome}"] = float(dist[:, idx][mo].mean())
            res[f"n_ocluido_{nome}"] = int(mo.sum())
    return res


def recuperacao(pred, alvo, sup, vis, ocl, horizonte_quadros):
    """Erro logo depois de a pessoa REAPARECER.

    Para cada amostra, procura o primeiro quadro do alvo em que a pessoa voltou a
    aparecer depois de um trecho ocluído e mede o erro ali. É a métrica de
    "recuperação pós-oclusão": o modelo_previu bem a pose que a pessoa tinha ao
    voltar à vista?
    """
    res = {}
    for h_s, idx in horizonte_quadros.items():
        # considera a janela de até 10 quadros em torno do horizonte
        i0, i1 = max(0, idx - 9), min(ocl.shape[1], idx + 10)
        d = np.sqrt(((pred[:, i0:i1] - alvo[:, i0:i1]) ** 2).sum(axis=-1))
        vals = []
        for n in range(len(pred)):
            oc = ocl[n, i0:i1, 0] > 0
            if not oc.any():
                continue
            ultimo_oc = int(np.where(oc)[0].max())
            # o primeiro quadro APÓS o trecho ocluído, dentro da janela
            cand = [k for k in range(ultimo_oc + 1, len(oc)) if not oc[k] and sup[n, i0 + k].sum() > 0]
            if not cand:
                continue
            k = cand[0]
            m = sup[n, i0 + k] > 0
            if m.any():
                vals.append(float(d[n, k][m].mean()))
        if vals:
            res[f"mae_recuperacao_{h_s}"] = float(np.mean(vals))
            res[f"n_recuperacao_{h_s}"] = len(vals)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dados/dataset.npz")
    p.add_argument("--modelos", default="resultados/modelos")
    p.add_argument("--saida", default="resultados")
    p.add_argument("--conjunto", default="teste", choices=["teste", "validacao"])
    args = p.parse_args()

    d = np.load(args.dataset, allow_pickle=True)
    cfg = json.loads(str(d["config"]))
    # a origem de cada amostra (1 = sintética, 0 = real) permite comparar por domínio
    chave_origem = f"{args.conjunto}_origem_sintetico"
    origem = d[chave_origem] if chave_origem in d.files else None
    T, H = cfg["T"], cfg["H"]
    ent = d[f"{args.conjunto}_entrada"]
    alvo = d[f"{args.conjunto}_alvo"]
    sup = d[f"{args.conjunto}_sup"]
    vis = d[f"{args.conjunto}_vis"]
    ocl = d[f"{args.conjunto}_ocl"]
    base = d[f"{args.conjunto}_base"]
    print(f"Conjunto {args.conjunto}: {len(ent)} amostras")

    modelos = carregar_modelos(args.modelos, T, H)
    print("Modelos carregados:", list(modelos.keys()))

    horizonte_quadros = {}
    for hs in (0.5, 1.0, 2.0):
        k = int(round(hs * FPS))
        if k <= H:
            horizonte_quadros[f"{str(hs).replace('.', '_')}s"] = k - 1

    # a base/alvo estão normalizados pela altura da pessoa; para voltar a pixels
    # usamos a escala (altura em px) de cada janela
    escala_janela = d[f"{args.conjunto}_escala_px"].astype(np.float64)
    escala_px = float(np.mean(escala_janela))
    print(f"Escala média da pessoa: {escala_px:.1f} px")

    previsoes = {"base_movimento_constante": prever_base(base)}
    import torch
    for nome, m in modelos.items():
        if isinstance(m, RFPose):
            preds = m.prever_lote(ent, base)
        else:
            with torch.no_grad():
                r = m.forward(torch.tensor(ent, dtype=torch.float32)).numpy()
            preds = base + r
        previsoes[nome] = preds
        print(f"  previsto: {nome}")

    previsoes["ekf"] = prever_ekf(ent, base, H, T)
    print("  previsto: ekf")

    # ---------------- métricas ----------------
    linhas = []
    for nome, pred in previsoes.items():
        mm = metricas(pred, alvo, sup, vis, ocl, escala_janela, horizonte_quadros)
        mm["modelo"] = nome
        linhas.append(mm)

    # erro vs. duração da oclusão (no alvo): agrupa pelo tempo que a pessoa já está
    # escondida quando começa a previsão
    colunas = ["modelo", "dominio", "mae_0_5s", "rmse_0_5s", "mae_1_0s", "rmse_1_0s",
               "mae_2_0s", "rmse_2_0s", "mae_visivel_2_0s", "mae_ocluido_2_0s",
               "mae_px_0_5s", "mae_px_1_0s", "mae_px_2_0s",
               "mae_recuperacao_0_5s", "mae_recuperacao_1_0s", "mae_recuperacao_2_0s",
               "n_recuperacao_0_5s", "n_recuperacao_1_0s", "n_recuperacao_2_0s"]
    # (as colunas são filtradas no final, depois de juntar as linhas por domínio)

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    # ---------------- recuperação pós-oclusão ----------------
    for nome, pred in previsoes.items():
        rr = recuperacao(pred, alvo, sup, vis, ocl, horizonte_quadros)
        for l in linhas:
            if l["modelo"] == nome and not l.get("dominio"):
                l.update(rr)
                break

    # ---------------- curva: erro vs. duração da oclusão ----------------
    curva = []
    faixas = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    for nome, pred in previsoes.items():
        for lo, hi in faixas:
            i0, i1 = int(lo * FPS), int(hi * FPS)
            m = ((sup[:, i0:i1].mean(axis=(1, 2)) > 0) &
                 (ocl[:, i0:i1].mean(axis=(1, 2)) > 0))
            if m.sum() == 0:
                curva.append({"modelo": nome, "faixa": f"{lo:.1f}-{hi:.1f}", "mae": None, "n": 0})
                continue
            d = np.sqrt(((pred[m][:, i0:i1] - alvo[m][:, i0:i1]) ** 2).sum(axis=-1))
            curva.append({"modelo": nome, "faixa": f"{lo:.1f}-{hi:.1f}",
                          "mae": float(d.mean()), "n": int(m.sum())})
    with open(saida / "erro_vs_oclusao.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["modelo", "faixa", "mae", "n"])
        w.writeheader()
        w.writerows(curva)

    # ---------------- métricas por domínio (sintético x real) ----------------
    if origem is not None:
        for rot, sel in (("sintetico", origem > 0), ("real", origem == 0)):
            if sel.sum() == 0:
                continue
            print(f"\n-- por domínio: {rot} ({int(sel.sum())} amostras) --")
            for nome, pred in previsoes.items():
                mm = metricas(pred[sel], alvo[sel], sup[sel], vis[sel], ocl[sel],
                              escala_janela[sel], horizonte_quadros)
                mm["modelo"] = nome
                mm["dominio"] = rot
                linhas.append(mm)

    # ---------------- CSV com todas as métricas ----------------
    colunas = [c for c in colunas if any(c in l for l in linhas)]
    with open(saida / "comparacao_modelos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colunas, extrasaction="ignore")
        w.writeheader()
        for l in linhas:
            w.writerow(l)

    # ---------------- relatório em texto ----------------
    nomes = list(previsoes.keys())
    txt = []
    txt.append("=" * 78)
    txt.append("COMPARAÇÃO DOS MODELOS TEMPORAIS - ESTIMAÇÃO DE POSE SOB OCLUSÃO")
    txt.append("=" * 78)
    txt.append(f"Conjunto: {args.conjunto} ({len(ent)} amostras) | entrada {T} quadros (1 s)"
               f" | saída {H} quadros (2 s) | escala {escala_px:.0f} px")
    txt.append("")
    txt.append("ERRO DE POSE POR HORIZONTE (unidades de altura do corpo; menor = melhor)")
    txt.append(f"{'modelo':<26}" + "".join(f"{h:>12}" for h in ["0,5 s", "1,0 s", "2,0 s"]))
    txt.append("-" * 78)
    for l in linhas:
        txt.append(f"{l['modelo']:<26}"
                   + f"{l.get('mae_0_5s', float('nan')):>12.4f}"
                   + f"{l.get('mae_1_0s', float('nan')):>12.4f}"
                   + f"{l.get('mae_2_0s', float('nan')):>12.4f}")
    txt.append("")
    txt.append("ERRO EM PIXELS (mesma ordem)")
    txt.append(f"{'modelo':<26}" + "".join(f"{h:>12}" for h in ["0,5 s", "1,0 s", "2,0 s"]))
    txt.append("-" * 78)
    for l in linhas:
        txt.append(f"{l['modelo']:<26}"
                   + f"{l.get('mae_px_0_5s', float('nan')):>12.1f}"
                   + f"{l.get('mae_px_1_0s', float('nan')):>12.1f}"
                   + f"{l.get('mae_px_2_0s', float('nan')):>12.1f}")
    txt.append("")
    txt.append("SOB OCLUSÃO vs. À VISTA (horizonte 2 s)")
    txt.append(f"{'modelo':<26}{'ocluído':>12}{'à vista':>12}")
    txt.append("-" * 78)
    for l in linhas:
        txt.append(f"{l['modelo']:<26}"
                   + f"{l.get('mae_ocluido_2_0s', float('nan')):>12.4f}"
                   + f"{l.get('mae_visivel_2_0s', float('nan')):>12.4f}")
    relatorio = "\n".join(txt)
    (saida / "avaliacao_modelos.txt").write_text(relatorio, encoding="utf-8")
    print()
    print(relatorio)

    # ---------------- gráficos ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1) erro por horizonte
        fig, ax = plt.subplots(figsize=(8, 4.5))
        hs = ["mae_0_5s", "mae_1_0s", "mae_2_0s"]
        rotulos = ["0,5 s", "1,0 s", "2,0 s"]
        for l in linhas:
            ys = [l.get(h, np.nan) for h in hs]
            ax.plot(range(len(hs)), ys, marker="o", label=l["modelo"])
        ax.set_xticks(range(len(hs)))
        ax.set_xticklabels(rotulos)
        ax.set_xlabel("horizonte de previsão")
        ax.set_ylabel("MAE da pose (alturas do corpo)")
        ax.set_title("Erro de pose por horizonte de previsão")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(saida / "comparacao_modelos.png", dpi=140)
        plt.close(fig)

        # 2) erro vs. duração da oclusão
        #    para cada amostra, em que quadro do alvo a pessoa está escondida há mais
        #    tempo? Usa o primeiro horizonte e classifica por "frames desde o início".
        fig, ax = plt.subplots(figsize=(8, 4.5))
        faixas = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
        for nome, pred in previsoes.items():
            ys = []
            for lo, hi in faixas:
                i0, i1 = int(lo * FPS), int(hi * FPS)
                # amostras cuja janela de alvo está (parcialmente) sob oclusão
                m = ((sup[:, i0:i1].mean(axis=(1, 2)) > 0) &
                     (ocl[:, i0:i1].mean(axis=(1, 2)) > 0))
                if m.sum() == 0:
                    ys.append(np.nan)
                    continue
                d = np.sqrt(((pred[m][:, i0:i1] - alvo[m][:, i0:i1]) ** 2).sum(axis=-1))
                ys.append(float(d.mean()))
            ax.plot([f"{lo:.1f}-{hi:.1f}s" for lo, hi in faixas], ys, marker="o", label=nome)
        ax.set_xlabel("faixa de tempo sob oclusão (s)")
        ax.set_ylabel("MAE da pose (alturas do corpo)")
        ax.set_title("Erro de pose durante a oclusão")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(saida / "erro_vs_oclusao.png", dpi=140)
        plt.close(fig)
        print(f"\nGráficos salvos em {saida}")
    except ImportError:
        print("(matplotlib ausente: gráficos pulados)")


if __name__ == "__main__":
    main()
