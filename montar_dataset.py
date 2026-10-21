"""Monta o dataset comum de sequências para a comparação dos modelos temporais.

Cada vídeo (sintético ou real) vira uma "trilha" (track) com, por quadro:
  - a pose OBSERVADA: 13 articulações (x, y) e um indicador de visibilidade por ponto;
  - a POSE LATENTE (verdade absoluta): existe só no sintético - é ela que permite
    medir o erro durante a oclusão, quando a pessoa está escondida atrás da pilastra.

Uma amostra é uma janela:
  entrada : T quadros passados da pose observada, com a máscara de visibilidade
  saída   : H quadros futuros da pose, com a máscara de SUPERVISÃO

Máscaras (o que cada uma serve para):
  vis  -> a articulação foi vista nesse quadro (medição do sensor)
  sup  -> existe alvo confiável para treinar/avaliar nesse ponto:
          sintético: a pessoa presente (mesmo escondida, a verdade absoluta é conhecida)
          real:     a pessoa presente E a articulação vista (não há GT escondido)
  ocl  -> a pessoa está presente mas escondida (cobertura baixa) -> erro SOB OCLUSÃO

Normalização: por janela, com a origem e a escala (altura da pessoa) do último quadro
da entrada em que a pessoa estava visível -> o modelo não depende da resolução nem do
tamanho da pessoa, e o alvo sai no mesmo referencial.

Partição: cenas sintéticas divididas POR CENA (sem vazamento) e vídeo real dividido no
TEMPO (60% treino / 20% validação / 20% teste).

Uso:
  python montar_dataset.py --sinteticos dados/sinteticos --real dados/pose_real.csv \
      --saida dados/dataset.npz
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cena_corredor import N_ARTICULACOES, NOMES_ARTICULACOES


# ----------------------------------------------------------------------------------
# LEITURA DAS TRILHAS
# ----------------------------------------------------------------------------------
def _montar_trilha(linhas, tem_latente):
    n = len(linhas)
    latente = np.zeros((n, N_ARTICULACOES, 2), np.float32)
    visivel = np.zeros((n, N_ARTICULACOES, 2), np.float32)
    mask = np.zeros((n, N_ARTICULACOES), np.float32)
    presente = np.zeros(n, bool)
    visivel_frame = np.zeros(n, bool)
    cobertura = np.zeros(n, np.float32)
    fps = 25.0

    for i, r in enumerate(linhas):
        if r.get("fps"):
            fps = float(r["fps"])
        cobertura[i] = float(r.get("cobertura") or 0.0)
        # "presente" = a pessoa EXISTE na cena (pode estar totalmente escondida: a
        # verdade absoluta da pose existe mesmo assim). "visivel" = a câmera a vê.
        presente[i] = int(r.get("na_cena", r.get("visivel") or 0)) == 1
        visivel_frame[i] = int(r.get("visivel") or 0) == 1
        for j, nome in enumerate(NOMES_ARTICULACOES):
            x = float(r[f"{nome}_x"])
            y = float(r[f"{nome}_y"])
            v = int(r[f"{nome}_vis"])
            latente[i, j] = (x, y)
            visivel[i, j] = (x, y) if v else (0.0, 0.0)
            mask[i, j] = v
    return {"latente": latente, "visivel": visivel, "mask": mask, "presente": presente,
            "visivel_frame": visivel_frame, "cobertura": cobertura, "fps": fps,
            "tem_latente": tem_latente}


def ler_trilha_sintetica(caminho):
    """GT do vídeo sintético: pose latente (sempre) + pose visível (a que a câmera vê)."""
    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    return _montar_trilha(linhas, tem_latente=True)


def ler_trilha_real(caminho):
    """Vídeo real: só existe a pose ESTIMADA (MediaPipe)."""
    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    return _montar_trilha(linhas, tem_latente=False)


# ----------------------------------------------------------------------------------
# JANELAS
# ----------------------------------------------------------------------------------
def _referencia(visivel, mask, i0, i1):
    """Último quadro da entrada com a pessoa visível -> (origem x,y ; escala ; índice)."""
    for i in range(i1 - 1, i0 - 1, -1):
        if mask[i].sum() >= 4:
            quadris = visivel[i][7:9].mean(axis=0)
            altura = float(np.linalg.norm(visivel[i][0] - quadris))
            if altura > 1e-3:
                return quadris, altura, i
    return None, 0.0, -1


def _velocidades(visivel, mask, i0, i1, j, dt):
    """Velocidade por quadro da articulação j (em px/quadro), usando os dois últimos
    pontos vistos na entrada. Devolve (vx, vy, x_ultimo, y_ultimo)."""
    pts = [(i, visivel[i, j]) for i in range(i0, i1) if mask[i, j] > 0]
    if not pts:
        return 0.0, 0.0, 0.0, 0.0
    i_last, p_last = pts[-1]
    if len(pts) >= 2:
        i_prev, p_prev = pts[-2]
        gap = max(1, i_last - i_prev)
        return (p_last[0] - p_prev[0]) / gap, (p_last[1] - p_prev[1]) / gap, p_last[0], p_last[1]
    return 0.0, 0.0, p_last[0], p_last[1]


def _base_rigida(visivel, mask, i0, i1, H, origem, escala):
    """Base de movimento constante ROBUSTA: a pessoa se move como um corpo rígido.

    A velocidade vem do QUADRIL (o centro do corpo), não de cada articulação: assim,
    quando uma articulação está escondida há vários quadros, a sua velocidade "velha"
    não sai do controle. A forma do corpo (o quanto cada articulação está do quadril)
    é mantida do último quadro em que ela foi vista.

    Devolve (H, J, 2) normalizado pela escala da janela.
    """
    # velocidade do quadril (centro dos quadris visíveis) nos últimos quadros
    centros = []
    for i in range(i0, i1):
        q = [visivel[i, 7], visivel[i, 8]]
        idx = [k for k in (7, 8) if mask[i, k] > 0]
        if idx:
            centros.append((i, visivel[i, idx].mean(axis=0)))
    if len(centros) >= 2:
        (i_a, c_a), (i_b, c_b) = centros[-2], centros[-1]
        gap = max(1, i_b - i_a)
        vx, vy = (c_b[0] - c_a[0]) / gap, (c_b[1] - c_a[1]) / gap
    elif len(centros) == 1:
        vx = vy = 0.0
    else:
        vx = vy = 0.0

    # forma do corpo: offset de cada articulação em relação ao quadril, normalizado
    forma = np.zeros((N_ARTICULACOES, 2), np.float32)
    tem_forma = np.zeros(N_ARTICULACOES, bool)
    for j in range(N_ARTICULACOES):
        for i in range(i1 - 1, i0 - 1, -1):
            if mask[i, j] > 0:
                idx = [k for k in (7, 8) if mask[i, k] > 0]
                if idx:
                    c = visivel[i, idx].mean(axis=0)
                    forma[j] = (visivel[i, j] - c) / escala
                    tem_forma[j] = True
                break

    # posição atual do quadril (para ancorar a extrapolação)
    if centros:
        c0 = centros[-1][1]
    else:
        c0 = np.array(origem, np.float32)

    base = np.zeros((H, N_ARTICULACOES, 2), np.float32)
    for k in range(H):
        passo = k + 1
        c_prev = (c0 + np.array([vx, vy]) * passo - origem) / escala
        for j in range(N_ARTICULACOES):
            base[k, j] = c_prev + forma[j] * tem_forma[j]
    return base


def janelas(trilha, T, H, passo):
    """Gera as janelas (janelas deslizantes) de uma trilha."""
    latente = trilha["latente"]
    visivel = trilha["visivel"]
    mask = trilha["mask"]
    presente = trilha["presente"]
    cobertura = trilha["cobertura"]
    n = len(latente)
    passo_real = max(1, int(round(passo)))
    saida = []

    for t0 in range(0, n - T - H + 1, passo_real):
        i_ini, i_fim = t0, t0 + T
        origem, escala, i_ref = _referencia(visivel, mask, i_ini, i_fim)
        if i_ref < 0:
            continue

        # ---------------- entrada ----------------
        ent = np.zeros((T, N_ARTICULACOES, 4), np.float32)
        v_ent = visivel[i_ini:i_fim].copy()
        v_ent[mask[i_ini:i_fim] == 0] = 0.0
        ent[:, :, :2] = (v_ent - origem) / escala
        ent[:, :, 2] = mask[i_ini:i_fim]
        ent[:, :, 3] = presente[i_ini:i_fim][:, None]

        # ---------------- alvo + máscara de supervisão ----------------
        i_sai = slice(t0 + T, t0 + T + H)
        alvo = (latente[i_sai] - origem) / escala
        pres_sai = presente[i_sai]
        vis_sai = mask[i_sai]

        if trilha["tem_latente"]:
            sup = np.repeat(pres_sai[:, None].astype(np.float32), N_ARTICULACOES, axis=1)
        else:
            sup = (pres_sai[:, None] * vis_sai).astype(np.float32)  # só onde foi visto

        # oclusão no alvo: a pessoa está lá, mas escondida
        ocl = ((pres_sai) & (cobertura[i_sai] < 0.35)).astype(np.float32)[:, None]  # (H, 1)

        # ---------------- base de movimento constante (corpo rígido) ----------------
        base = _base_rigida(visivel, mask, i_ini, i_fim, H, origem, escala)

        saida.append({
            "entrada": ent, "alvo": alvo, "sup": sup, "vis": vis_sai, "ocl": ocl,
            "base": base, "t0": np.int32(t0), "presente_ini": bool(presente[i_ref]),
            "escala_px": np.float32(escala), "origem_px": origem.astype(np.float32),
        })
    return saida


def empilhar(lista):
    return {
        "entrada": np.stack([a["entrada"] for a in lista]),
        "alvo": np.stack([a["alvo"] for a in lista]),
        "sup": np.stack([a["sup"] for a in lista]),
        "vis": np.stack([a["vis"] for a in lista]),
        "ocl": np.stack([a["ocl"] for a in lista]),
        "base": np.stack([a["base"] for a in lista]),
        "t0": np.stack([a["t0"] for a in lista]),
        "presente_ini": np.stack([a["presente_ini"] for a in lista]),
        "escala_px": np.stack([a["escala_px"] for a in lista]),
        "origem_px": np.stack([a["origem_px"] for a in lista]),
        "origem_sintetico": np.stack([a["origem_sintetico"] for a in lista]),
    }


def _tipo_cena(gt):
    """'lateral' ou 'direcao', lido dos parâmetros da cena (meta.json)."""
    meta = gt.parent / "meta.json"
    try:
        p = json.loads(meta.read_text(encoding="utf-8"))["parametros"]
        return "direcao" if abs(p["y1_frac"] - p["y0_frac"]) > 0.12 else "lateral"
    except (OSError, ValueError, KeyError):
        return "lateral"


def _dividir_cenas(cenas, fracao_treino=0.70, fracao_val=0.15):
    """Divide as cenas em treino/validação/teste, mantendo a proporção de cada tipo
    de cena em cada conjunto (divisão estratificada, sem vazamento)."""
    por_tipo = {}
    for i, gt in enumerate(cenas):
        por_tipo.setdefault(_tipo_cena(gt), []).append(i)
    grupos = {}
    for tipo, idxs in por_tipo.items():
        n = len(idxs)
        lim_tr = int(n * fracao_treino)
        lim_va = lim_tr + int(n * fracao_val)
        for pos, i in enumerate(idxs):
            grupos[i] = "treino" if pos < lim_tr else ("validacao" if pos < lim_va else "teste")
    return grupos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sinteticos", default="dados/sinteticos", help="pasta com as cenas sintéticas")
    p.add_argument("--real", default="dados/pose_real.csv")
    p.add_argument("--saida", default="dados/dataset.npz")
    p.add_argument("--T", type=int, default=25, help="quadros de entrada (1 s a 25 fps)")
    p.add_argument("--H", type=int, default=50, help="quadros de saída (2 s a 25 fps)")
    p.add_argument("--passo", type=int, default=5, help="passo entre janelas (quadros)")
    p.add_argument("--frac-real-treino", type=float, default=0.6)
    p.add_argument("--frac-real-val", type=float, default=0.2)
    args = p.parse_args()

    T, H = args.T, args.H
    conjuntos = {"treino": [], "validacao": [], "teste": []}

    # ---------- cenas sintéticas ----------
    pasta = Path(args.sinteticos)
    cenas = sorted(pasta.glob("cena_*/gt.csv")) if pasta.exists() else []
    n_cenas = len(cenas)
    print(f"{n_cenas} cenas sintéticas em {pasta}")

    # A divisão é ESTRATIFICADA por tipo de cena (lateral = a pessoa atravessa a
    # imagem; direção = a pessoa vem do fundo do corredor na direção da câmera).
    # Sem isso, validação e teste teriamIGNorar a dificuldade diferente e a
    # comparação entre modelos ficaria enviesada.
    grupos_cena = _dividir_cenas(cenas)
    for i, gt in enumerate(cenas):
        grupo = grupos_cena[i]
        trilha = ler_trilha_sintetica(gt)
        jan = janelas(trilha, T, H, args.passo)
        for a in jan:
            a["origem_sintetico"] = 1
        conjuntos[grupo].extend(jan)
        print(f"  {gt.parent.name}: {len(jan):5d} janelas -> {grupo}")

    # ---------- vídeo real ----------
    real = Path(args.real)
    if real.exists():
        trilha = ler_trilha_real(real)
        jan = janelas(trilha, T, H, args.passo)
        n = len(jan)
        a_tr = int(n * args.frac_real_treino)
        a_va = a_tr + int(n * args.frac_real_val)
        for k, a in enumerate(jan):
            a["origem_sintetico"] = 0
            conjuntos["treino" if k < a_tr else ("validacao" if k < a_va else "teste")].append(a)
        print(f"  vídeo real: {n} janelas ({a_tr} treino / {a_va - a_tr} val / {n - a_va} teste)")

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    dados = {}
    for grupo, lista in conjuntos.items():
        if not lista:
            continue
        emp = empilhar(lista)
        for k, v in emp.items():
            dados[f"{grupo}_{k}"] = v
        n_sint = sum(1 for a in lista if a["origem_sintetico"])
        print(f"{grupo}: {len(lista)} amostras ({n_sint} sintéticas / {len(lista) - n_sint} reais)")

    dados["config"] = np.array(json.dumps({
        "T": T, "H": H, "passo": args.passo, "n_articulacoes": N_ARTICULACOES,
        "n_cenas": n_cenas, "fps": 25.0,
        "horizontes_s": [0.5, 1.0, 2.0],
    }), dtype=object)
    np.savez_compressed(saida, **dados)
    print(f"Dataset salvo em {saida}")


if __name__ == "__main__":
    main()
