"""Gera um CONJUNTO de cenas sintéticas variadas para treinar e testar os modelos
temporais (LSTM, GRU, EKF, Transformer, T-GNN).

Cada cena é uma variação do corredor (trajetória, velocidade, posição e largura da
pilastra = duração da oclusão, resolução e duração) e traz a VERDADE ABSOLUTA da pose:
as 13 articulações em CADA quadro, inclusive nos quadros em que a pessoa está
totalmente escondida atrás da pilastra (a "pose latente"). Isso permite:

  - treinar o modelo para reconstruir a pose ESCONDIDA (o que o vídeo real não permite,
    porque lá não existe GT sob a oclusão);
  - medir o erro de previsão durante a oclusão, quadro a quadro.

Arquivos gerados (em dados/sinteticos/cena_XX/):
  gt.csv    - GT por quadro: pose, visibilidade por articulação, caixa e cobertura
  video.mp4 - o vídeo da cena (mesma cena, para inspeção/validação visual)
  meta.json - parâmetros da cena

Uso:
  python gerar_dados_sinteticos.py --cenas 24 --saida dados/sinteticos
"""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cena_corredor import Cena, gerar_cena

# Semente fixa: o conjunto de cenas é reprodutível.
SEMENTE_MESTRE = 20240


def construir_cena(i, semente=SEMENTE_MESTRE):
    """Cena i: parâmetros sorteados de forma determinística (variedade controlada).

    Dois tipos de cena, para o modelo ver situations diferentes:

      * LATERAL (maioria): a pessoa atravessa a imagem de um lado para o outro, numa
        profundidade constante. Ela passa ATRÁS da pilastra por um tempo curto e bem
        controlado (largura da pilastra / velocidade) -> oclusão total curta, com
        oclusão parcial nas bordas. É o cenário clássico de "atravessar atrás de um
        poste".
      * EM DIREÇÃO (minoria): a pessoa vem do fundo do corredor na direção da câmera,
        como na cena principal. A oclusão é longa (ela fica atrás da pilastra até
        passar por ela).
    """
    r = _rng(i, semente)
    tipo = "lateral" if r.random() < 0.75 else "direcao"

    largura = int(r.choice([640, 640, 800, 960]))
    altura = int(round(largura * 9 / 16))
    fps = 25
    duracao = float(r.choice([16.0, 18.0, 20.0, 22.0]))

    # ---- pilastra: define a duração da oclusão ----
    pil_x = float(r.uniform(0.40, 0.72))
    pil_w = float(r.choice([0.05, 0.07, 0.09, 0.12, 0.15]))
    topo = float(r.uniform(0.08, 0.20))
    base = float(r.uniform(0.70, 0.88))

    if tipo == "lateral":
        sentido = r.choice([-1, 1])
        vx = sentido * float(r.uniform(0.030, 0.060))
        # a pessoa começa de um lado e sai pelo outro
        sobra = 0.10
        x0 = (1.0 - sobra - vx * duracao) if sentido > 0 else (sobra - vx * duracao)
        x0 = min(max(x0, 0.02), 0.9)
        # profundidade quase constante: só um leve deslocamento (a câmera é fixa)
        y0 = float(r.uniform(0.52, 0.62))
        y1 = y0 + float(r.choice([-0.06, -0.03, 0.0, 0.03, 0.06]))
        y1 = min(max(y1, 0.40), 0.88)
        curva = float(r.choice([0.0, 0.0, 0.008, 0.015]))
    else:
        sentido = r.choice([-1, 1])
        vx = sentido * float(r.uniform(0.015, 0.035))
        x0 = float(r.uniform(0.30, 0.70))
        y0 = float(r.uniform(0.48, 0.56))
        y1 = float(r.uniform(0.80, 0.95))
        curva = float(r.choice([0.0, 0.012, 0.025, 0.04]))

    return Cena(
        duracao=duracao, futuro=2.0, fps=fps, largura=largura, altura=altura,
        t_aparece=float(r.uniform(0.3, 1.2)),
        x0_frac=x0, vx_frac=vx, curva=curva, per_curva=float(r.choice([6.0, 8.0, 11.0])),
        y0_frac=y0, y1_frac=y1,
        pilastra_x_frac=pil_x, pilastra_w_frac=pil_w,
        pilastra_topo_frac=topo, pilastra_base_frac=base,
    )


def _rng(i, semente):
    import random
    return random.Random(semente * 100003 + i)


def gerar_uma(args):
    i, pasta, escrever_video = args
    cena = construir_cena(i)
    d = Path(pasta) / f"cena_{i:03d}"
    d.mkdir(parents=True, exist_ok=True)
    gt = d / "gt.csv"
    meta = gerar_cena(cena, d / "video.mp4", gt, d / "meta.json", escrever_video=escrever_video)
    return i, meta["quadros_totalmente_ocultos"], meta["quadros_parciais"], meta["quadros_video"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cenas", type=int, default=24, help="quantidade de cenas variadas")
    p.add_argument("--saida", default="dados/sinteticos", help="pasta das cenas")
    p.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--sem-video", action="store_true", help="não grava os vídeos (só o GT)")
    p.add_argument("--semente", type=int, default=SEMENTE_MESTRE)
    args = p.parse_args()

    Path(args.saida).mkdir(parents=True, exist_ok=True)
    tarefas = [(i, args.saida, not args.sem_video) for i in range(args.cenas)]

    print(f"Gerando {args.cenas} cenas sintéticas em {args.saida} "
          f"({args.processos} processos)...")
    res = []
    if args.processos > 1:
        with ProcessPoolExecutor(max_workers=args.processos) as ex:
            for r in ex.map(gerar_uma, tarefas):
                res.append(r)
                print(f"  cena {r[0]:03d}: {r[3]} quadros, {r[1]} totalmente ocultos, {r[2]} parciais",
                      flush=True)
    else:
        for t in tarefas:
            r = gerar_uma(t)
            res.append(r)
            print(f"  cena {r[0]:03d}: {r[3]} quadros, {r[1]} totalmente ocultos, {r[2]} parciais",
                  flush=True)

    indice = [{
        "cena": i,
        "pasta": f"cena_{i:03d}",
        "quadros_totalmente_ocultos": oc,
        "quadros_parciais": pa,
        "quadros_video": qv,
    } for i, oc, pa, qv in sorted(res)]
    Path(args.saida, "indice.json").write_text(json.dumps(indice, indent=2), encoding="utf-8")

    tot_oc = sum(x["quadros_totalmente_ocultos"] for x in indice)
    tot_pa = sum(x["quadros_parciais"] for x in indice)
    print(f"Pronto: {len(indice)} cenas, {tot_oc} quadros totalmente ocultos, {tot_pa} parciais")
    print(f"Índice: {Path(args.saida, 'indice.json')}")


if __name__ == "__main__":
    main()
