"""Avalia o resultado da Etapa 3 (processar_video.py) contra a verdade absoluta do vídeo
sintético.

O vídeo sintético (gerar_video_sintetico.py) tem ground truth: a posição real da pessoa
em cada quadro, inclusive nos segundos que ainda não existem no vídeo. Isso permite
medir, com número:

  1. PERCEPÇÃO/RASTREIO  - erro entre a caixa que o pipeline acompanhou e a caixa real
     (IoU e distância do centro), e o quanto o pipeline perdeu a pessoa.
  2. OCLUSÃO             - os intervalos em que a pessoa estava escondida de verdade
     (ground truth) contra os intervalos em que o pipeline marcou OCLUSAO.
  3. PREVISÃO            - erro da posição prevista (Random Forest e movimento constante)
     contra a posição real, para horizontes de 1 s, 2 s, 3 s...
  4. TRECHO EXTRA        - erro da previsão feita no fim do vídeo contra a trajetória
     real dos segundos seguintes (que o vídeo ainda não mostra).

Uso:
    python avaliar_sintetico.py --gt videos/video_sintetico_gt.csv \
        --log resultados/sintetico_previsao.csv \
        --previsoes resultados/sintetico_previsao_previsoes.csv \
        --info resultados/sintetico_previsao_info.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def ler_csv(caminho):
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = lambda c: max(0, c[2] - c[0]) * max(0, c[3] - c[1])
    uni = area(a) + area(b) - inter
    return inter / uni if uni > 0 else 0.0


def intervalos(quadros):
    """[0, 5, 6, 9] -> [(0, 5), (6, 9)]"""
    out = []
    for q in quadros:
        if out and q == out[-1][1] + 1:
            out[-1][1] = q
        else:
            out.append([q, q])
    return [tuple(i) for i in out]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", default="videos/video_sintetico_gt.csv")
    p.add_argument("--log", default="resultados/sintetico_previsao.csv")
    p.add_argument("--previsoes", default="resultados/sintetico_previsao_previsoes.csv")
    p.add_argument("--info", default="resultados/sintetico_previsao_info.json")
    p.add_argument("--meta", default="videos/video_sintetico_meta.json")
    p.add_argument("--saida", default="resultados/sintetico_avaliacao.txt")
    args = p.parse_args()

    gt = ler_csv(args.gt)
    log = ler_csv(args.log)
    info = json.loads(Path(args.info).read_text(encoding="utf-8")) if Path(args.info).exists() else {}
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8")) if Path(args.meta).exists() else {}

    fps = info.get("fps") or meta.get("fps", 25)
    n_video = info.get("quadros") or meta.get("quadros_video", 0)
    # o GT está na resolução nativa do vídeo sintético; o log está na resolução de saída
    esc_x = (info.get("largura", meta.get("largura")) or meta.get("largura", 1280)) / max(1, meta.get("largura", 1280))
    esc_y = (info.get("altura", meta.get("altura")) or meta.get("altura", 720)) / max(1, meta.get("altura", 720))

    # ---------- 1. percepção / rastreio ----------
    gt_por_quadro = {int(r["quadro"]): r for r in gt}
    log_por_quadro = {int(r["quadro"]): r for r in log}
    ious, dists, perdidos = [], [], 0
    for q in range(n_video):
        g, l = gt_por_quadro.get(q), log_por_quadro.get(q)
        if g is None or int(g["visivel"]) == 0:
            continue  # no chão truth a pessoa não está visível
        if l is None or l["x_centro"] == "":
            perdidos += 1
            continue
        gb = (float(g["x1"]), float(g["y1"]), float(g["x2"]), float(g["y2"]))
        gb = (gb[0] * esc_x, gb[1] * esc_y, gb[2] * esc_x, gb[3] * esc_y)
        cx, cy = float(l["x_centro"]), float(l["y_pes"])
        w, h = float(l.get("largura") or 0.0), float(l["altura"])
        lb = (cx - w / 2, cy - h, cx + w / 2, cy)
        ious.append(iou(gb, lb))
        dists.append(float(np.hypot(cx - float(g["cx_real"]) * esc_x, cy - float(g["y_pes"]) * esc_y)))

    # ---------- 2. oclusão ----------
    gt_oculto = [q for q in range(n_video)
                 if q in gt_por_quadro and gt_por_quadro[q]["cx_real"] and int(gt_por_quadro[q]["visivel"]) == 0]
    gt_presente = [q for q in range(n_video)
                   if q in gt_por_quadro and gt_por_quadro[q]["cx_real"]]
    gt_parcial = [q for q in gt_presente if int(gt_por_quadro[q]["visivel"]) == 1
                  and float(gt_por_quadro[q]["cobertura"]) < 0.9]
    pip_oclu = [q for q in range(n_video)
                if q in log_por_quadro and log_por_quadro[q]["estado"] == "OCLUSAO"]
    acertos = len(set(pip_oclu) & set(gt_oculto))
    recall = acertos / len(gt_oculto) if gt_oculto else 0.0
    precisao = acertos / len(pip_oclu) if pip_oclu else 0.0

    def fmt_interv(qs, fps_):
        return ", ".join(f"{a / fps_:.1f}-{b / fps_:.1f} s" for a, b in intervalos(sorted(qs))) or "-"

    # ---------- 3. previsão dentro do vídeo ----------
    prev = ler_csv(args.previsoes) if Path(args.previsoes).exists() else []
    erros = {}
    for r in prev:
        alvo = int(r["quadro_alvo"])
        g = gt_por_quadro.get(alvo)
        if g is None or not g["cx_real"]:
            continue
        h = float(r["horizonte_s"])
        if h < 0.5 or alvo > n_video:
            continue
        if abs(h - round(h)) > 0.06:  # só os horizontes exatos (1,0 s / 2,0 s / ...)
            continue
        real = np.array([float(g["cx_real"]) * esc_x, float(g["y_pes"]) * esc_y])
        rf = np.array([float(r["x_rf"]), float(r["y_rf"])])
        base = np.array([float(r["x_base"]), float(r["y_base"])])
        k = int(round(h))
        erros.setdefault(k, {"rf": [], "base": []})
        erros[k]["rf"].append(float(np.hypot(*(rf - real))))
        erros[k]["base"].append(float(np.hypot(*(base - real))))

    # ---------- 4. trecho extra (previsão depois do fim do vídeo) ----------
    extra = {"rf": [], "base": []}
    for r in prev:
        alvo = int(r["quadro_alvo"])
        g = gt_por_quadro.get(alvo)
        if g is None or not g["cx_real"] or alvo <= n_video:
            continue
        real = np.array([float(g["cx_real"]) * esc_x, float(g["y_pes"]) * esc_y])
        rf = np.array([float(r["x_rf"]), float(r["y_rf"])])
        base = np.array([float(r["x_base"]), float(r["y_base"])])
        extra["rf"].append(((alvo - n_video) / fps, float(np.hypot(*(rf - real)))))
        extra["base"].append(((alvo - n_video) / fps, float(np.hypot(*(base - real)))))

    L = []
    L.append("AVALIAÇÃO DA ETAPA 3 CONTRA O GROUND TRUTH DO VÍDEO SINTÉTICO")
    L.append("=" * 78)
    L.append(f"Vídeo: {info.get('video', '(desconhecido)')} | {n_video} quadros ({n_video / fps:.1f} s) "
             f"| {fps:.0f} fps | saída {info.get('largura')}x{info.get('altura')}")
    L.append(f"Detector usado: {info.get('detector', '?')} | oclusão total no GT: "
             f"{len(gt_oculto)} quadros | parcial: {len(gt_parcial)} quadros")
    L.append("")
    L.append("1) PERCEPÇÃO E RASTREIO (a caixa do pipeline contra a caixa real)")
    if ious:
        L.append(f"   IoU médio com a caixa real ............ {np.mean(ious):.3f} "
                 f"(mediana {np.median(ious):.3f})")
        L.append(f"   erro do centro da caixa ............... {np.mean(dists):.1f} px")
        L.append(f"   quadros em que a pessoa estava visível e o pipeline perdeu a "
                 f"pessoa: {perdidos}")
    else:
        L.append("   (sem caixas do pipeline para comparar)")
    L.append("")
    L.append("2) DETECÇÃO DE OCLUSÃO")
    L.append(f"   pessoa escondida de verdade (GT) ...... {fmt_interv(gt_oculto, fps)}")
    L.append(f"   pipeline marcou OCLUSAO ............... {fmt_interv(pip_oclu, fps)}")
    L.append(f"   recall (oculto detectado / oculto real) {recall:.2f}   "
             f"precisão (real / marcado) {precisao:.2f}")
    L.append("")
    L.append("3) PREVISÃO DENTRO DO VÍDEO (erro em px contra a posição real)")
    L.append("   horizonte | Random Forest | movimento constante | amostras")
    for k in sorted(erros):
        e = erros[k]
        L.append(f"   +{k} s      | {np.mean(e['rf']):12.1f}     | {np.mean(e['base']):18.1f}      | {len(e['rf'])}")
    if not erros:
        L.append("   (sem previsões numéricas)")
    L.append("")
    L.append("4) TRECHO EXTRA (previsão depois do fim do vídeo, contra a trajetória real)")
    if extra["rf"]:
        L.append(f"   erro médio no trecho extra ............ RF {np.mean([e for _, e in extra['rf']]):.1f} px | "
                 f"constante {np.mean([e for _, e in extra['base']]):.1f} px")
        L.append("   erro no instante exato de cada segundo depois do fim do vídeo (RF | constante):")
        pontos = {round(s, 2): e for s, e in extra["rf"]}
        pontos_b = {round(s, 2): e for s, e in extra["base"]}
        for k in range(0, int(max(pontos)) + 1):
            s = min(pontos, key=lambda v: abs(v - k))  # amostra mais próxima do segundo exato
            L.append(f"     +{k} s  ->  {pontos[s]:7.1f} px | {pontos_b.get(s, float('nan')):7.1f} px")
    else:
        L.append("   (sem linhas de previsão para o trecho extra)")
    L.append("")
    L.append("Nota: o erro é a distância entre a posição prevista e a posição REAL dos pés")
    L.append("      da pessoa (centro da caixa, base). O 'movimento constante' é o baseline.")

    texto = "\n".join(L)
    print(texto)
    Path(args.saida).write_text(texto, encoding="utf-8")
    print(f"\nRelatório salvo em {args.saida}")


if __name__ == "__main__":
    main()
