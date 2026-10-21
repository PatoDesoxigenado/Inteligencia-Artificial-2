"""Extrai a POSE (13 articulações) de um vídeo gravado, usando o MediaPipe Pose.

O MediaPipe devolve 33 pontos com um índice de VISIBILIDADE por ponto. Aqui eles são
convertidos para o esqueleto canônico de 13 articulações (a mesma ordem do vídeo
sintético), e cada articulação fica com:

    x, y   -> posição em pixels do vídeo original
    vis   -> 1 se a articulação foi vista com confiança (visibilidade do MediaPipe)

Esse arquivo é a "entrada real" do dataset: no vídeo real não existe verdade absoluta,
o que se tem é a pose ESTIMADA pelo detector. Quadros em que a pessoa não aparece
ficam com vis = 0 em todas as articulações (a oclusão total).

Uso:
  python extrair_pose.py --video videos/meu_video.mp4 --saida dados/pose_real.csv
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from cena_corredor import INDICES_MEDIAPIPE, N_ARTICULACOES, NOMES_ARTICULACOES

# acima deste valor de visibilidade a articulação é considerada vista
LIMITE_VISIBILIDADE = 0.5


def extrair(video, modelo=1, conf=0.5, tracking=0.5, min_joints=4):
    """Roda o MediaPipe no vídeo e devolve (N, 13, 2) posições, (N, 13) visibilidade e
    (N,) flags de quadro válido."""
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir o vídeo: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=modelo,
                                  min_detection_confidence=conf, min_tracking_confidence=tracking)

    posicoes, visiveis, validos = [], [], []
    i = 0
    t0 = time.time()
    while True:
        ok, quadro = cap.read()
        if not ok:
            break
        r = pose.process(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))
        lm = r.pose_landmarks.landmark if r.pose_landmarks is not None else None

        p = np.zeros((N_ARTICULACOES, 2), np.float32)
        v = np.zeros(N_ARTICULACOES, np.float32)
        if lm is not None and len(lm) > max(INDICES_MEDIAPIPE):
            for j, ind in enumerate(INDICES_MEDIAPIPE):
                conf_j = float(lm[ind].visibility)
                v[j] = 1.0 if conf_j >= LIMITE_VISIBILIDADE else 0.0
                # posição só é usada quando a articulação foi vista
                p[j] = (lm[ind].x * W, lm[ind].y * H) if v[j] else (0.0, 0.0)
        posicoes.append(p)
        visiveis.append(v)
        validos.append(bool(v.sum() >= min_joints))
        i += 1

    cap.release()
    pose.close()
    print(f"MediaPipe: {i} quadros em {time.time() - t0:.1f} s "
          f"({sum(validos)} com a pessoa detectada)")
    return np.array(posicoes), np.array(visiveis), np.array(validos, bool), fps, W, H


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="videos/meu_video.mp4")
    p.add_argument("--saida", default="dados/pose_real.csv")
    p.add_argument("--modelo", type=int, default=1, choices=[0, 1, 2], help="complexidade do MediaPipe")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--tracking", type=float, default=0.5)
    args = p.parse_args()

    pos, vis, val, fps, W, H = extrair(args.video, args.modelo, args.conf, args.tracking)

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cabecalho = ["quadro", "tempo_s", "detectado", "na_cena"]
        for nome in NOMES_ARTICULACOES:
            cabecalho += [f"{nome}_x", f"{nome}_y", f"{nome}_vis"]
        w.writerow(cabecalho)
        for i in range(len(pos)):
            # no video real so existe o que a camera viu: na_cena = detectado
            linha = [i, f"{i / fps:.2f}", int(val[i]), int(val[i])]
            for j in range(N_ARTICULACOES):
                linha += [f"{pos[i, j, 0]:.2f}", f"{pos[i, j, 1]:.2f}", int(vis[i, j])]
            w.writerow(linha)

    print(f"Pose real salva em {saida} ({len(pos)} quadros, {W}x{H} @ {fps:.0f} fps)")


if __name__ == "__main__":
    main()
