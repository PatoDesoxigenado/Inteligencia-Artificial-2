"""Desenha sobre um vídeo as previsões de pose dos modelos temporais.

Para cada quadro com a pessoa detectada, o script:
  - desenha a pose OBSERVADA (MediaPipe);
  - desenha a pose PREVISTA para os próximos horizontes (0,5 s / 1,0 s / 2,0 s),
    colorida por modelo;
  - quando a pessoa está escondida (sem detecção), mostra a caixa do último alvo e a
    trajetória prevista - é a situação de oclusão que o projeto quer resolver.

Também salva um CSV com as previsões de cada modelo em cada quadro.

Uso:
  python visualizar_previsoes.py --video videos/meu_video.mp4 --modelos resultados/modelos \
      --saida resultados/video_previsoes_pose.mp4
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from cena_corredor import INDICES_MEDIAPIPE, N_ARTICULACOES, NOMES_ARTICULACOES
from modelos_temporais import (EKF, GRUPose, LSTMPose, RFPose, TGNNPose, TransformerPose)

CORES = {
    "ekf": (255, 200, 0),
    "lstm": (0, 200, 0),
    "gru": (0, 160, 255),
    "transformer": (200, 0, 200),
    "tgnn": (0, 0, 255),
}
ESQUELETO = [(1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8),
             (7, 9), (9, 11), (8, 10), (10, 12), (7, 8), (0, 1), (0, 2)]


def desenhar_pose(img, pontos, cor, espessura=2, circulos=True):
    for a, b in ESQUELETO:
        if a < len(pontos) and b < len(pontos):
            p1 = tuple(int(v) for v in pontos[a])
            p2 = tuple(int(v) for v in pontos[b])
            if all(0 <= v < max(img.shape[:2]) for v in (p1 + p2)):
                cv2.line(img, p1, p2, cor, espessura, cv2.LINE_AA)
    if circulos:
        for p in pontos:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), 3, cor, -1, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="videos/meu_video.mp4")
    p.add_argument("--pose", default="dados/pose_real.csv", help="CSV com a pose (extrair_pose.py)")
    p.add_argument("--modelos", default="resultados/modelos")
    p.add_argument("--saida", default="resultados/video_previsoes_pose.mp4")
    p.add_argument("--csv", default="resultados/previsoes_pose.csv")
    p.add_argument("--T", type=int, default=25)
    p.add_argument("--H", type=int, default=50)
    p.add_argument("--horizontes", default="0.5,1.0,2.0",
                   help="horizontes (s) desenhados no vídeo")
    p.add_argument("--espessura", type=int, default=1)
    args = p.parse_args()

    import torch
    from avaliar_modelos_seq import prever_ekf

    # ---- modelos ----
    modelos = {}
    pasta = Path(args.modelos)
    fabricas = {"lstm": LSTMPose, "gru": GRUPose, "transformer": TransformerPose, "tgnn": TGNNPose}
    for nome, cls in fabricas.items():
        arq = pasta / f"{nome}.pt"
        if arq.exists():
            d = torch.load(arq, map_location="cpu", weights_only=False)
            m = cls(d.get("T", args.T), d.get("H", args.H))
            m.load_state_dict(d["estado"])
            m.eval()
            modelos[nome] = m
    if (pasta / "rf.joblib").exists():
        import joblib
        modelos["rf"] = joblib.load(pasta / "rf.joblib")

    # ---- pose observada ----
    with open(args.pose, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    fps = 25.0
    n = len(linhas)
    obs = np.zeros((n, N_ARTICULACOES, 2), np.float32)
    vis = np.zeros((n, N_ARTICULACOES), np.float32)
    for i, r in enumerate(linhas):
        for j, nome in enumerate(NOMES_ARTICULACOES):
            if int(r[f"{nome}_vis"]):
                obs[i, j] = (float(r[f"{nome}_x"]), float(r[f"{nome}_y"]))
                vis[i, j] = 1

    # ---- janelas (igual ao dataset) ----
    import montar_dataset as md
    trilha = md.ler_trilha_real(args.pose)
    janelas = md.janelas(trilha, args.T, args.H, 1)
    t0_de = {j["t0"]: k for k, j in enumerate(janelas)}
    if janelas:
        ent = np.stack([j["entrada"] for j in janelas])
        base = np.stack([j["base"] for j in janelas])
        prev = {}
        for nome, m in modelos.items():
            if isinstance(m, RFPose):
                prev[nome] = m.prever_lote(ent, base)
            else:
                with torch.no_grad():
                    prev[nome] = base + m.forward(torch.tensor(ent, dtype=torch.float32)).numpy()
        prev["ekf"] = prever_ekf(ent, base, args.H, args.T)
    else:
        prev = {}

    # ---- desenho ----
    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    esc = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hh))
    registos = []

    horizonte_px = {float(h): int(float(h) * fps) for h in args.horizontes.split(",") if h}

    i = 0
    while True:
        ok, quadro = cap.read()
        if not ok:
            break
        if i < n and vis[i].sum() >= 4:
            desenhar_pose(quadro, obs[i], (255, 255, 255), 2)

        k = t0_de.get(i - args.T + 1)  # janela que termina neste quadro
        if k is not None and prev:
            # converte do referencial normalizado da janela para pixels
            ref = linhas[i - args.T + 1]
            org = np.array([float(ref[f"{NOMES_ARTICULACOES[7]}_x"]),
                            float(ref[f"{NOMES_ARTICULACOES[7]}_y"])])
            q = np.array([float(ref[f"{NOMES_ARTICULACOES[8]}_x"]),
                          float(ref[f"{NOMES_ARTICULACOES[8]}_y"])])
            cab = np.array([float(ref[f"{NOMES_ARTICULACOES[0]}_x"]),
                            float(ref[f"{NOMES_ARTICULACOES[0]}_y"])])
            c = (org + q) / 2
            esc_p = float(np.linalg.norm(cab - c)) or 1.0
            for h_s, passo in horizonte_px.items():
                for nome, preds in prev.items():
                    if passo - 1 >= preds.shape[1]:
                        continue
                    px = preds[k, passo - 1] * esc_p + c
                    desenhar_pose(quadro, px, CORES.get(nome, (128, 128, 128)),
                                  args.espessura, circulos=False)
                    registos.append([i, round(h_s, 2), nome] + [f"{v:.1f}" for v in px.reshape(-1)])

        # HUD
        presente = i < n and vis[i].sum() >= 4
        cv2.putText(quadro, "pose OBSERVADA (branco)" if presente else "PESSOA ESCONDIDA (oclusao)",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(quadro, "pose OBSERVADA (branco)" if presente else "PESSOA ESCONDIDA (oclusao)",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        hs_txt = " / ".join(f"{h:g}s" for h in sorted(horizonte_px))
        for k2, nome in enumerate(prev):
            cor = CORES.get(nome, (200, 200, 200))
            cv2.putText(quadro, f"{nome}: {hs_txt}", (16, 68 + 26 * k2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2, cv2.LINE_AA)
        esc.write(quadro)
        i += 1

    cap.release()
    esc.release()

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quadro", "horizonte_s", "modelo"] +
                   [f"{n}_x" for n in NOMES_ARTICULACOES] + [f"{n}_y" for n in NOMES_ARTICULACOES])
        w.writerows(registos)
    print(f"Vídeo anotado: {saida} | previsões: {args.csv}")


if __name__ == "__main__":
    main()
