"""Gera um VÍDEO SINTÉTICO (cena desenhada por código, sem nenhuma filmagem real) para
testar a Etapa 3 do projeto.

Por que isso é útil:
  - o pipeline (processar_video.py) passa a ter um caso de teste que não depende de
    baixar pesos da internet nem de ter um vídeo gravado;
  - a cena tem VERDADE ABSOLUTA (ground truth): a posição real da pessoa em cada quadro,
    inclusive nos segundos que ainda não existem no vídeo. Assim dá para conferir se a
    previsão acertou (erro em pixels) e para medir a detecção de oclusão.

O cenário é propositalmente DIFERENTE do vídeo real gravado: aqui a pessoa caminha num
CORREDOR INTERNO de prédio (perspectiva, piso xadrez, portas, luminárias) e passa por
trás de uma PILASTRA, que a esconde por completo (oclusão total) e depois deixa só
metade do corpo à mostra (oclusão parcial).

Arquivos gerados:
  videos/video_sintetico.mp4        - o vídeo sintético (22 s)
  videos/video_sintetico_gt.csv     - ground truth por quadro (caixa, cobertura, visibilidade)
  videos/video_sintetico_meta.json  - metadados da cena (fps, resolução, trajetória)
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------------
# Geometria da cena (tudo em pixels da imagem final)
# ----------------------------------------------------------------------------------
HORIZONTE_FRAC = 0.42       # fração da altura onde fica o ponto de fuga do corredor
RAZAO_ALTURA = 1.03        # altura da pessoa (px) = RAZAO_ALTURA * (y_dos_pes - horizonte)
LARGURA_REL = 0.34         # largura do corpo (px) = LARGURA_REL * altura

# A pessoa só entra em cena depois deste tempo (o detector de fundo usa os primeiros
# segundos como referência limpa).
T_APARECE = 2.0

# Caminho da pessoa dentro do corredor (t em segundos): vem do fundo e vem em nossa
# direção, desviando levemente para a direita.
X0, VX, CURVA, PER_CURVA = 520.0, 24.0, 22.0, 8.0
Y0, Y1 = 395.0, 610.0       # y dos pés: de longe (395) para perto (610)

# Pilastra do corredor (desenhada DEPOIS da pessoa -> a esconde)
PILASTRA_X = 760.0
PILASTRA_W = 85.0
PILASTRA_TOPO, PILASTRA_BASE = 110.0, 560.0


# ----------------------------------------------------------------------------------
# Trajetória (função analítica: vale também para o "futuro" além do vídeo)
# ----------------------------------------------------------------------------------
def posicao_pessoa(t, duracao_total):
    """t em segundos -> (x_centro, y_dos_pes)."""
    dt = t - T_APARECE
    if dt <= 0:
        return None
    p = dt / max(1e-6, duracao_total - T_APARECE)
    x = X0 + VX * dt + CURVA * math.sin(2 * math.pi * dt / PER_CURVA)
    y = Y0 + (Y1 - Y0) * (min(p, 1.6) ** 1.15)
    return x, y


def horizonte_px(H):
    return int(H * HORIZONTE_FRAC)


def tamanho_pessoa(y_pes, horizonte):
    return max(8.0, RAZAO_ALTURA * (y_pes - horizonte))


# ----------------------------------------------------------------------------------
# Desenho do corredor
# ----------------------------------------------------------------------------------
def _poligono(img, pts, cor):
    cv2.fillPoly(img, [np.array(pts, np.int32)], cor, cv2.LINE_AA)


def fundo_corredor(W, H, vp):
    """Corredor em perspectiva: piso xadrez, paredes com portas, teto com luminárias."""
    vx, vy = vp
    fx1, fx2 = int(vx - 60), int(vx + 60)   # bordas da parede no fundo do corredor
    fy1, fy2 = int(vy - 45), int(vy + 45)   # teto e piso no fundo do corredor

    img = np.zeros((H, W, 3), np.uint8)
    _poligono(img, [(0, 0), (W, 0), (fx2, fy1), (fx1, fy1)], (198, 200, 203))          # teto
    _poligono(img, [(0, 0), (fx1, fy1), (fx1, fy2), (0, H)], (170, 196, 214))          # parede esq.
    _poligono(img, [(W, 0), (W, H), (fx2, fy2), (fx2, fy1)], (170, 196, 214))          # parede dir.
    _poligono(img, [(0, H), (W, H), (fx2, fy2), (fx1, fy2)], (150, 151, 154))          # piso

    # profundidade: d = 0 no fundo do corredor, d = 1 na câmera
    profs = [1 - 0.62 ** k for k in range(8)]

    def y_piso(d):
        return fy2 + (H - fy2) * d

    def y_teto(d):
        return fy1 * (1 - d)

    def x_esq(d):
        return fx1 * (1 - d)

    def x_dir(d):
        return fx2 + (W - fx2) * d

    # --- piso xadrez ---
    for k in range(len(profs) - 1):
        d0, d1 = profs[k], profs[k + 1]
        for j in range(-3, 12):
            x0 = j * (W / 8.0)
            p0 = (vx + (x0 - vx) * d0, y_piso(d0))
            p1 = (vx + (x0 - vx) * d1, y_piso(d1))
            cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                     (128, 129, 132), 1, cv2.LINE_AA)
        y = y_piso(d1)
        cv2.line(img, (int(x_esq(d1)), int(y)), (int(x_dir(d1)), int(y)), (120, 121, 124), 2, cv2.LINE_AA)

    # --- rodapé e portas nas paredes ---
    for k in range(len(profs) - 1):
        d0, d1 = profs[k], profs[k + 1]
        # rodapé (faixa escura no pé da parede)
        _poligono(img, [(x_esq(d1), y_piso(d1) - 46 * (d1 + 0.15)), (x_esq(d0), y_piso(d0) - 46 * d0),
                        (x_esq(d0), y_piso(d0)), (x_esq(d1), y_piso(d1))], (120, 145, 165))
        _poligono(img, [(x_dir(d1), y_piso(d1) - 46 * (d1 + 0.15)), (x_dir(d0), y_piso(d0) - 46 * d0),
                        (x_dir(d0), y_piso(d0)), (x_dir(d1), y_piso(d1))], (120, 145, 165))
        if k % 2 == 0:  # portas
            _poligono(img, [(x_esq(d1), y_teto(d1) + 40 * d1 + 30), (x_esq(d0), y_teto(d0) + 40 * d0 + 12),
                            (x_esq(d0), y_piso(d0) - 46 * d0 - 12), (x_esq(d1), y_piso(d1) - 46 * d1 - 30)],
                      (95, 105, 120))
            _poligono(img, [(x_dir(d1), y_teto(d1) + 40 * d1 + 30), (x_dir(d0), y_teto(d0) + 40 * d0 + 12),
                            (x_dir(d0), y_piso(d0) - 46 * d0 - 12), (x_dir(d1), y_piso(d1) - 46 * d1 - 30)],
                      (95, 105, 120))
            # maçaneta
            mx = x_esq(d0) + 6 * d0 if d0 > 0.05 else x_esq(d0) + 3
            my = (y_teto(d0) + y_piso(d0)) / 2
            cv2.circle(img, (int(mx), int(my)), max(1, int(4 * d0 + 1)), (210, 210, 215), -1, cv2.LINE_AA)

    # --- luminárias no teto (do fundo para a frente: cada uma aparece separada) ---
    for k in range(len(profs) - 2, -1, -1):
        d0, d1 = profs[k], profs[k + 1]
        _poligono(img, [(vx - 130 * d0, y_teto(d0) + 12 * d0), (vx + 130 * d0, y_teto(d0) + 12 * d0),
                        (vx + 130 * d1, y_teto(d1) + 12 * d1), (vx - 130 * d1, y_teto(d1) + 12 * d1)],
                  (238, 242, 246))

    # --- porta clara no fundo do corredor (a pessoa aparece por ali) ---
    _poligono(img, [(fx1 + 6, fy1 + 4), (fx2 - 6, fy1 + 4), (fx2 - 6, fy2 - 6), (fx1 + 6, fy2 - 6)],
              (225, 235, 245))
    return img


def desenhar_pilastra(img):
    """Pilastra do corredor (primeiro plano): é desenhada por cima da pessoa."""
    x1 = int(PILASTRA_X - PILASTRA_W / 2)
    x2 = int(PILASTRA_X + PILASTRA_W / 2)
    # sombra no chão, para a pilastra não parecer flutuar
    sombra = img.copy()
    cv2.ellipse(sombra, (int(PILASTRA_X), int(PILASTRA_BASE)), (int(PILASTRA_W * 0.8), 16), 0, 0, 360,
                (120, 121, 125), -1, cv2.LINE_AA)
    cv2.addWeighted(sombra, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (x1, int(PILASTRA_TOPO)), (x2, int(PILASTRA_BASE)), (175, 178, 182), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, int(PILASTRA_TOPO)), (x1 + 12, int(PILASTRA_BASE)), (150, 153, 158), -1)
    cv2.rectangle(img, (x2 - 12, int(PILASTRA_TOPO)), (x2, int(PILASTRA_BASE)), (198, 200, 203), -1)
    cv2.rectangle(img, (x1 - 8, int(PILASTRA_BASE - 26)), (x2 + 8, int(PILASTRA_BASE)), (160, 163, 168), -1)
    return x1, x2


def desenhar_pessoa(camada, cx, y_pes, h, fase):
    """Desenha a pessoa na camada. h = altura em pixels, fase = ciclo da caminhada.

    As cores são escolhidas para contrastar no CINZA (luminância) com o corredor: a
    camisa é escura e a calça clara. Assim o detector por subtração de fundo (que
    trabalha em escala de cinza) consegue separar a pessoa do cenário.
    """
    pele, cabelo = (110, 160, 210), (25, 25, 30)
    camisa, calca, sapato = (45, 40, 35), (75, 65, 55), (25, 25, 30)

    balanco = -0.018 * h * abs(math.cos(fase))
    quadril = y_pes - 0.50 * h + balanco * 0.4
    ombro = y_pes - 0.80 * h + balanco

    # torso
    _poligono(camada, [(cx - 0.15 * h, ombro), (cx + 0.15 * h, ombro),
                        (cx + 0.12 * h, quadril + 0.02 * h), (cx - 0.12 * h, quadril + 0.02 * h)], camisa)
    # cabeça
    r = 0.075 * h
    cy_cab = y_pes - 0.90 * h + balanco
    cv2.ellipse(camada, (int(cx), int(cy_cab)), (int(r), int(r * 1.15)), 0, 0, 360, pele, -1, cv2.LINE_AA)
    cv2.ellipse(camada, (int(cx), int(cy_cab - r * 0.45)), (int(r), int(r * 0.75)), 0, 180, 360, cabelo, -1, cv2.LINE_AA)

    # braços (balanço contrário às pernas)
    for s in (-1, 1):
        a = -0.30 * math.sin(fase) * s
        ox, oy = cx + s * 0.135 * h, ombro
        comp = 0.32 * h
        ex, ey = ox + comp * math.sin(a), oy + comp * math.cos(a)
        cv2.line(camada, (int(ox), int(oy)), (int(ex), int(ey)), camisa, max(2, int(0.055 * h)), cv2.LINE_AA)
        cv2.circle(camada, (int(ex), int(ey)), max(2, int(0.03 * h)), pele, -1, cv2.LINE_AA)

    # pernas
    for s in (-1, 1):
        a = 0.32 * math.sin(fase) * s
        ox, oy = cx + s * 0.06 * h, quadril
        comp = 0.50 * h
        px_, py_ = ox + comp * math.sin(a), oy + comp * math.cos(a)
        joelho_x = ox + 0.5 * comp * math.sin(a) - 0.04 * h * s
        joelho_y = oy + 0.5 * comp * math.cos(a)
        cv2.line(camada, (int(ox), int(oy)), (int(joelho_x), int(joelho_y)), calca,
                 max(2, int(0.075 * h)), cv2.LINE_AA)
        cv2.line(camada, (int(joelho_x), int(joelho_y)), (int(px_), int(py_)), calca,
                 max(2, int(0.065 * h)), cv2.LINE_AA)
        cv2.ellipse(camada, (int(px_), int(py_)), (max(2, int(0.05 * h)), max(2, int(0.025 * h))),
                    0, 0, 360, sapato, -1, cv2.LINE_AA)


def sombra_suave(img, cx, y_pes, h):
    """Sombra bem discreta no chão (fora da máscara da pessoa, para não confundir o GT)."""
    camada = img.copy()
    cv2.ellipse(camada, (int(cx), int(y_pes)), (int(0.20 * h), int(0.040 * h)), 0, 0, 360, (105, 106, 110), -1)
    cv2.addWeighted(camada, 0.5, img, 0.5, 0, img)


# ----------------------------------------------------------------------------------
# Geração
# ----------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saida", default="videos/video_sintetico.mp4")
    p.add_argument("--gt", default="videos/video_sintetico_gt.csv")
    p.add_argument("--meta", default="videos/video_sintetico_meta.json")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--largura", type=int, default=1280)
    p.add_argument("--altura", type=int, default=720)
    p.add_argument("--duracao", type=float, default=22.0, help="duração do vídeo (s)")
    p.add_argument("--futuro", type=float, default=6.0, help="segundos de trajetória além do vídeo (GT)")
    p.add_argument("--quadros", default="", help="pasta para salvar alguns quadros de exemplo")
    p.add_argument("--semente", type=int, default=42)
    args = p.parse_args()

    W, H, fps = args.largura, args.altura, args.fps
    horizonte = horizonte_px(H)
    vp = (W // 2, horizonte)
    n_video = int(round(args.duracao * fps))
    n_total = n_video + int(round(args.futuro * fps))
    dur_total = n_total / fps

    fundo = fundo_corredor(W, H, vp)

    video = Path(args.saida)
    video.parent.mkdir(parents=True, exist_ok=True)
    escritor = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not escritor.isOpened():
        raise SystemExit("Não consegui criar o vídeo (codec mp4v indisponível?).")

    pasta_quadros = Path(args.quadros) if args.quadros else None
    if pasta_quadros:
        pasta_quadros.mkdir(parents=True, exist_ok=True)

    gt = open(args.gt, "w", newline="", encoding="utf-8")
    w = csv.writer(gt)
    w.writerow(["quadro", "tempo_s", "dentro_do_video", "cx_real", "y_pes", "altura_real", "largura_real",
                 "x1", "y1", "x2", "y2", "cobertura", "visivel"])

    dist = 0.0
    pos_anterior = None
    n_ocluidos = n_parciais = 0
    for i in range(n_total):
        t = i / fps
        pos = posicao_pessoa(t, dur_total)
        frame = fundo.copy()

        # a pilastra é cenário fixo: é desenhada em TODOS os quadros (e por cima da
        # pessoa, porque está mais perto da câmera)
        pilastra = np.zeros((H, W, 3), np.uint8)
        desenhar_pilastra(pilastra)
        m_pilastra = pilastra.max(axis=2) > 0

        if pos is None:
            quad = None
            cobertura, visivel, cx, y_pes, h = 0.0, 0, 0, 0, 0
        else:
            cx, y_pes = pos
            h = tamanho_pessoa(y_pes, horizonte)
            if pos_anterior is not None:
                dist += abs(cx - pos_anterior[0])
            pos_anterior = pos
            fase = 2 * math.pi * dist / max(1e-6, 0.55 * h)

            sombra_suave(frame, cx, y_pes, h)
            camada = np.zeros((H, W, 3), np.uint8)
            desenhar_pessoa(camada, cx, y_pes, h, fase)
            m_pessoa = camada.max(axis=2) > 0
            frame[m_pessoa] = camada[m_pessoa]

            # o que a câmera realmente vê da pessoa (fora a pilastra)
            visivel_mask = m_pessoa & ~m_pilastra
            area_total = int(m_pessoa.sum())
            area_vis = int(visivel_mask.sum())
            cobertura = area_vis / area_total if area_total else 0.0
            ys, xs = np.nonzero(visivel_mask)
            if area_vis > 0:
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            else:
                x1 = y1 = x2 = y2 = -1
            quad = (x1, y1, x2, y2)
            visivel = int(cobertura >= 0.35)
            if cobertura < 0.02:
                visivel = 0
                n_ocluidos += 1
            elif cobertura < 0.9:
                n_parciais += 1

        frame[m_pilastra] = pilastra[m_pilastra]

        if i < n_video:
            escritor.write(frame)

        if quad is None:
            w.writerow([i, f"{t:.2f}", int(i < n_video), "", "", "", "", "", "", "", "", "0.000", 0])
        else:
            w.writerow([i, f"{t:.2f}", int(i < n_video), f"{cx:.1f}", f"{y_pes:.1f}", f"{h:.1f}",
                        f"{LARGURA_REL * h:.1f}", x1, y1, x2, y2, f"{cobertura:.3f}", visivel])

        if pasta_quadros and i in (int(n_video * 0.30), int(n_video * 0.50), int(n_video * 0.60),
                                   int(n_video * 0.75), n_video - 1):
            cv2.imwrite(str(pasta_quadros / f"quadro_{i:04d}.png"), frame)

    escritor.release()
    gt.close()

    meta = {
        "cenario": "corredor interno de prédio",
        "fps": fps, "largura": W, "altura": H, "quadros_video": n_video, "quadros_total": n_total,
        "duracao_video_s": args.duracao, "futuro_s": args.futuro, "horizonte": horizonte,
        "t_aparece": T_APARECE, "pilastra_x": PILASTRA_X,
        "quadros_totalmente_ocultos": n_ocluidos, "quadros_parciais": n_parciais,
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Vídeo sintético: {video} ({args.duracao:.1f} s, {W}x{H}, {fps} fps)")
    print(f"Ground truth:   {args.gt} ({n_total} quadros, sendo {n_video} no vídeo)")
    print(f"Metadados:      {args.meta}")
    print(f"Oclusão total em {n_ocluidos} quadros | oclusão parcial em {n_parciais} quadros")


if __name__ == "__main__":
    main()
