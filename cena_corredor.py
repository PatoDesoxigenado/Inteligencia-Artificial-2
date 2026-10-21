"""Cena do corredor (cenário sintético) com VERDADE ABSOLUTA da POSE.

Este módulo generaliza o gerador original (gerar_video_sintetico.py): a cena é
parametrizada (trajetória, velocidade, posição/largura da pilastra, escala) e, além
da caixa da pessoa, exporta as 13 articulações do corpo e a visibilidade de cada uma
(a pilastra cobre parte do corpo -> a articulação "some" da imagem).

Esqueleto canônico (13 articulações) - a MESMA ordem usada pelo MediaPipe no vídeo real:

    0 cabeca        1 ombro_esq    2 ombro_dir    3 cotovelo_esq  4 cotovelo_dir
    5 pulso_esq     6 pulso_dir     7 quadril_esq   8 quadril_dir   9 joelho_esq
    10 joelho_dir  11 tornozelo_esq 12 tornozelo_dir

"esq/dir" é sempre do ponto de vista da PESSOA. Como a pessoa caminha na direção da
câmera, o lado esquerdo dela aparece do lado DIREITO da imagem.
"""
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

NOMES_ARTICULACOES = [
    "cabeca", "ombro_esq", "ombro_dir", "cotovelo_esq", "cotovelo_dir",
    "pulso_esq", "pulso_dir", "quadril_esq", "quadril_dir",
    "joelho_esq", "joelho_dir", "tornozelo_esq", "tornozelo_dir",
]
N_ARTICULACOES = len(NOMES_ARTICULACOES)

# índices do MediaPipe Pose (33 landmarks) equivalentes ao esqueleto canônico
INDICES_MEDIAPIPE = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


@dataclass
class Cena:
    """Parâmetros de uma cena do corredor."""

    duracao: float = 16.0          # segundos de vídeo
    futuro: float = 2.0            # segundos de trajetória além do vídeo (só GT)
    fps: int = 25
    largura: int = 640
    altura: int = 360

    horizonte_frac: float = 0.42   # fração da altura com o ponto de fuga
    razao_altura: float = 1.03     # altura da pessoa = razao * (y_pes - horizonte)
    largura_rel: float = 0.34      # largura do corpo = largura_rel * altura

    t_aparece: float = 1.0         # a pessoa só entra em cena depois deste tempo

    # trajetória (em frações da imagem, para independer da resolução)
    x0_frac: float = 0.42
    vx_frac: float = 0.030         # fração da largura por segundo
    curva: float = 0.02
    per_curva: float = 8.0
    y0_frac: float = 0.55
    y1_frac: float = 0.85

    # pilastra (primeiro plano): é o que causa a oclusão
    pilastra_x_frac: float = 0.60
    pilastra_w_frac: float = 0.07
    pilastra_topo_frac: float = 0.15
    pilastra_base_frac: float = 0.78


def posicao_pessoa(t, duracao_total, cena):
    """t em segundos -> (x_centro, y_dos_pes) em pixels, ou None se ainda não apareceu."""
    W, H = cena.largura, cena.altura
    dt = t - cena.t_aparece
    if dt <= 0:
        return None
    p = dt / max(1e-6, duracao_total - cena.t_aparece)
    x = (cena.x0_frac + cena.vx_frac * dt + cena.curva * math.sin(2 * math.pi * dt / cena.per_curva)) * W
    y = (cena.y0_frac + (cena.y1_frac - cena.y0_frac) * (min(p, 1.6) ** 1.15)) * H
    return x, y


def horizonte_px(cena):
    return int(cena.altura * cena.horizonte_frac)


def tamanho_pessoa(y_pes, horizonte, cena):
    return max(6.0, cena.razao_altura * (y_pes - horizonte))


# ----------------------------------------------------------------------------------
# Desenho do corredor
# ----------------------------------------------------------------------------------
def _poligono(img, pts, cor):
    cv2.fillPoly(img, [np.array(pts, np.int32)], cor, cv2.LINE_AA)


def fundo_corredor(cena, vp):
    """Corredor em perspectiva: piso xadrez, paredes com portas, teto com luminárias."""
    W, H = cena.largura, cena.altura
    vx, vy = vp
    e = H / 720.0  # fator de escala (o desenho original foi feito em 720p)
    fx1, fx2 = int(vx - 60 * e), int(vx + 60 * e)
    fy1, fy2 = int(vy - 45 * e), int(vy + 45 * e)

    img = np.zeros((H, W, 3), np.uint8)
    _poligono(img, [(0, 0), (W, 0), (fx2, fy1), (fx1, fy1)], (198, 200, 203))
    _poligono(img, [(0, 0), (fx1, fy1), (fx1, fy2), (0, H)], (170, 196, 214))
    _poligono(img, [(W, 0), (W, H), (fx2, fy2), (fx2, fy1)], (170, 196, 214))
    _poligono(img, [(0, H), (W, H), (fx2, fy2), (fx1, fy2)], (150, 151, 154))

    profs = [1 - 0.62 ** k for k in range(8)]

    def y_piso(d):
        return fy2 + (H - fy2) * d

    def y_teto(d):
        return fy1 * (1 - d)

    def x_esq(d):
        return fx1 * (1 - d)

    def x_dir(d):
        return fx2 + (W - fx2) * d

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

    for k in range(len(profs) - 1):
        d0, d1 = profs[k], profs[k + 1]
        _poligono(img, [(x_esq(d1), y_piso(d1) - 46 * e * (d1 + 0.15)), (x_esq(d0), y_piso(d0) - 46 * e * d0),
                         (x_esq(d0), y_piso(d0)), (x_esq(d1), y_piso(d1))], (120, 145, 165))
        _poligono(img, [(x_dir(d1), y_piso(d1) - 46 * e * (d1 + 0.15)), (x_dir(d0), y_piso(d0) - 46 * e * d0),
                         (x_dir(d0), y_piso(d0)), (x_dir(d1), y_piso(d1))], (120, 145, 165))
        if k % 2 == 0:
            _poligono(img, [(x_esq(d1), y_teto(d1) + 40 * e * d1 + 30 * e), (x_esq(d0), y_teto(d0) + 40 * e * d0 + 12 * e),
                             (x_esq(d0), y_piso(d0) - 46 * e * d0 - 12 * e), (x_esq(d1), y_piso(d1) - 46 * e * d1 - 30 * e)],
                      (95, 105, 120))
            _poligono(img, [(x_dir(d1), y_teto(d1) + 40 * e * d1 + 30 * e), (x_dir(d0), y_teto(d0) + 40 * e * d0 + 12 * e),
                             (x_dir(d0), y_piso(d0) - 46 * e * d0 - 12 * e), (x_dir(d1), y_piso(d1) - 46 * e * d1 - 30 * e)],
                      (95, 105, 120))
            mx = x_esq(d0) + 6 * e * d0 if d0 > 0.05 else x_esq(d0) + 3 * e
            my = (y_teto(d0) + y_piso(d0)) / 2
            cv2.circle(img, (int(mx), int(my)), max(1, int(4 * e * d0 + 1)), (210, 210, 215), -1, cv2.LINE_AA)

    for k in range(len(profs) - 2, -1, -1):
        d0, d1 = profs[k], profs[k + 1]
        _poligono(img, [(vx - 130 * e * d0, y_teto(d0) + 12 * e * d0), (vx + 130 * e * d0, y_teto(d0) + 12 * e * d0),
                         (vx + 130 * e * d1, y_teto(d1) + 12 * e * d1), (vx - 130 * e * d1, y_teto(d1) + 12 * e * d1)],
                  (238, 242, 246))

    _poligono(img, [(fx1 + 6 * e, fy1 + 4 * e), (fx2 - 6 * e, fy1 + 4 * e),
                    (fx2 - 6 * e, fy2 - 6 * e), (fx1 + 6 * e, fy2 - 6 * e)], (225, 235, 245))
    return img


def desenhar_pilastra(img, cena):
    """Pilastra do corredor (primeiro plano): desenhada POR CIMA da pessoa -> a oclui."""
    W, H = cena.largura, cena.altura
    e = H / 720.0
    px = cena.pilastra_x_frac * W
    pw = cena.pilastra_w_frac * W
    topo = cena.pilastra_topo_frac * H
    base = cena.pilastra_base_frac * H
    x1, x2 = int(px - pw / 2), int(px + pw / 2)

    sombra = img.copy()
    cv2.ellipse(sombra, (int(px), int(base)), (int(pw * 0.8), int(16 * e)), 0, 0, 360, (120, 121, 125), -1, cv2.LINE_AA)
    cv2.addWeighted(sombra, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (x1, int(topo)), (x2, int(base)), (175, 178, 182), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, int(topo)), (x1 + int(12 * e), int(base)), (150, 153, 158), -1)
    cv2.rectangle(img, (x2 - int(12 * e), int(topo)), (x2, int(base)), (198, 200, 203), -1)
    cv2.rectangle(img, (x1 - int(8 * e), int(base - 26 * e)), (x2 + int(8 * e), int(base)), (160, 163, 168), -1)
    return x1, x2, topo, base


# ----------------------------------------------------------------------------------
# POSE da pessoa
# ----------------------------------------------------------------------------------
def pontos_pessoa(cx, y_pes, h, fase):
    """Articulações da pessoa desenhada, em pixels, indexadas pelo LADO DA IMAGEM
    (-1 = lado esquerdo da imagem, +1 = lado direito)."""
    balanco = -0.018 * h * abs(math.cos(fase))
    quadril = y_pes - 0.50 * h + balanco * 0.4
    ombro = y_pes - 0.80 * h + balanco
    r = 0.075 * h
    cy_cab = y_pes - 0.90 * h + balanco

    p = {"cabeca": (cx, cy_cab), "ombro": {}, "cotovelo": {}, "pulso": {},
         "quadril": {}, "joelho": {}, "tornozelo": {}}

    for s in (-1, 1):  # braços (balanço contrário às pernas)
        a = -0.30 * math.sin(fase) * s
        ox, oy = cx + s * 0.135 * h, ombro
        comp = 0.32 * h
        ex, ey = ox + comp * math.sin(a), oy + comp * math.cos(a)
        p["ombro"][s] = (ox, oy)
        p["pulso"][s] = (ex, ey)
        p["cotovelo"][s] = ((ox + ex) / 2.0, (oy + ey) / 2.0)

    for s in (-1, 1):  # pernas
        a = 0.32 * math.sin(fase) * s
        ox, oy = cx + s * 0.06 * h, quadril
        comp = 0.50 * h
        p["quadril"][s] = (ox, oy)
        p["joelho"][s] = (ox + 0.5 * comp * math.sin(a) - 0.04 * h * s, oy + 0.5 * comp * math.cos(a))
        p["tornozelo"][s] = (ox + comp * math.sin(a), oy + comp * math.cos(a))

    return p


def pontos_canonicos(p):
    """Esqueleto canônico (lado da PESSOA). A pessoa caminha na direção da câmera,
    então o lado esquerdo dela fica do lado direito da imagem."""
    esq, dir_ = 1, -1
    return np.array([
        p["cabeca"],
        p["ombro"][esq], p["ombro"][dir_],
        p["cotovelo"][esq], p["cotovelo"][dir_],
        p["pulso"][esq], p["pulso"][dir_],
        p["quadril"][esq], p["quadril"][dir_],
        p["joelho"][esq], p["joelho"][dir_],
        p["tornozelo"][esq], p["tornozelo"][dir_],
    ], dtype=np.float64)


def desenhar_pessoa(camada, cx, y_pes, h, fase):
    """Desenha a pessoa. h = altura em pixels, fase = ciclo da caminhada."""
    pele, cabelo = (110, 160, 210), (25, 25, 30)
    camisa, calca, sapato = (45, 40, 35), (75, 65, 55), (25, 25, 30)

    p = pontos_pessoa(cx, y_pes, h, fase)
    balanco = -0.018 * h * abs(math.cos(fase))
    quadril = y_pes - 0.50 * h + balanco * 0.4
    ombro = y_pes - 0.80 * h + balanco
    r = 0.075 * h
    cy_cab = y_pes - 0.90 * h + balanco

    _poligono(camada, [(cx - 0.15 * h, ombro), (cx + 0.15 * h, ombro),
                        (cx + 0.12 * h, quadril + 0.02 * h), (cx - 0.12 * h, quadril + 0.02 * h)], camisa)
    cv2.ellipse(camada, (int(cx), int(cy_cab)), (int(r), int(r * 1.15)), 0, 0, 360, pele, -1, cv2.LINE_AA)
    cv2.ellipse(camada, (int(cx), int(cy_cab - r * 0.45)), (int(r), int(r * 0.75)), 0, 180, 360, cabelo, -1, cv2.LINE_AA)

    for s in (-1, 1):
        ox, oy = p["ombro"][s]
        ex, ey = p["pulso"][s]
        cv2.line(camada, (int(ox), int(oy)), (int(ex), int(ey)), camisa, max(2, int(0.055 * h)), cv2.LINE_AA)
        cv2.circle(camada, (int(ex), int(ey)), max(2, int(0.03 * h)), pele, -1, cv2.LINE_AA)

    for s in (-1, 1):
        ox, oy = p["quadril"][s]
        jx, jy = p["joelho"][s]
        px_, py_ = p["tornozelo"][s]
        cv2.line(camada, (int(ox), int(oy)), (int(jx), int(jy)), calca, max(2, int(0.075 * h)), cv2.LINE_AA)
        cv2.line(camada, (int(jx), int(jy)), (int(px_), int(py_)), calca, max(2, int(0.065 * h)), cv2.LINE_AA)
        cv2.ellipse(camada, (int(px_), int(py_)), (max(2, int(0.05 * h)), max(2, int(0.025 * h))),
                    0, 0, 360, sapato, -1, cv2.LINE_AA)


def sombra_suave(img, cx, y_pes, h):
    camada = img.copy()
    cv2.ellipse(camada, (int(cx), int(y_pes)), (int(0.20 * h), int(0.040 * h)), 0, 0, 360, (105, 106, 110), -1)
    cv2.addWeighted(camada, 0.5, img, 0.5, 0, img)


def visibilidade_articulacoes(pontos, pilastra, W, H, na_frente=False):
    """Visibilidade de cada articulação: a pilastra (primeiro plano) cobre a parte do
    corpo que está atrás dela. Uma articulação está visível se estiver fora da pilastra
    (e dentro da imagem). Se a pessoa já passou à frente da pilastra, nada é coberto."""
    x1, x2, topo, base = pilastra
    vis = np.zeros(N_ARTICULACOES, dtype=np.int32)
    if na_frente:
        return np.ones(N_ARTICULACOES, dtype=np.int32)
    for i, (x, y) in enumerate(pontos):
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        dentro = (x1 <= x <= x2) and (topo <= y <= base)
        vis[i] = 0 if dentro else 1
    return vis


# ----------------------------------------------------------------------------------
# Geração de uma cena
# ----------------------------------------------------------------------------------
def gerar_cena(cena, saida_video, saida_gt, saida_meta=None, escrever_video=True):
    """Gera o vídeo e o ground truth (pose + caixa + visibilidade) de uma cena.

    PROFUNDIDADE: a pilastra é o objeto mais próximo da câmera. A pessoa é escondida
    enquanto estiver ATRÁS dela; depois de passar à frente (y dos pés abaixo da base da
    pilastra) ela aparece por cima, como numa cena real.
    """
    W, H, fps = cena.largura, cena.altura, cena.fps
    horizonte = horizonte_px(cena)
    vp = (W // 2, horizonte)
    n_video = int(round(cena.duracao * fps))
    n_total = n_video + int(round(cena.futuro * fps))
    dur_total = n_total / fps

    fundo = fundo_corredor(cena, vp)

    escritor = None
    if escrever_video:
        Path(saida_video).parent.mkdir(parents=True, exist_ok=True)
        escritor = cv2.VideoWriter(str(saida_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not escritor.isOpened():
            raise SystemExit("Não consegui criar o vídeo (codec mp4v indisponível?).")

    gt = open(saida_gt, "w", newline="", encoding="utf-8")
    w = csv.writer(gt)
    cabecalho = ["quadro", "tempo_s", "dentro_do_video", "cobertura", "visivel", "na_cena",
                 "cx_real", "y_pes", "altura_real", "largura_real", "x1", "y1", "x2", "y2"]
    for nome in NOMES_ARTICULACOES:
        cabecalho += [f"{nome}_x", f"{nome}_y", f"{nome}_vis"]
    w.writerow(cabecalho)

    rect_pilastra = (int(cena.pilastra_x_frac * W - cena.pilastra_w_frac * W / 2),
                     int(cena.pilastra_x_frac * W + cena.pilastra_w_frac * W / 2),
                     cena.pilastra_topo_frac * H, cena.pilastra_base_frac * H)

    dist = 0.0
    pos_anterior = None
    n_ocluidos = n_parciais = 0
    for i in range(n_total):
        t = i / fps
        pos = posicao_pessoa(t, dur_total, cena)
        frame = fundo.copy()

        pilastra = np.zeros((H, W, 3), np.uint8)
        desenhar_pilastra(pilastra, cena)
        m_pilastra = pilastra.max(axis=2) > 0

        if pos is None:
            quad = (0, 0, 0, 0)
            cobertura, visivel, na_cena, cx, y_pes, h = 0.0, 0, 0, 0.0, 0.0, 0.0
            pontos = np.zeros((N_ARTICULACOES, 2))
            vis_art = np.zeros(N_ARTICULACOES, dtype=np.int32)
            frame[m_pilastra] = pilastra[m_pilastra]
        else:
            cx, y_pes = pos
            h = tamanho_pessoa(y_pes, horizonte, cena)
            if pos_anterior is not None:
                dist += abs(cx - pos_anterior[0])
            pos_anterior = pos
            fase = 2 * math.pi * dist / max(1e-6, 0.55 * h)

            sombra_suave(frame, cx, y_pes, h)
            camada = np.zeros((H, W, 3), np.uint8)
            desenhar_pessoa(camada, cx, y_pes, h, fase)
            m_pessoa = camada.max(axis=2) > 0

            # a pessoa está NA FRENTE da pilastra quando já passou por ela
            na_frente = y_pes > rect_pilastra[3]
            if na_frente:
                frame[m_pilastra] = pilastra[m_pilastra]
                frame[m_pessoa] = camada[m_pessoa]
                m_visivel = m_pessoa
            else:
                frame[m_pessoa] = camada[m_pessoa]
                frame[m_pilastra] = pilastra[m_pilastra]
                m_visivel = m_pessoa & ~m_pilastra

            area_total = int(m_pessoa.sum())
            area_vis = int(m_visivel.sum())
            cobertura = area_vis / area_total if area_total else 0.0
            ys, xs = np.nonzero(m_visivel)
            if area_vis > 0:
                quad = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            else:
                quad = (-1, -1, -1, -1)
            visivel = int(cobertura >= 0.35)
            na_cena = 1          # a pessoa existe na cena, mesmo que totalmente escondida
            if cobertura < 0.02:
                visivel = 0
                n_ocluidos += 1
            elif cobertura < 0.9:
                n_parciais += 1

            pontos = pontos_canonicos(pontos_pessoa(cx, y_pes, h, fase))
            vis_art = visibilidade_articulacoes(pontos, rect_pilastra, W, H, na_frente)

        if i < n_video and escritor is not None:
            escritor.write(frame)

        linha = [i, f"{t:.2f}", int(i < n_video), f"{cobertura:.3f}", visivel, na_cena,
                 f"{cx:.1f}", f"{y_pes:.1f}", f"{h:.1f}", f"{cena.largura_rel * h:.1f}",
                 quad[0], quad[1], quad[2], quad[3]]
        for k in range(N_ARTICULACOES):
            linha += [f"{pontos[k, 0]:.2f}", f"{pontos[k, 1]:.2f}", int(vis_art[k])]
        w.writerow(linha)

    if escritor is not None:
        escritor.release()
    gt.close()

    meta = {
        "cenario": "corredor interno de prédio",
        "fps": fps, "largura": W, "altura": H, "quadros_video": n_video, "quadros_total": n_total,
        "duracao_video_s": cena.duracao, "futuro_s": cena.futuro, "horizonte": horizonte,
        "t_aparece": cena.t_aparece, "pilastra_x": cena.pilastra_x_frac * W,
        "quadros_totalmente_ocultos": n_ocluidos, "quadros_parciais": n_parciais,
        "parametros": asdict(cena),
    }
    if saida_meta:
        Path(saida_meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
