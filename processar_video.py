"""Etapa 3: acompanha UMA pessoa em um vídeo, interpreta oclusões e PREVÊ para onde ela vai.

Pipeline:
  1. PERCEPÇÃO (CNN): um detector pré-treinado no COCO (Faster R-CNN) encontra a pessoa
     a cada --passo quadros. (Alternativa sem download: --detector fundo, que usa
     subtração de fundo e só funciona com câmera fixa e início do vídeo sem a pessoa.)
  2. RASTREIO: segue a mesma pessoa ao longo do tempo. Quando ela deixa de ser detectada
     depois de já ter sido vista, o quadro é marcado como OCLUSAO / pessoa perdida.
  3. PREVISÃO (Random Forest): aprende com a própria trajetória (velocidade recente, posição
     e tamanho na imagem) e prevê o deslocamento futuro do centro e da altura da caixa.
     Usa uma base de movimento constante e o RF corrige essa base ("modo residual").
     Dentro do vídeo o RF só usa dados do passado (treino "walk-forward"), então a previsão
     pode ser comparada com o que realmente aconteceu (erro impresso no final).
  4. SAÍDA: vídeo anotado + trecho EXTRA no final (--extra-segundos) mostrando a previsão
     de onde a pessoa vai, + CSVs (detecções e log) + métricas de erro.
"""
import argparse
import csv
from itertools import chain, islice
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------------------------------
# Cores (BGR) e utilidades de desenho
# ----------------------------------------------------------------------------------
VERDE, VERMELHO, AMARELO = (0, 200, 0), (0, 0, 255), (0, 220, 255)
LARANJA, CIANO, BRANCO, PRETO = (0, 140, 255), (255, 200, 0), (255, 255, 255), (0, 0, 0)


def escrever(img, texto, org, escala, cor, espessura=1):
    """Texto com contorno preto (legível em qualquer fundo). Só ASCII: OpenCV não faz acentos."""
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, PRETO, espessura + 2, cv2.LINE_AA)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, espessura, cv2.LINE_AA)


def linha_pontilhada(img, pontos, cor, espessura=2, passo=3):
    for i in range(0, len(pontos) - 1, passo * 2):
        j = min(i + passo, len(pontos) - 1)
        cv2.line(img, tuple(int(v) for v in pontos[i]), tuple(int(v) for v in pontos[j]),
                 cor, espessura, cv2.LINE_AA)


def caixa_de(linha):
    """linha = [x_centro, y_dos_pes, largura, altura] -> caixa (x1, y1, x2, y2)."""
    cx, fy, w, h = linha
    return (cx - w / 2, fy - h, cx + w / 2, fy)


def caixa_translucida(img, caixa, cor, alfa, borda=2):
    x1, y1, x2, y2 = [int(round(v)) for v in caixa]
    if alfa > 0:
        camada = img.copy()
        cv2.rectangle(camada, (x1, y1), (x2, y2), cor, -1)
        cv2.addWeighted(camada, alfa, img, 1 - alfa, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), cor, borda, cv2.LINE_AA)


def desenhar_hud(img, linhas):
    """linhas: lista de (texto, cor)."""
    w = img.shape[1]
    esc = max(0.5, w / 1700)
    passo = int(34 * esc)
    cv2.rectangle(img, (0, 0), (w, passo * len(linhas) + int(10 * esc)), PRETO, -1)
    y = int(26 * esc)
    for texto, cor in linhas:
        escrever(img, texto, (12, y), esc, cor)
        y += passo


def desenhar_previsao(img, caminho, incerteza, ate_i, cor):
    """Caminho previsto (pontilhado) até o índice 'ate_i' + caixa fantasma e círculo de incerteza."""
    pontos = caminho[:ate_i + 1, :2]
    if len(pontos) > 2:
        linha_pontilhada(img, pontos, cor, 3)
    cx, fy = caminho[ate_i][:2]
    caixa_translucida(img, caixa_de(caminho[ate_i]), cor, 0.25, 2)
    cv2.circle(img, (int(cx), int(fy)), int(max(10, incerteza[ate_i])), cor, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------------
# Leitura do vídeo (sem guardar todos os quadros na memória)
# ----------------------------------------------------------------------------------
def dim_saida(w0, h0, largura):
    if largura and w0 > largura:
        return largura, int(h0 * largura / w0)
    return w0, h0


def info_video(caminho, largura):
    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir o vídeo: {caminho}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    w, h = dim_saida(w0, h0, largura)
    return fps, n, w, h


def iterar_quadros(caminho, largura):
    cap = cv2.VideoCapture(str(caminho))
    try:
        while True:
            ok, q = cap.read()
            if not ok:
                break
            w, h = dim_saida(q.shape[1], q.shape[0], largura)
            if (w, h) != (q.shape[1], q.shape[0]):
                q = cv2.resize(q, (w, h), interpolation=cv2.INTER_AREA)
            yield q
    finally:
        cap.release()


# ----------------------------------------------------------------------------------
# 1. PERCEPÇÃO
# ----------------------------------------------------------------------------------
def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = lambda c: max(0, c[2] - c[0]) * max(0, c[3] - c[1])
    uniao = area(a) + area(b) - inter
    return inter / uniao if uniao > 0 else 0.0


def escolher_alvo(candidatos, ultimo, perdido_ha, max_perdido_quadros):
    """candidatos: lista de (caixa, score). Segue a pessoa anterior; se estava perdida há muito
    tempo (ou ainda não havia alvo), escolhe a mais proeminente (score x área)."""
    # descarta formas que não parecem uma pessoa (muito largas em relação à altura)
    candidatos = [(c, s) for c, s in candidatos
                  if (c[2] - c[0]) <= 1.6 * max(1.0, c[3] - c[1])]
    if not candidatos:
        return None
    if ultimo is not None and perdido_ha <= max_perdido_quadros:
        centro = lambda c: ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2)
        ux, uy = centro(ultimo)
        diag = max(1.0, np.hypot(ultimo[2] - ultimo[0], ultimo[3] - ultimo[1]))
        melhor, melhor_s = None, -1.0
        h_ult = max(1.0, ultimo[3] - ultimo[1])
        for caixa, score in candidatos:
            if not 0.5 <= (caixa[3] - caixa[1]) / h_ult <= 2.0:  # tamanho incompatível com o alvo
                continue
            x, y = centro(caixa)
            prox = 1.0 - min(1.0, np.hypot(x - ux, y - uy) / (3 * diag))
            s = iou(caixa, ultimo) + 0.5 * prox
            if s > melhor_s:
                melhor, melhor_s = (caixa, score), s
        return melhor if melhor_s > 0.05 else None
    return max(candidatos, key=lambda cs: cs[1] * (cs[0][2] - cs[0][0]) * (cs[0][3] - cs[0][1]))


class DetectorCNN:
    """Faster R-CNN pré-treinado no COCO (classe 'person' = 1). Baixa os pesos na 1ª execução."""

    def __init__(self, nome, tamanho_min, conf_min):
        import torch
        from torchvision.models import detection as det

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kw = dict(min_size=tamanho_min, max_size=int(tamanho_min * 1.8), box_score_thresh=conf_min)
        if nome == "resnet50":
            self.modelo = det.fasterrcnn_resnet50_fpn(weights=det.FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **kw)
        else:
            self.modelo = det.fasterrcnn_mobilenet_v3_large_fpn(
                weights=det.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT, **kw)
        self.modelo.eval().to(self.device)
        print(f"Detector: Faster R-CNN ({nome}) em {self.device}")

    def __call__(self, quadro_bgr):
        torch = self.torch
        rgb = np.ascontiguousarray(cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB))
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.no_grad():
            saida = self.modelo([x])[0]
        candidatos = []
        for caixa, rotulo, score in zip(saida["boxes"].cpu().numpy(), saida["labels"].cpu().numpy(),
                                        saida["scores"].cpu().numpy()):
            if int(rotulo) == 1:  # 1 = person no COCO
                candidatos.append((tuple(float(v) for v in caixa), float(score)))
        return candidatos


class DetectorFundo:
    """Alternativa sem download: subtração de fundo (câmera fixa; início do vídeo sem a pessoa).
    Fragmentos próximos são fundidos numa só caixa; o rastreio escolhe o que parece ser a pessoa."""

    def __init__(self, quadros_ref):
        cinza = [cv2.GaussianBlur(cv2.cvtColor(q, cv2.COLOR_BGR2GRAY), (7, 7), 0) for q in quadros_ref]
        self.ref = np.median(np.stack(cinza), axis=0).astype(np.uint8)
        print(f"Detector: subtração de fundo (referência = {len(quadros_ref)} quadros iniciais)")

    def __call__(self, quadro_bgr):
        h, w = quadro_bgr.shape[:2]
        k = lambda px: int(px * w / 1280) // 2 * 2 + 1  # tamanhos de kernel proporcionais à largura
        g = cv2.GaussianBlur(cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        m = (cv2.absdiff(g, self.ref) > 28).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k(5), k(5)), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k(31), k(31)), np.uint8))
        agrupado = cv2.dilate(m, np.ones((k(41), k(41)), np.uint8))  # junta fragmentos vizinhos
        contornos, _ = cv2.findContours(agrupado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_min = 0.0006 * h * w
        candidatos = []
        for c in contornos:
            mascara = np.zeros_like(m)
            cv2.drawContours(mascara, [c], -1, 255, -1)
            parte = cv2.bitwise_and(m, mascara)
            area = cv2.countNonZero(parte)
            if area >= area_min:
                x, y, bw, bh = cv2.boundingRect(parte)
                if y + bh < 0.40 * h:  # pés muito no alto da imagem: copa de árvores/céu, não a pessoa
                    continue
                candidatos.append(((float(x), float(y), float(x + bw), float(y + bh)), min(0.99, area / (8 * area_min))))
        return candidatos


def detectar(args, fps, n, W, H):
    """Roda o detector a cada --passo quadros e segue o alvo."""
    quadros = iterar_quadros(args.video, args.largura)
    if args.detector == "fundo":
        buf = list(islice(quadros, max(5, int(args.ref_segundos * fps))))
        detector = DetectorFundo(buf)
        quadros = chain(buf, quadros)
    else:
        detector = DetectorCNN(args.detector, args.tamanho_deteccao, args.conf_min)
    deteccoes, info = {}, {}
    ultimo, ultimo_i = None, -10**9
    max_perdido = int(args.max_perdido * fps)
    n_det = -(-n // args.passo)
    print(f"Detectando a pessoa em ~{n_det} quadros (1 a cada {args.passo})...")
    feitos = 0
    for i, q in enumerate(quadros):
        if i % args.passo:
            continue
        cands = detector(q)
        escolhido = escolher_alvo(cands, ultimo, i - ultimo_i, max_perdido)
        if escolhido is not None:
            caixa, score = escolhido
            deteccoes[i], info[i] = caixa, (len(cands), score)
            ultimo, ultimo_i = caixa, i
        feitos += 1
        if feitos % 20 == 0:
            print(f"  {feitos}/{n_det}", flush=True)
    return deteccoes, info


def salvar_deteccoes(caminho, deteccoes, info, W, H):
    """Salva coordenadas normalizadas (0-1), assim o CSV vale para qualquer --largura."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quadro", "x1", "y1", "x2", "y2", "score", "n_candidatos"])
        for i in sorted(deteccoes):
            x1, y1, x2, y2 = deteccoes[i]
            w.writerow([i, f"{x1 / W:.5f}", f"{y1 / H:.5f}", f"{x2 / W:.5f}", f"{y2 / H:.5f}",
                        f"{info[i][1]:.3f}", info[i][0]])


def carregar_deteccoes(caminho, W, H):
    deteccoes, info = {}, {}
    with open(caminho, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = int(r["quadro"])
            deteccoes[i] = (float(r["x1"]) * W, float(r["y1"]) * H, float(r["x2"]) * W, float(r["y2"]) * H)
            info[i] = (int(r["n_candidatos"]), float(r["score"]))
    return deteccoes, info


# ----------------------------------------------------------------------------------
# 2. RASTREIO -> série temporal por quadro
# ----------------------------------------------------------------------------------
def montar_serie(deteccoes, n_quadros, fps, max_lacuna_s, encolhimento_min=0.6):
    """Array (N,4) [x_centro, y_dos_pes, largura, altura] nos quadros detectados, interpolado entre
    detecções próximas (lacuna <= max_lacuna_s). Fora disso fica NaN (= pessoa não visível).

    Oclusão parcial: se a caixa encolhe de repente em relação à mediana do último 1,5 s
    (altura < encolhimento_min x mediana, ou largura < 0,5 x mediana), a detecção é considerada
    não confiável (pessoa parcialmente escondida) e NÃO entra na série. Devolve também o vetor
    'parcial' (True nesses quadros)."""
    serie = np.full((n_quadros, 4), np.nan)
    parcial = np.zeros(n_quadros, bool)
    idx = sorted(i for i in deteccoes if i < n_quadros)
    aceitas = []  # (quadro, largura, altura)
    for i in idx:
        x1, y1, x2, y2 = deteccoes[i]
        w, h = x2 - x1, y2 - y1
        recentes = [(ww, hh) for j, ww, hh in aceitas if i - j <= 1.5 * fps]
        if encolhimento_min > 0 and len(recentes) >= 3:
            w_med, h_med = np.median([r[0] for r in recentes]), np.median([r[1] for r in recentes])
            if h < encolhimento_min * h_med or w < 0.5 * w_med:
                parcial[i] = True
                continue
        aceitas.append((i, w, h))
        serie[i] = [(x1 + x2) / 2, y2, w, h]  # x do centro, y dos PÉS, largura, altura
    ok = [j for j, _, _ in aceitas]
    max_lac = int(max_lacuna_s * fps)
    for a, b in zip(ok[:-1], ok[1:]):
        if 1 < b - a <= max_lac:
            for k in range(4):
                serie[a + 1:b, k] = np.interp(np.arange(a + 1, b), [a, b], [serie[a, k], serie[b, k]])
    return serie, ~np.isnan(serie[:, 0]), parcial


def suavizar(serie, visivel, janela):
    """Média móvel causal (só usa o passado) para reduzir o tremor da caixa do detector."""
    s = serie.copy()
    for i in range(len(serie)):
        if visivel[i]:
            ini = max(0, i - janela + 1)
            s[i] = serie[ini:i + 1][visivel[ini:i + 1]].mean(axis=0)
    return s


# ----------------------------------------------------------------------------------
# 3. PREVISÃO (Random Forest)
# ----------------------------------------------------------------------------------
def atributos(serie, t, fps, largura, altura, lags_s):
    """Entrada do modelo no instante t (usa só quadros <= t): posição, tamanho e velocidades
    em várias janelas de tempo. Devolve (vetor, velocidades brutas por lag)."""
    cx, cy, w, h = serie[t]
    f = [cx / largura, cy / altura, h / altura]
    vels = []
    for lag_s in lags_s:
        j = max(0, t - int(round(lag_s * fps)))
        while j < t and np.isnan(serie[j, 0]):
            j += 1
        dt = max(1, t - j) / fps
        vx, vy = (cx - serie[j, 0]) / dt, (cy - serie[j, 1]) / dt
        vh = np.log(max(h, 1.0) / max(serie[j, 3], 1.0)) / dt
        f += [vx / largura, vy / altura, vh]
        vels.append((vx, vy, vh))
    return np.array(f), vels


def deslocamento_base(vels, lag_idx, s):
    """Base de movimento constante: velocidade (dx, dy, dlog_altura por s) x tempo."""
    vx, vy, vh = vels[lag_idx]
    return np.array([vx * s, vy * s, vh * s])


class PrevisorRF:
    """Um Random Forest por horizonte futuro. Cada um prevê (dx, dy, dlog_altura) a partir dos
    atributos de movimento. No modo residual, prevê a CORREÇÃO sobre o movimento constante."""

    def __init__(self, horizontes, fps, W, H, lags, lag_base, n_arvores, residual, semente, peso_rf=1.0):
        self.peso_rf = peso_rf  # 1 = confia totalmente na correção do RF; 0 = só movimento constante
        self.hs, self.fps, self.W, self.H = horizontes, fps, W, H
        self.lags, self.lag_base = lags, lag_base
        self.n_arvores, self.residual, self.semente = n_arvores, residual, semente
        self.modelos = {}

    def treinar(self, serie, visivel, ate_t):
        """Treina só com instantes t cujo alvo t+h já aconteceu até 'ate_t' (sem olhar o futuro)."""
        passo = max(1, int(self.fps // 8))
        X, V, alvos = [], [], {k: [] for k in range(len(self.hs))}
        for t in range(0, ate_t + 1, passo):
            if not visivel[t]:
                continue
            validos = {}
            for k, h_s in enumerate(self.hs):
                tf = t + int(round(h_s * self.fps))
                if tf <= ate_t and visivel[tf]:
                    validos[k] = tf
            if not validos:
                continue
            x, vels = atributos(serie, t, self.fps, self.W, self.H, self.lags)
            X.append(x)
            V.append(vels)
            for k, tf in validos.items():
                d = [serie[tf, 0] - serie[t, 0], serie[tf, 1] - serie[t, 1],
                     np.log(max(serie[tf, 3], 1.0) / max(serie[t, 3], 1.0))]
                alvos[k].append((len(X) - 1, d))
        self.modelos = {}
        if not X:
            return
        X = np.array(X)
        for k, h_s in enumerate(self.hs):
            if len(alvos[k]) < 8:
                continue
            idx = [i for i, _ in alvos[k]]
            Y = np.array([d for _, d in alvos[k]])
            if self.residual:
                Y = Y - np.array([deslocamento_base(V[i], self.lag_base, h_s) for i in idx])
            rf = RandomForestRegressor(n_estimators=self.n_arvores, min_samples_leaf=3,
                                       random_state=self.semente, n_jobs=-1)
            rf.fit(X[idx], Y)
            self.modelos[k] = rf

    @property
    def pronto(self):
        return bool(self.modelos)

    def prever(self, serie, t):
        """Para cada horizonte treinado: (h_s, dx, dy, dlogh, incerteza_px). A incerteza é o
        desvio-padrão entre as árvores da floresta."""
        x, vels = atributos(serie, t, self.fps, self.W, self.H, self.lags)
        x32 = x[None].astype(np.float32)
        saida = []
        for k, h_s in enumerate(self.hs):
            if k not in self.modelos:
                continue
            arvores = np.stack([a.predict(x32, check_input=False)[0] for a in self.modelos[k].estimators_])
            d, dp = arvores.mean(axis=0), arvores.std(axis=0)
            if self.residual:
                d = self.peso_rf * d + deslocamento_base(vels, self.lag_base, h_s)
            saida.append((h_s, d[0], d[1], d[2], float(np.hypot(dp[0], dp[1]))))
        return saida


def caminho_previsto(serie, t, previsao, fps, ate_s):
    """Pontos [x_centro, y_pes, largura, altura] a cada 1/fps a partir de t. Como há um RF por
    horizonte, ajusta uma curva suave (a*t + b*t^2, partindo da posição atual) pelas previsões
    para o caminho não ficar em zigue-zague."""
    cx, fy, w, h = serie[t]
    hs = np.array([p[0] for p in previsao])
    ts = np.arange(0, min(ate_s, hs[-1]) + 1e-9, 1.0 / fps)
    A = np.stack([hs, hs ** 2], axis=1) if len(hs) >= 3 else hs[:, None]
    B = np.stack([ts, ts ** 2], axis=1) if len(hs) >= 3 else ts[:, None]

    def curva(j):
        coef, *_ = np.linalg.lstsq(A, np.array([p[j] for p in previsao]), rcond=None)
        return B @ coef

    Ha = h * np.exp(curva(3))
    return ts, np.stack([cx + curva(1), fy + curva(2), Ha * (w / max(h, 1.0)), Ha], axis=1), \
        np.interp(ts, hs, [p[4] for p in previsao])


def previsao_base(serie, t, fps, W, H, lags, lag_base, ate_s):
    """Linha de base: movimento constante (mantém a velocidade recente para sempre)."""
    _, vels = atributos(serie, t, fps, W, H, lags)
    cx, cy, w, h = serie[t]
    ts = np.arange(0, ate_s + 1e-9, 1.0 / fps)
    d = np.stack([deslocamento_base(vels, lag_base, s) for s in ts])
    Ha = h * np.exp(d[:, 2])
    return np.stack([cx + d[:, 0], cy + d[:, 1], Ha * (w / max(h, 1.0)), Ha], axis=1)


# ----------------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--saida", default="resultados/video_previsao.mp4")
    p.add_argument("--detector", default="mobilenet", choices=["mobilenet", "resnet50", "fundo"],
                   help="mobilenet (rápido) / resnet50 (mais preciso, lento) / fundo (sem download)")
    p.add_argument("--passo", type=int, default=3, help="detecta a cada N quadros")
    p.add_argument("--conf-min", type=float, default=0.5, help="confiança mínima do detector")
    p.add_argument("--tamanho-deteccao", type=int, default=480, help="lado menor da imagem dentro do detector")
    p.add_argument("--largura", type=int, default=1280, help="largura do vídeo de saída (0 = original)")
    p.add_argument("--max-perdido", type=float, default=2.0,
                   help="segundos sem detecção antes de aceitar outra pessoa como alvo")
    p.add_argument("--max-lacuna", type=float, default=0.6,
                   help="até quantos segundos sem detecção são interpolados (acima disso = oclusão)")
    p.add_argument("--encolhimento-min", type=float, default=0.6,
                   help="se a altura da caixa cair abaixo desta fração da mediana recente, "
                        "considera oclusão parcial e ignora a caixa (0 = desliga)")
    p.add_argument("--ref-segundos", type=float, default=1.5, help="[fundo] segundos iniciais sem a pessoa")
    p.add_argument("--extra-segundos", type=float, default=4.0, help="duração extra de previsão no fim (3 a 5 s)")
    p.add_argument("--horizonte-max", type=float, default=0.0,
                   help="horizonte máximo de previsão em s (0 = automático)")
    p.add_argument("--arvores", type=int, default=100)
    p.add_argument("--modo-rf", default="residual", choices=["residual", "direto"])
    p.add_argument("--peso-rf", type=float, default=0.5,
                   help="[modo residual] peso da correção do RF sobre o movimento constante (0 a 1)")
    p.add_argument("--retreino", type=float, default=1.5, help="a cada quantos s o RF é retreinado (walk-forward)")
    p.add_argument("--mostrar-previsao", type=float, default=2.0,
                   help="dentro do vídeo, quantos s à frente desenhar a previsão (0 = não desenhar)")
    p.add_argument("--deteccoes", default="resultados/deteccoes.csv")
    p.add_argument("--reusar-deteccoes", action="store_true",
                   help="usa o CSV de detecções existente (pula a CNN; bom para ajustar a previsão)")
    p.add_argument("--semente", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.semente)
    fps, N, W, H = info_video(args.video, args.largura)
    print(f"Vídeo: {W}x{H} (saída), {fps:.1f} fps, {N} quadros ({N / fps:.1f} s)")

    # ---------- 1 e 2. percepção + rastreio ----------
    caminho_det = Path(args.deteccoes)
    if args.reusar_deteccoes and caminho_det.exists():
        deteccoes, info = carregar_deteccoes(caminho_det, W, H)
        print(f"Detecções carregadas de {caminho_det} ({len(deteccoes)} quadros)")
    else:
        deteccoes, info = detectar(args, fps, N, W, H)
        salvar_deteccoes(caminho_det, deteccoes, info, W, H)
        print(f"Detecções salvas em {caminho_det}")
    if len(deteccoes) < 10:
        raise SystemExit("Poucas detecções da pessoa. Tente --detector resnet50, reduza --conf-min "
                         "ou confira se a pessoa aparece no vídeo.")

    serie_bruta, visivel, parcial = montar_serie(deteccoes, N, fps, args.max_lacuna, args.encolhimento_min)
    parcial_ativo = np.zeros(N, bool)  # "oclusão parcial" fica ativa por um curto tempo após cada sinalização
    for f in np.flatnonzero(parcial):
        parcial_ativo[f:f + 2 * args.passo + 1] = True
    serie = suavizar(serie_bruta, visivel, janela=max(1, int(0.4 * fps)))
    primeiro, ultimo = int(np.argmax(visivel)), int(N - 1 - np.argmax(visivel[::-1]))
    print(f"Pessoa visível de {primeiro / fps:.1f} s a {ultimo / fps:.1f} s")

    # ---------- 3. previsão ----------
    resto_s = (N - 1 - ultimo) / fps  # tempo entre a última detecção e o fim do vídeo (oclusão final)
    h_max = args.horizonte_max or float(np.clip(np.ceil(resto_s + args.extra_segundos), 3.0, 8.0))
    horizontes = [float(x) for x in np.arange(1.0, h_max + 1e-9, 1.0)]
    lags, lag_base = [0.5, 1.0, 2.0], 1  # a base de movimento constante usa a velocidade da janela de 1 s
    prev = PrevisorRF(horizontes, fps, W, H, lags, lag_base, args.arvores, args.modo_rf == "residual",
                      args.semente, args.peso_rf)

    mostrar_s = args.mostrar_previsao
    janela_s = max(mostrar_s, 3.0)
    previsoes_t = {}
    erros = {s: {"rf": [], "base": []} for s in (1.0, 2.0, 3.0)}
    ultimo_treino = -10**9
    print("Treinando o Random Forest (walk-forward, só com dados do passado)...")
    for t in range(primeiro, ultimo + 1):
        if not visivel[t] or t % args.passo:
            continue
        if t - ultimo_treino >= args.retreino * fps:
            prev.treinar(serie, visivel, t)
            ultimo_treino = t
        if not prev.pronto:
            continue
        pr = prev.prever(serie, t)
        if not pr:
            continue
        ts, cam, inc = caminho_previsto(serie, t, pr, fps, janela_s)
        cam_b = previsao_base(serie, t, fps, W, H, lags, lag_base, janela_s)
        previsoes_t[t] = (cam, inc)
        for s in erros:
            tf, k = t + int(round(s * fps)), int(round(s * fps))
            if tf < N and visivel[tf] and k < len(cam):
                alvo = serie[tf, :2]
                erros[s]["rf"].append(float(np.hypot(*(cam[k, :2] - alvo))))
                erros[s]["base"].append(float(np.hypot(*(cam_b[k, :2] - alvo))))

    linhas_met = [f"Erro médio da posição prevista (pixels; vídeo com {W} px de largura)",
                  "horizonte | Random Forest | movimento constante | amostras"]
    for s, e in erros.items():
        if e["rf"]:
            linhas_met.append(f"  +{s:.0f} s    | {np.mean(e['rf']):9.1f}     | {np.mean(e['base']):14.1f}      | {len(e['rf'])}")
    print("\n" + "\n".join(linhas_met))

    # Modelo final (com todo o passado) para o trecho extra, a partir da última detecção
    prev.treinar(serie, visivel, ultimo)
    horizonte_extra = resto_s + args.extra_segundos
    pr_final = prev.prever(serie, ultimo) if prev.pronto else []
    cam_f = None
    if pr_final:
        ts_f, cam_f, inc_f = caminho_previsto(serie, ultimo, pr_final, fps, horizonte_extra)
        cam_fb = previsao_base(serie, ultimo, fps, W, H, lags, lag_base, ts_f[-1])
        if ts_f[-1] + 1e-6 < horizonte_extra:
            print(f"Aviso: a previsão só alcança {ts_f[-1]:.1f} s (necessário {horizonte_extra:.1f} s). "
                  "Aumente --horizonte-max ou reduza --extra-segundos.")
    else:
        print("Aviso: poucos dados para treinar o RF no final; o trecho extra ficará sem previsão.")

    # ---------- 4. saída ----------
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    escritor = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not escritor.isOpened():
        raise SystemExit("Não consegui criar o vídeo de saída (codec mp4v indisponível?).")
    log = open(saida.with_suffix(".csv"), "w", newline="", encoding="utf-8")
    wl = csv.writer(log)
    wl.writerow(["quadro", "tempo_s", "estado", "x_centro", "y_pes", "altura"])

    trilha, ultimo_visto, ultimo_prev, ultimo_q = [], primeiro, None, None
    print("Gerando o vídeo de saída...")
    for i, q in enumerate(iterar_quadros(args.video, args.largura)):
        if i >= N:
            break
        ultimo_q = q
        img = q.copy()
        if i in previsoes_t:
            ultimo_prev = i
        if visivel[i]:
            estado, ultimo_visto = "RASTREANDO", i
            caixa = caixa_de(serie[i])
            trilha.append(tuple(serie[i, :2]))  # trilha dos pés no chão
            caixa_translucida(img, caixa, VERDE, 0.15, 3)
            escrever(img, "pessoa", (int(caixa[0]), max(20, int(caixa[1]) - 8)), 0.6, VERDE)
            linhas = [(f"t = {i / fps:5.1f} s  |  Estado: RASTREANDO (CNN)", VERDE)]
            if mostrar_s > 0 and ultimo_prev is not None and i - ultimo_prev <= 2 * args.passo:
                cam, inc = previsoes_t[ultimo_prev]
                n_pts = min(int(round(mostrar_s * fps)), len(cam) - 1)
                desenhar_previsao(img, cam, inc, n_pts, LARANJA)
                linhas.append((f"Laranja: previsao do Random Forest para {mostrar_s:.0f} s a frente", LARANJA))
        elif i < primeiro:
            estado = "AGUARDANDO"
            linhas = [(f"t = {i / fps:5.1f} s  |  Estado: aguardando a pessoa aparecer", BRANCO)]
        else:
            estado = "OCLUSAO"
            seg = (i - ultimo_visto) / fps
            if parcial_ativo[i]:
                motivo = "OCLUSAO PARCIAL - a caixa da pessoa encolheu (parcialmente escondida)"
            else:
                motivo = f"OCLUSAO - pessoa nao detectada ha {seg:.1f} s"
            linhas = [(f"t = {i / fps:5.1f} s  |  Estado: {motivo}", VERMELHO),
                      ("Caixa vermelha: posicao estimada pelo modelo", VERMELHO)]
            est = None
            if i > ultimo and cam_f is not None:
                est = cam_f[min(i - ultimo, len(cam_f) - 1)]
            elif ultimo_prev is not None:
                cam, _ = previsoes_t[ultimo_prev]
                if i - ultimo_prev < len(cam):
                    est = cam[i - ultimo_prev]
            if est is not None:
                caixa_translucida(img, caixa_de(est), VERMELHO, 0.2, 2)
        if len(trilha) > 1:
            cv2.polylines(img, [np.array(trilha[-int(6 * fps):], np.int32)], False, CIANO, 2, cv2.LINE_AA)
        desenhar_hud(img, linhas)
        escritor.write(img)
        ok = visivel[i]
        wl.writerow([i, f"{i / fps:.2f}", estado, f"{serie[i, 0]:.1f}" if ok else "",
                     f"{serie[i, 1]:.1f}" if ok else "", f"{serie[i, 3]:.1f}" if ok else ""])

    # trecho EXTRA: último quadro congelado + previsão animada
    n_extra = int(round(args.extra_segundos * fps))
    base_img = ultimo_q if ultimo_q is not None else np.zeros((H, W, 3), np.uint8)
    for e in range(n_extra):
        img = base_img.copy()
        caixa = caixa_de(serie[ultimo])
        caixa_translucida(img, caixa, VERDE, 0.0, 2)
        escrever(img, "ultima posicao detectada", (int(caixa[0]), max(20, int(caixa[1]) - 8)), 0.6, VERDE)
        if len(trilha) > 1:
            cv2.polylines(img, [np.array(trilha[-int(6 * fps):], np.int32)], False, CIANO, 2, cv2.LINE_AA)
        if cam_f is not None:
            k = int(np.clip((N - 1 - ultimo) + e + 1, 0, len(cam_f) - 1))
            linha_pontilhada(img, cam_fb[:, :2], AMARELO, 2)
            desenhar_previsao(img, cam_f, inc_f, k, LARANJA)
            for s in range(1, int(args.extra_segundos) + 1):  # marcadores: 1 s, 2 s... depois do fim do vídeo
                kk = (N - 1 - ultimo) + int(round(s * fps))
                if 0 < kk <= k and kk < len(cam_f):
                    pt = (int(cam_f[kk, 0]), int(cam_f[kk, 1]))
                    cv2.circle(img, pt, 5, LARANJA, -1, cv2.LINE_AA)
                    escrever(img, f"+{s}s", (pt[0] + 8, pt[1] - 6), 0.6, LARANJA)
        desenhar_hud(img, [(f"PREVISAO  |  +{(e + 1) / fps:3.1f} s alem do fim do video", LARANJA),
                           ("Laranja: Random Forest  |  Amarelo: movimento constante (base)", BRANCO),
                           ("Sem video real nesta parte: a posicao da pessoa e uma estimativa", BRANCO)])
        escritor.write(img)
        wl.writerow([N + e, f"{(N + e) / fps:.2f}", "PREVISAO", "", "", ""])

    escritor.release()
    log.close()
    saida.with_name(saida.stem + "_metricas.txt").write_text("\n".join(linhas_met), encoding="utf-8")
    print(f"\nPronto! {saida}  ({N / fps:.1f} s do vídeo + {args.extra_segundos:.1f} s de previsão"
          f" = {(N + n_extra) / fps:.1f} s)")
    print(f"Log: {saida.with_suffix('.csv')}  |  Métricas: {saida.with_name(saida.stem + '_metricas.txt')}")


if __name__ == "__main__":
    main()
