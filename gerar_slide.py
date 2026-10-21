"""Gera a apresentação (PPTX) com os resultados da comparação de modelos temporais.

A apresentação tem 4 slides:
  1. Capa / resumo executivo (o slide principal)
  2. Protocolo (como a comparação foi feita)
  3. Resultados (tabela + gráficos)
  4. Oclusão, conclusões e limitações

Uso:
  python gerar_slide.py --resultados resultados --saida resultados/apresentacao_modelos.pptx
"""
import argparse
import csv
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ----------------------------------------------------------------------------------
# PALETA E TIPOGRAFIA
# ----------------------------------------------------------------------------------
INK = RGBColor(0x0F, 0x17, 0x2A)
SUAVE = RGBColor(0x64, 0x74, 0x8B)
CLARO = RGBColor(0xE2, 0xE8, 0xF0)
FUNDO = RGBColor(0xFF, 0xFF, 0xFF)
CARD = RGBColor(0xF1, 0xF5, 0xF9)
AZUL = RGBColor(0x25, 0x63, 0xEB)
LARANJA = RGBColor(0xF9, 0x73, 0x16)
VERDE = RGBColor(0x16, 0xA3, 0x4A)
VERMELHO = RGBColor(0xDC, 0x26, 0x26)

CORES_MODELO = {
    "lstm": RGBColor(0x25, 0x63, 0xEB),
    "transformer": RGBColor(0x7C, 0x3A, 0xED),
    "tgnn": RGBColor(0x08, 0x91, 0xB2),
    "gru": RGBColor(0xF5, 0x9E, 0x0B),
    "ekf": RGBColor(0x64, 0x74, 0x8B),
    "rf": RGBColor(0xDC, 0x26, 0x26),
    "base_movimento_constante": RGBColor(0x94, 0xA3, 0xB8),
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

FONTE = "Segoe UI"        # cai para Arial/Liberation em outros sistemas
FONTE_TITULO = "Segoe UI Semibold"

LARGURA, ALTURA = Inches(13.333), Inches(7.5)
MARGEM = Inches(0.62)


# ----------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------
def texto(slide, x, y, w, h, txt, size=18, bold=False, cor=INK, align=PP_ALIGN.LEFT,
          fonte=FONTE, espaco=1.0, ancorar=MSO_ANCHOR.TOP, italic=False):
    """Adiciona uma caixa de texto com controle de linha (quebra manual por \\n)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = ancorar
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    linhas = txt.split("\n")
    for i, linha in enumerate(linhas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = espaco
        r = p.add_run()
        r.text = linha
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = cor
        r.font.name = fonte
    return tb


def retangulo(slide, x, y, w, h, cor=CARD, borda=None, raio=None, sombra=False):
    forma = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if raio is not None else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if raio is not None:
        # raio do canto (proporcional ao menor lado)
        try:
            forma.adjustments[0] = raio
        except (IndexError, ValueError):
            pass
    forma.fill.solid()
    forma.fill.fore_color.rgb = cor
    if borda is not None:
        forma.line.color.rgb = borda
        forma.line.width = Pt(1.25)
    else:
        forma.line.fill.background()
    # `shadow.inherit = False` desliga a sombra herdada do tema. NÃO se deve remover o
    # elemento de sombra na mão: o XML dele é o spPr, e apagá-lo leva junto o
    # preenchimento e a borda da forma (aí o cartão some).
    forma.shadow.inherit = False
    forma.text_frame.text = ""
    return forma


def linha(slide, x, y, w, cor=CLARO, espessura=1.0):
    forma = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, Pt(espessura))
    forma.fill.solid()
    forma.fill.fore_color.rgb = cor
    forma.line.fill.background()
    forma.shadow.inherit = False
    return forma


def sem_borda(forma):
    """Remove a borda de uma forma mantendo o preenchimento."""
    forma.line.fill.background()
    return forma


def titulo_slide(slide, titulo, subtitulo=None, numero=None):
    """Cabeçalho padrão: título, régua colorida e subtítulo."""
    texto(slide, MARGEM, Inches(0.42), Inches(11.4), Inches(0.62), titulo,
          size=30, bold=True, cor=INK, fonte=FONTE_TITULO)
    linha(slide, MARGEM, Inches(1.06), Inches(1.05), cor=AZUL, espessura=3.5)
    if subtitulo:
        texto(slide, MARGEM, Inches(1.22), Inches(11.6), Inches(0.42), subtitulo,
              size=14.5, cor=SUAVE)
    if numero:
        texto(slide, Inches(12.35), Inches(0.42), Inches(0.4), Inches(0.4), numero,
              size=13, cor=CLARO, align=PP_ALIGN.RIGHT, bold=True)


def rodape(slide, texto_rodape):
    linha(slide, MARGEM, Inches(6.92), Inches(12.1), cor=CLARO)
    texto(slide, MARGEM, Inches(7.02), Inches(12.1), Inches(0.3), texto_rodape,
          size=10, cor=SUAVE)


def bloco_numero(slide, x, y, w, h, numero, rotulo, cor=AZUL, sub=None):
    """Cartão com um número grande (KPI)."""
    retangulo(slide, x, y, w, h, cor=CARD, borda=CLARO, raio=0.06)
    retangulo(slide, x, y, Inches(0.055), h, cor=cor)   # barra de cor
    texto(slide, x + Inches(0.26), y + Inches(0.16), w - Inches(0.4), Inches(0.5),
          numero, size=27, bold=True, cor=cor, fonte=FONTE_TITULO)
    texto(slide, x + Inches(0.26), y + Inches(0.66), w - Inches(0.42), Inches(0.46),
          rotulo, size=11, cor=SUAVE, bold=True, espaco=1.1)
    if sub:
        texto(slide, x + Inches(0.26), y + Inches(1.08), w - Inches(0.42), Inches(0.3),
              sub, size=9.5, cor=SUAVE)


def marcador(slide, x, y, cor, titulo, descricao, w=Inches(5.6), size_t=15, size_d=12,
             altura_titulo=0.30):
    """Item com marcador colorido + título + descrição.

    O espaço da descrição é calculado pelo TAMANHO DO TEXTO (nº de caracteres por
    linha), para que a caixa seguinte nunca sobrescreva o texto: sem isso, uma
    descrição de 2 linhas invade o título de baixo.
    """
    retangulo(slide, x, y + Inches(0.04), Inches(0.15), Inches(0.15), cor=cor, raio=0.5)
    texto(slide, x + Inches(0.3), y - Inches(0.03), w - Inches(0.3), Inches(0.32),
          titulo, size=size_t, bold=True, cor=INK, espaco=1.05)
    texto(slide, x + Inches(0.3), y + Inches(altura_titulo), w - Inches(0.3), Inches(0.62),
          descricao, size=size_d, cor=SUAVE, espaco=1.16)
    return altura_titulo


# ----------------------------------------------------------------------------------
# DADOS
# ----------------------------------------------------------------------------------
def ler(pasta):
    with open(Path(pasta) / "comparacao_modelos.csv", newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    geral = {l["modelo"]: l for l in linhas if not l.get("dominio")}
    dom = {}
    for l in linhas:
        if l.get("dominio"):
            dom.setdefault(l["dominio"], {})[l["modelo"]] = l
    treino = {}
    arq = Path(pasta) / "modelos" / "treinamento.json"
    if arq.exists():
        import json
        treino = json.loads(arq.read_text(encoding="utf-8"))
    return geral, dom, treino


# ----------------------------------------------------------------------------------
# SLIDES
# ----------------------------------------------------------------------------------
def slide_capa(prs, geral, treino):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    # faixa lateral colorida
    retangulo(s, Inches(0), Inches(0), Inches(0.22), ALTURA, cor=AZUL)
    retangulo(s, Inches(0.22), Inches(0), Inches(0.07), ALTURA, cor=LARANJA)

    texto(s, MARGEM, Inches(0.62), Inches(9), Inches(0.3),
          "INTELIGÊNCIA ARTIFICIAL 2  ·  ETAPA 4  ·  VISÃO COMPUTACIONAL",
          size=11.5, bold=True, cor=AZUL)
    texto(s, MARGEM, Inches(1.05), Inches(11.9), Inches(1.5),
          "Prever a pose de uma pessoa\nenquanto a câmera a esconde",
          size=40, bold=True, cor=INK, espaco=1.02, fonte=FONTE_TITULO)
    texto(s, MARGEM, Inches(2.42), Inches(11.6), Inches(0.5),
          "LSTM · GRU · EKF · Transformer temporal · T-GNN — comparados no mesmo "
          "protocolo, treinados com vídeo real e vídeo sintético",
          size=15, cor=SUAVE, espaco=1.2)

    linha(s, MARGEM, Inches(3.05), Inches(12.1), cor=CLARO)

    # --- 3 KPIs ---
    lstm = geral["lstm"]
    base = geral["base_movimento_constante"]
    ganho = (float(base["mae_0_5s"]) - float(lstm["mae_0_5s"])) / float(base["mae_0_5s"]) * 100
    y = Inches(3.25)
    h = Inches(1.28)
    larg = Inches(3.9)
    gap = Inches(0.24)
    bloco_numero(s, MARGEM, y, larg, h, f"{float(lstm['mae_0_5s']):.3f}",
                 "ERRO DE POSE A 0,5 s  (LSTM, o melhor)",
                 cor=AZUL, sub="em alturas do corpo — 1,0 = a pessoa de lado")
    bloco_numero(s, MARGEM + larg + gap, y, larg, h, f"−{ganho:.0f}%",
                 "MELHORIA SOBRE MOVIMENTO CONSTANTE",
                 cor=VERDE, sub="base sem nenhum modelo de aprendizado")
    bloco_numero(s, MARGEM + 2 * (larg + gap), y, larg, h, "1.281",
                 "AMOSTRAS (TREINO)  ·  VÍDEO REAL + SINTÉTICO",
                 cor=LARANJA, sub="24 cenas sintéticas + 23 s de vídeo gravado")

    # --- linha de destaque ---
    y2 = Inches(4.72)
    retangulo(s, MARGEM, y2, Inches(12.1), Inches(2.06), cor=RGBColor(0xF1, 0xF5, 0xF9),
               borda=CLARO, raio=0.04)
    texto(s, MARGEM + Inches(0.32), y2 + Inches(0.16), Inches(5.4), Inches(0.28),
          "O QUE A COMPARAÇÃO MOSTRA", size=10.5, bold=True, cor=AZUL)
    # os 4 destaques: slots FIXOS (linha 1 e linha 2), sem depender do tamanho do texto
    xa, xb = MARGEM + Inches(0.32), MARGEM + Inches(6.35)
    wa, wb = Inches(5.45), Inches(5.35)
    y_l1 = y2 + Inches(0.50)
    y_l2 = y2 + Inches(1.24)
    marcador(s, xa, y_l1, AZUL,
             "As redes vencem — e por pouco",
             "LSTM e Transformer ficam empatados (~3%). Com 1,3 mil amostras, "
             "o volume de dados, não a arquitetura, é o gargalo.",
             w=wa, size_t=13.5, size_d=11, altura_titulo=0.27)
    marcador(s, xa, y_l2, LARANJA,
             "O EKF impressiona sem treinar",
             "O filtro de Kalman chega perto das redes: o estado (posição, velocidade, "
             "escala) já é uma boa hipótese temporal.",
             w=wa, size_t=13.5, size_d=11, altura_titulo=0.27)
    marcador(s, xb, y_l1, VERMELHO,
             "O modelo clássico da Etapa 3 fica atrás",
             "O Random Forest erra 2,7× mais que o LSTM a 0,5 s: não há sequência "
             "nem noção de forma do corpo.",
             w=wb, size_t=13.5, size_d=11, altura_titulo=0.27)
    marcador(s, xb, y_l2, AZUL,
             "Sob oclusão, a vantagem encolhe",
             "O erro sobe ~20% para todos: prever a pose escondida é o problema "
             "realmente difícil.",
             w=wb, size_t=13.5, size_d=11, altura_titulo=0.27)

    rodape(s, "Teste: 498 amostras · 25 quadros de entrada (1 s) → 50 quadros de saída (2 s) · "
              "erro em unidades de altura do corpo")
    return s


def slide_protocolo(prs, geral):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    retangulo(s, Inches(0), Inches(0), Inches(0.22), ALTURA, cor=AZUL)
    titulo_slide(s, "Um protocolo, cinco modelos", "O que muda entre os modelos é só a "
                 "arquitetura — entrada, saída, divisão e métrica são idênticos.", "02")

    # --- fluxograma ---
    y = Inches(1.95)
    h = Inches(1.30)
    larg = Inches(2.28)
    gap = Inches(0.30)
    x0 = MARGEM

    etapas = [
        ("DADOS", "Vídeo real\n+ 24 cenas sintéticas", RGBColor(0x0F, 0x17, 0x2A)),
        ("POSE", "MediaPipe (real)\n+ GT absoluto (sintético)", AZUL),
        ("JANELA", "1 s de entrada\n→ 2 s de saída", RGBColor(0x7C, 0x3A, 0xED)),
        ("MODELOS", "LSTM · GRU · EKF\nTransformer · T-GNN", LARANJA),
        ("MÉTRICA", "MAE por horizonte\nerro sob oclusão", VERDE),
    ]
    for i, (rot, desc, cor) in enumerate(etapas):
        x = x0 + i * (larg + gap)
        retangulo(s, x, y, larg, h, cor=FUNDO, borda=CLARO, raio=0.08)
        linha(s, x, y, larg, cor=cor, espessura=3.2)
        texto(s, x + Inches(0.2), y + Inches(0.2), larg - Inches(0.4), Inches(0.26),
              rot, size=10.5, bold=True, cor=cor)
        texto(s, x + Inches(0.2), y + Inches(0.52), larg - Inches(0.4), Inches(0.55),
              desc, size=11.5, cor=INK, espaco=1.12)
        if i < len(etapas) - 1:
            texto(s, x + larg + Inches(0.04), y + Inches(0.4), Inches(0.26), Inches(0.3),
                  "→", size=17, bold=True, cor=CLARO, align=PP_ALIGN.CENTER)

    # --- blocos de garantia ---
    y2 = Inches(3.62)
    blocos = [
        ("SEM VAZAMENTO", "As 24 cenas sintéticas são divididas por CENA (uma cena inteira "
         "vai para um conjunto só) e o vídeo real é dividido no TEMPO (60/20/20).", AZUL),
        ("RESIDUAL SOBRE A BASE", "Todos os modelos preveem a correção sobre o movimento "
         "constante — a mesma ideia do \"modo residual\" da Etapa 3.", RGBColor(0x7C, 0x3A, 0xED)),
        ("TREINO SEM OVERFIT", "Loss Huber + early stopping: o melhor ponto cai na época "
         "8–17 e a validação só piora depois disso.", LARANJA),
        ("AJUSTE HONESTO", "Os parâmetros do EKF foram calibrados no conjunto de "
         "VALIDAÇÃO; o teste foi usado só para a comparação final.", VERDE),
    ]
    larg2 = Inches(5.95)
    for i, (tit, desc, cor) in enumerate(blocos):
        x = MARGEM + (i % 2) * (larg2 + Inches(0.22))
        yy = y2 + (i // 2) * Inches(1.42)
        retangulo(s, x, yy, larg2, Inches(1.24), cor=CARD, borda=CLARO, raio=0.05)
        retangulo(s, x, yy, Inches(0.055), Inches(1.24), cor=cor)
        texto(s, x + Inches(0.26), yy + Inches(0.16), larg2 - Inches(0.5), Inches(0.28),
              tit, size=11, bold=True, cor=cor)
        texto(s, x + Inches(0.26), yy + Inches(0.48), larg2 - Inches(0.5), Inches(0.55),
              desc, size=11.5, cor=SUAVE, espaco=1.18)

    rodape(s, "A entrada é sempre a pose OBSERVADA (com a máscara de visibilidade por "
              "articulação); a saída é a pose futura das 13 articulações.")
    return s


def slide_resultados(prs, geral, pasta_graficos):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    retangulo(s, Inches(0), Inches(0), Inches(0.22), ALTURA, cor=AZUL)
    titulo_slide(s, "O resultado: quem acerta mais", "Erro médio da pose (MAE) no conjunto "
                 "de teste, em alturas do corpo — 1,0 = a pessoa inteira de lado.", "03")

    # --- gráfico ranking ---
    g = Path(pasta_graficos) / "erro_ranking.png"
    if g.exists():
        s.shapes.add_picture(str(g), MARGEM, Inches(1.85), height=Inches(3.35))

    # --- tabela ---
    tx = Inches(7.28)
    ty = Inches(1.85)
    larg = Inches(5.43)
    larg_col = Inches(1.68)
    larg_num = Inches(1.25)
    alt_linha = Inches(0.42)
    texto(s, tx, ty, larg, Inches(0.3), "ERRO POR HORIZONTE DE PREVISÃO", size=10.5,
          bold=True, cor=SUAVE)
    hy = ty + Inches(0.36)
    retangulo(s, tx, hy, larg, Inches(0.38), cor=RGBColor(0xF1, 0xF5, 0xF9))
    texto(s, tx + Inches(0.12), hy + Inches(0.09), larg_col, Inches(0.24), "Modelo",
          size=11, bold=True, cor=SUAVE)
    for j, nome in enumerate(("0,5 s", "1,0 s", "2,0 s")):
        x = tx + larg_col + j * larg_num
        texto(s, x, hy + Inches(0.09), larg_num, Inches(0.24), nome, size=11, bold=True,
              cor=SUAVE, align=PP_ALIGN.CENTER)
    y = hy + Inches(0.42)
    for i, m in enumerate(ORDEM):
        if m not in geral:
            continue
        l = geral[m]
        melhor = m == "lstm"
        if melhor:
            retangulo(s, tx, y, larg, alt_linha, cor=RGBColor(0xEF, 0xF6, 0xFF))
        x = tx + Inches(0.12)
        retangulo(s, x, y + Inches(0.13), Inches(0.13), Inches(0.13), cor=CORES_MODELO[m],
                   raio=0.5)
        texto(s, x + Inches(0.24), y + Inches(0.1), larg_col - Inches(0.3), Inches(0.24),
              NOMES[m], size=11.5, bold=melhor, cor=INK)
        for j, h in enumerate(("0_5s", "1_0s", "2_0s")):
            xx = tx + larg_col + j * larg_num
            texto(s, xx, y + Inches(0.1), larg_num, Inches(0.24), f"{float(l[f'mae_{h}']):.3f}",
                  size=11.5, bold=melhor, cor=AZUL if melhor else INK, align=PP_ALIGN.CENTER)
        y += alt_linha
        if i < len(ORDEM) - 1:
            linha(s, tx, y - Inches(0.02), larg, cor=RGBColor(0xF1, 0xF5, 0xF9))

    # --- destaque abaixo ---
    y3 = Inches(5.5)
    lstm = geral["lstm"]
    base = geral["base_movimento_constante"]
    gan0 = (float(base["mae_0_5s"]) - float(lstm["mae_0_5s"])) / float(base["mae_0_5s"]) * 100
    gan2 = (float(base["mae_2_0s"]) - float(lstm["mae_2_0s"])) / float(base["mae_2_0s"]) * 100
    tr = geral["transformer"]
    rf = geral["rf"]
    itens = [
        (f"−{gan0:.0f}%", "LSTM vs. movimento constante\na 0,5 s", VERDE),
        (f"{float(lstm['mae_2_0s']):.3f}", "melhor erro a 2,0 s\n(Transformer: "
         f"{float(tr['mae_2_0s']):.3f})", AZUL),
        (f"{float(rf['mae_0_5s']) / float(lstm['mae_0_5s']):.1f}×", "erro do Random Forest\n"
         "em relação ao LSTM", VERMELHO),
    ]
    larg3 = Inches(3.9)
    for i, (num, rot, cor) in enumerate(itens):
        x = MARGEM + i * (larg3 + Inches(0.24))
        retangulo(s, x, y3, larg3, Inches(1.05), cor=CARD, borda=CLARO, raio=0.06)
        texto(s, x + Inches(0.24), y3 + Inches(0.14), Inches(1.25), Inches(0.5), num,
              size=24, bold=True, cor=cor, fonte=FONTE_TITULO)
        texto(s, x + Inches(1.55), y3 + Inches(0.22), larg3 - Inches(1.75), Inches(0.7),
              rot, size=11, cor=SUAVE, espaco=1.15)

    rodape(s, "Base de movimento constante (0,149) e EKF (0,166) também aparecem: o EKF "
              "não é treinado e ainda assim se aproxima das redes.")
    return s


def slide_oclusao(prs, geral, pasta_graficos):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    retangulo(s, Inches(0), Inches(0), Inches(0.22), ALTURA, cor=LARANJA)
    titulo_slide(s, "O caso difícil: a pessoa está escondida", "O erro sobe ~20% para todos os "
                 "modelos — prever a pose oculta é o problema de verdade.", "04")

    g = Path(pasta_graficos) / "erro_oclusao.png"
    if g.exists():
        s.shapes.add_picture(str(g), MARGEM, Inches(1.9), height=Inches(3.05))

    g2 = Path(pasta_graficos) / "erro_curva.png"
    if g2.exists():
        s.shapes.add_picture(str(g2), Inches(7.05), Inches(1.9), height=Inches(3.05))

    # --- 3 conclusões ---
    y = Inches(5.25)
    itens = [
        ("O erro sobe para todo mundo", "Sob oclusão, o MAE a 2 s piora ~20% (0,30 → 0,38 "
         "no LSTM). A dificuldade é comum, não culpa do modelo.", LARANJA),
        ("O Random Forest desaba", "Ele passa de 0,41 para 1,10: sem sequência e sem forma "
         "do corpo, ele não tem como reconstruir o que não vê.", VERMELHO),
        ("O EKF sofre mais no longo prazo", "Sob oclusão a 2 s ele é o pior entre os "
         "modelos: a hipótese de velocidade constante é a que mais pesa quando a "
         "pessoa some.", RGBColor(0x64, 0x74, 0x8B)),
    ]
    larg = Inches(3.9)
    for i, (tit, desc, cor) in enumerate(itens):
        x = MARGEM + i * (larg + Inches(0.24))
        retangulo(s, x, y, larg, Inches(1.32), cor=CARD, borda=CLARO, raio=0.05)
        retangulo(s, x, y, larg, Inches(0.05), cor=cor)
        texto(s, x + Inches(0.24), y + Inches(0.2), larg - Inches(0.45), Inches(0.3),
              tit, size=12.5, bold=True, cor=INK)
        texto(s, x + Inches(0.24), y + Inches(0.56), larg - Inches(0.45), Inches(0.72),
              desc, size=10.5, cor=SUAVE, espaco=1.16)

    rodape(s, "A verdade absoluta da pose escondida existe só no vídeo sintético — por isso "
              "essa é a métrica que mais vale no projeto.")
    return s


def slide_final(prs, geral):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    retangulo(s, Inches(0), Inches(0), Inches(0.22), ALTURA, cor=VERDE)
    titulo_slide(s, "O que fica", "Três conclusões e três limites honestos.", "05")

    y = Inches(1.82)
    texto(s, MARGEM, y, Inches(5.6), Inches(0.3), "CONCLUSÕES", size=11, bold=True, cor=VERDE)
    marcadores = [
        ("LSTM e Transformer são o melhor par", "Diferença de ~3% entre eles. Com ~1,3 mil "
         "amostras, aumentar o modelo não resolve — falta dado.", VERDE),
        ("O EKF é a melhor escolha se não há treino", "Kalman estendido com modelo de corpo "
         "chega a 0,166 a 0,5 s sem nenhum rótulo e 113 mil parâmetros.", AZUL),
        ("Sequência vence features manuais", "O RF da Etapa 3 erra 2,7× mais: prever a "
         "forma do corpo exige memória temporal.", RGBColor(0x7C, 0x3A, 0xED)),
    ]
    yy = y + Inches(0.44)
    for tit, desc, cor in marcadores:
        marcador(s, MARGEM, yy, cor, tit, desc, w=Inches(5.6), size_t=14, size_d=11.5,
                 altura_titulo=0.3)
        yy += Inches(1.18)

    x2 = Inches(6.85)
    texto(s, x2, y, Inches(5.85), Inches(0.3), "LIMITES", size=11, bold=True, cor=VERMELHO)
    limites = [
        ("O vídeo real tem 23 segundos", "Dá 84 janelas no total: serve para validar o "
         "domínio, não para treinar do zero.", VERMELHO),
        ("Sem GT sob oclusão no vídeo real", "A verdade da pose escondida só existe no "
         "sintético — a oclusão real não tem ground truth.", VERMELHO),
        ("Câmera fixa, uma pessoa", "Nada de múltiplas pessoas, nada de câmera em movimento: o modelo "
         "não foi testado nesses cenários.", VERMELHO),
    ]
    yy = y + Inches(0.44)
    for tit, desc, cor in limites:
        marcador(s, x2, yy, cor, tit, desc, w=Inches(5.85), size_t=14, size_d=11.5,
                 altura_titulo=0.3)
        yy += Inches(1.18)

    # rodapé de reprodutibilidade
    y3 = Inches(5.52)
    retangulo(s, MARGEM, y3, Inches(12.1), Inches(1.16), cor=CARD, borda=CLARO, raio=0.05)
    retangulo(s, MARGEM, y3, Inches(0.055), Inches(1.16), cor=AZUL)
    texto(s, MARGEM + Inches(0.3), y3 + Inches(0.16), Inches(4), Inches(0.3),
          "REPRODUZIR", size=11, bold=True, cor=AZUL)
    texto(s, MARGEM + Inches(0.3), y3 + Inches(0.48), Inches(11.4), Inches(0.62),
          "python gerar_dados_sinteticos.py  →  python extrair_pose.py  →  "
          "python montar_dataset.py  →  python treinar_modelos_seq.py  →  "
          "python avaliar_modelos_seq.py  →  python gerar_relatorio.py",
          size=11.5, cor=INK, espaco=1.25)

    rodape(s, "Branch feat/comparacao-temporal · relatório completo em "
              "resultados/RELATORIO_COMPARACAO.md")
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resultados", default="resultados")
    p.add_argument("--graficos", default="resultados/slide")
    p.add_argument("--saida", default="resultados/apresentacao_modelos.pptx")
    args = p.parse_args()

    geral, dom, treino = ler(args.resultados)
    prs = Presentation()
    prs.slide_width = LARGURA
    prs.slide_height = ALTURA

    slide_capa(prs, geral, treino)
    slide_protocolo(prs, general=geral) if False else slide_protocolo(prs, geral)
    slide_resultados(prs, geral, args.graficos)
    slide_oclusao(prs, geral, args.graficos)
    slide_final(prs, geral)

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(saida))
    print(f"Apresentação salva em {saida} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
