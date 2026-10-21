# Inteligência Artificial 2 — Geração de Exemplos de Visão Computacional

Pipeline híbrido de visão computacional: uma **CNN pré-treinada (ResNet18)** extrai
features das imagens e um **Random Forest** faz a classificação.

```
Imagens (CIFAR-10) → ResNet18 (sem a última camada) → vetor de 512 features → Random Forest → classe
```

Extensão (Etapa 3): acompanhar **uma pessoa em um vídeo**, sinalizar **oclusões** e **prever para
onde ela vai** (CNN detectora + Random Forest), gerando um vídeo de saída anotado com alguns
segundos extras de previsão.

## Estrutura

```
├── extrair_features.py   # Etapa 1: baixa o CIFAR-10 e extrai features com a CNN
├── treinar_rf.py         # Etapa 2: treina/avalia o Random Forest (+ baseline)
├── processar_video.py    # Etapa 3: rastreia a pessoa, trata oclusões e prevê o caminho
├── gerar_video_sintetico.py # Gera um vídeo SINTÉTICO (corredor) com verdade absoluta
├── avaliar_sintetico.py  # Compara a Etapa 3 com a verdade absoluta do vídeo sintético
├── cena_corredor.py      # Cena parametrizada do corredor (com GT da POSE: 13 articulações)
├── gerar_dados_sinteticos.py # Gera o conjunto de cenas sintéticas variadas (treino/teste)
├── extrair_pose.py       # Extrai a pose do vídeo real (MediaPipe) e salva em CSV
├── montar_dataset.py     # Monta o dataset comum (sintético + real) em janelas
├── modelos_temporais.py  # LSTM, GRU, EKF, Transformer temporal, T-GNN e RF
├── treinar_modelos_seq.py# Treina os 4 modelos e salva os pesos
├── avaliar_modelos_seq.py# Compara todos os modelos (tabela, gráficos, relatório)
├── gerar_relatorio.py    # Gera o relatório Markdown da comparação
├── visualizar_previsoes.py # Desenha as previsões sobre o vídeo real
├── requirements.txt
└── .gitignore
```

## Como reproduzir (Windows / PowerShell)

```powershell
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Baixar o dataset + extrair features (o CIFAR-10 baixa sozinho para ./data)
python extrair_features.py

# 4. Treinar e avaliar o Random Forest
python treinar_rf.py
```

Se o PowerShell bloquear a ativação do ambiente, rode antes:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## Opções úteis (Etapas 1 e 2)

```powershell
# Teste rápido (menos imagens e entrada menor)
python extrair_features.py --n-treino 1000 --n-teste 200 --tamanho 112

# Mais dados
python extrair_features.py --n-treino 10000 --n-teste 2000

# Variar hiperparâmetros do Random Forest
python treinar_rf.py --n-arvores 500 --max-profundidade 20
```

## Saídas das Etapas 1 e 2 (pasta `resultados/`)

- `relatorio.txt` — acurácia do RF, baseline e relatório por classe
- `matriz_confusao.png`
- `importancia_features.png`
- `random_forest.joblib` — modelo treinado (não versionado)

## Etapa 3 — Acompanhar uma pessoa em vídeo, interpretar oclusões e prever o caminho

> As Etapas 1 e 2 (CIFAR-10) mostram o pipeline CNN + Random Forest em classificação de imagens.
> O CIFAR-10 não tem a classe "pessoa" e usa imagens 32x32, então para vídeos reais a Etapa 3 usa
> uma CNN **detectora de pessoas pré-treinada no COCO** (Faster R-CNN) e reserva o **Random Forest**
> para prever o movimento.

```powershell
# Coloque o vídeo em ./videos (a pasta é ignorada pelo Git)
mkdir videos
python processar_video.py --video videos\meu_video.mp4
```

Na 1ª execução o `torchvision` baixa os pesos do detector (algumas dezenas de MB).

### Saídas (pasta `resultados/`)

- `video_previsao.mp4` — vídeo original anotado **+ trecho extra** no final com a previsão
- `video_previsao.csv` — estado por quadro (AGUARDANDO / RASTREANDO / OCLUSAO / PREVISAO) e posição
- `video_previsao_previsoes.csv` — as previsões em números (posição prevista a cada quadro,
  para conferir contra o ground truth)
- `video_previsao_metricas.txt` — erro da previsão (em pixels) comparado ao movimento constante
- `video_previsao_info.json` — metadados da execução (resolução, fps, quadros, detector)
- `deteccoes.csv` — caixas encontradas pela CNN (coordenadas normalizadas; reutilizável)

### Como funciona

1. **Percepção (CNN):** a cada `--passo` quadros, o Faster R-CNN encontra pessoas; o rastreio segue
   a mesma pessoa (continuidade de posição/tamanho) e ignora formas que não parecem uma pessoa.
2. **Oclusão:** se a pessoa deixa de ser detectada, ou se a caixa **encolhe de repente** (pessoa
   parcialmente escondida), o quadro é marcado como OCLUSAO e uma caixa vermelha mostra a posição
   estimada pelo modelo.
3. **Previsão (Random Forest):** a posição é o ponto dos **pés** (base da caixa). Com velocidades
   recentes (0,5 / 1 / 2 s), posição e tamanho, um RF por horizonte (1 s, 2 s, ...) prevê o
   deslocamento futuro. Ele corrige uma base de movimento constante (`--modo-rf residual`,
   `--peso-rf` define o peso da correção). O treino é *walk-forward*: em cada instante o modelo só
   conhece o passado, então a previsão dentro do vídeo (laranja) pode ser conferida com o que
   aconteceu depois. A incerteza (círculo) é a dispersão entre as árvores.
4. **Trecho extra:** depois do último quadro, o vídeo continua por `--extra-segundos`
   (padrão 4 s; use 3 a 5) com o último quadro congelado e a pessoa "fantasma" percorrendo o caminho
   previsto (laranja = Random Forest, amarelo = movimento constante). Com o vídeo de 22,8 s
   e 4 s extras, a saída tem ~26,8 s.

### Parâmetros principais

| Parâmetro | Padrão | Para que serve |
|---|---|---|
| `--extra-segundos` | 4 | Duração do trecho de previsão no fim (3 a 5 s) |
| `--detector` | mobilenet | `mobilenet` (rápido), `resnet50` (mais preciso, lento), `fundo` (sem download) ou `sintetico` (ground truth) |
| `--passo` | 3 | Detecta a cada N quadros (maior = mais rápido, menos amostras) |
| `--conf-min` | 0.5 | Confiança mínima do detector |
| `--largura` | 1280 | Largura do vídeo de saída (0 = original) |
| `--max-lacuna` | 0.6 | Segundos sem detecção que ainda são interpolados |
| `--encolhimento-min` | 0.6 | Fração da altura abaixo da qual a caixa é tratada como oclusão parcial (0 = desliga) |
| `--peso-rf` | 0.5 | Peso da correção do RF sobre o movimento constante (0 a 1) |
| `--modo-rf` | residual | `residual` (corrige a base) ou `direto` |
| `--mostrar-previsao` | 2 | Quantos s à frente desenhar a previsão durante o vídeo (0 = não desenhar) |
| `--reusar-deteccoes` | desligado | Reaproveita `deteccoes.csv` e pula a CNN (ótimo para ajustar a previsão) |

Exemplos:

```powershell
# Ajustar só a previsão, sem rodar a CNN de novo
python processar_video.py --video videos\meu_video.mp4 --reusar-deteccoes --extra-segundos 5

# Detector mais preciso (mais lento)
python processar_video.py --video videos\meu_video.mp4 --detector resnet50

# Sem baixar pesos (câmera fixa e início do vídeo SEM a pessoa)
python processar_video.py --video videos\meu_video.mp4 --detector fundo
```

### Limitações (importante)

- A previsão vem da trajetória da **própria pessoa no mesmo vídeo** (poucos segundos de dados).
  Ela deve ser vista como uma estimativa com incerteza; em vídeos curtos o ganho sobre o movimento
  constante é modesto (veja `video_previsao_metricas.txt`). O Random Forest não extrapola além do que
  viu no treino.
- Funciona para **uma pessoa** como alvo; se houver várias, segue a mais proeminente e depois a de
  posição/tamanho mais parecidos.
- A previsão é feita no plano da imagem (pixels), sem modelar a perspectiva 3D nem os obstáculos
  (por exemplo, não "sabe" que há uma árvore no caminho).
- Os rótulos do vídeo estão sem acentos porque o OpenCV não os desenha.

### Onde conseguir vídeos de teste

- **O vídeo sintético do próprio projeto** (próxima seção): não precisa baixar nada.
- **Gravar o seu**: celular fixo, ~15-20 s, uma pessoa andando e passando atrás de algum obstáculo.
- **Pexels Videos** (pexels.com/videos) e **Pixabay Videos** (pixabay.com/videos): vídeos gratuitos;
  prefira câmera fixa e uma única pessoa. Confira a licença de cada vídeo.
- **MOTChallenge** (motchallenge.net): pedestres com muita oclusão (vêm como sequências de imagens;
  converta em vídeo com `ffmpeg`).

## Exemplo sintético: um vídeo SEM filmagem real, com verdade absoluta

O vídeo real depende de uma gravação e não tem resposta certa. O projeto também traz um
**exemplo sintético**: um cenário desenhado por código (nenhuma imagem gravada) que serve
para testar a Etapa 3 sem baixar pesos da internet e **saber se o resultado está certo**.

O cenário é propositalmente diferente do vídeo real gravado: em vez da rua, a pessoa
caminha num **corredor interno de prédio** (piso em perspectiva, portas, luminárias) e
passa atrás de uma **pilastra**, que a esconde por completo e depois deixa só parte do
corpo à vista.

O que o gerador entrega:

- `videos/video_sintetico.mp4` — o vídeo (22 s, 1280x720, 25 fps);
- `videos/video_sintetico_gt.csv` — **ground truth por quadro**: a caixa realmente visível
  da pessoa, a fração que está escondida (`cobertura`) e a posição verdadeira dos pés;
- `videos/video_sintetico_meta.json` — metadados (resolução, fps, posição da pilastra, etc.).

O truque do ground truth: a trajetória é uma função analítica, então o gerador também
calcula **os segundos que ainda não existem no vídeo** (6 s à frente). Assim dá para
conferir a previsão final ("para onde a pessoa vai") contra o que realmente aconteceu.

### Como rodar

```powershell
# 1. Gera o vídeo sintético + o ground truth
python gerar_video_sintetico.py --futuro 6

# 2. Roda a Etapa 3 nele (dois modos, ambos sem baixar nada)

# 2a. detector "de verdade": subtração de fundo (o pipeline acha a pessoa sozinho)
python processar_video.py --video videos\video_sintetico.mp4 --detector fundo `
    --saida resultados\sintetico_fundo.mp4 --deteccoes resultados\sintetico_fundo_deteccoes.csv

# 2b. detector "oráculo": usa a caixa verdadeira do ground truth (isola o teste da previsão)
python processar_video.py --video videos\video_sintetico.mp4 --detector sintetico `
    --saida resultados\sintetico_previsao.mp4 --deteccoes resultados\sintetico_deteccoes.csv

# 3. Compara tudo com a verdade absoluta
python avaliar_sintetico.py --log resultados\sintetico_fundo.csv `
    --previsoes resultados\sintetico_fundo_previsoes.csv --info resultados\sintetico_fundo_info.json
```

### O que a avaliação mede

`avaliar_sintetico.py` confere quatro coisas e salva um relatório em
`resultados/sintetico_avaliacao.txt`:

1. **Percepção/rastreio** — IoU entre a caixa que o pipeline acompanhou e a caixa real;
2. **Oclusão** — os intervalos em que a pessoa estava escondida de verdade contra os
   intervalos em que o pipeline marcou `OCLUSAO` (recall/precisão);
3. **Previsão dentro do vídeo** — erro (px) da posição prevista a 1 s, 2 s, 3 s, comparando
   o Random Forest com o movimento constante;
4. **Trecho extra** — erro da previsão feita no fim do vídeo contra a trajetória real dos
   segundos seguintes (que o vídeo ainda não mostra).

Resultados obtidos neste ambiente (vídeo sintético, 22 s):

| Métrica | detector `fundo` | detector `sintetico` |
|---|---|---|
| IoU médio com a caixa real | 0.89 | 0.89 |
| Oclusão real x detectada | 10.3–12.8 s x 10.2–12.3 s (recall 0.82) | 10.3–12.8 s x 10.2–12.8 s (recall 1.00) |
| Erro da previsão a +3 s | RF 41 px / constante 43 px | RF 42 px / constante 45 px |
| Erro 3 s depois do fim do vídeo | RF 6 px / constante 28 px | RF 6 px / constante 29 px |

O que o exemplo mostra de forma clara: **o Random Forest aprende a curva do caminho**
(o movimento constante erra 28 px aos 3 s do trecho extra, o RF erra 6 px), e a detecção
de oclusão acerta bem a janela em que a pessoa está atrás da pilastra.

O que ele também expõe (limitações reais do pipeline, agora mensuráveis):

- o detector por subtração de fundo é **frágil à luminância**: ele separa a pessoa do
  cenário pelo cinza, então a roupa da pessoa precisa contrastar com o fundo (no vídeo
  sintético a camisa é escura de propósito);
- quando a pessoa reaparece *parcialmente* atrás da pilastra, a caixa encolhe e o
  pipeline pode marcar o trecho como oclusão antes de recuperar o alvo;
- a previsão final erra mais à medida que o horizonte cresce (a trajetória real é uma
  curva que o RF só "viu" por alguns segundos).

---

## Etapa 4 — Comparação de modelos temporais (pose sob oclusão de câmera)

Esta etapa pega o problema da Etapa 3 (prever para onde a pessoa vai) e o leva para a
**estimação de POSE**: em vez de prever só a caixa, cada modelo prevê as **13
articulações do corpo** (cabeça, ombros, cotovelos, pulsos, quadris, joelhos,
tornozelos) nos próximos 2 segundos, inclusive enquanto a pessoa está **escondida**.

Cinco técnicas são comparadas no **mesmo protocolo**, usando **vídeo real e vídeo
sintético** para treinar:

| Modelo | Tipo | Observação |
|---|---|---|
| **LSTM** | rede recorrente | sequência de poses |
| **GRU** | rede recorrente | sequência de poses |
| **EKF** | Filtro de Kalman Estendido | **não é treinado**: modelo de processo com medição das articulações |
| **Transformer temporal** | transformer | encoder temporal + decoder por horizonte |
| **T-GNN** | GNN temporal (torch-geometric) | o **esqueleto é o grafo**: 13 nós, arestas = ossos |
| Random Forest | baseline clássico | o modelo da Etapa 3, adaptado para a pose |
| Movimento constante | baseline | sem modelo nenhum (referência) |

### Como reproduzir (Windows / PowerShell)

```powershell
# 1. Gerar as 24 cenas sintéticas variadas (com GT absoluto da pose)
python gerar_dados_sinteticos.py --cenas 24 --saida dados\sinteticos

# 2. Extrair a pose do vídeo REAL (MediaPipe, CPU)
python extrair_pose.py --video videos\meu_video.mp4 --saida dados\pose_real.csv

# 3. Montar o dataset comum (sintético + real, janelas de 1 s -> 2 s)
python montar_dataset.py --sinteticos dados\sinteticos --real dados\pose_real.csv `
    --saida dados\dataset.npz

# 4. Treinar LSTM, GRU, Transformer e T-GNN (+ RF)
python treinar_modelos_seq.py --dataset dados\dataset.npz --saida resultados\modelos

# 5. Comparar todos (incluindo o EKF, que não precisa de treino)
python avaliar_modelos_seq.py --dataset dados\dataset.npz --modelos resultados\modelos

# 6. Gerar o relatório em Markdown
python gerar_relatorio.py

# 7. (opcional) Vídeo com as previsões de cada modelo desenhadas sobre o vídeo real
python visualizar_previsoes.py --video videos\meu_video.mp4 --modelos resultados\modelos
```

### O que é medido

1. **Erro de pose por horizonte** — MAE/RMSE a 0,5 s, 1,0 s e 2,0 s, em unidades de
   altura do corpo (comparável entre o vídeo real e o sintético) e em pixels;
2. **Erro sob oclusão** — separado entre os quadros em que a pessoa está escondida
   (verdade absoluta existe no sintético) e os quadros em que ela está à vista;
3. **Erro vs. duração da oclusão** — gráfico do erro em função de quanto tempo a
   pessoa já está escondida;
4. **Tempo de treino** e **parâmetros** de cada modelo.

### Protocolo (o que garante uma comparação justa)

- a entrada é sempre a **pose observada** (com a máscara de visibilidade por
  articulação) e a saída é a **pose futura**;
- todos os modelos aprendem/calculam o **residual sobre a base de movimento
  constante** (a mesma ideia do "modo residual" da Etapa 3);
- a divisão é feita **por cena** no sintético (uma cena inteira fica num conjunto só,
  sem vazamento) e **no tempo** no vídeo real (60% / 20% / 20%);
- os hiperparâmetros do EKF foram ajustados no conjunto de **validação**, nunca no
  de teste.

### Saídas

| Arquivo | Conteúdo |
|---|---|
| `resultados/RELATORIO_COMPARACAO.md` | relatório completo (gerado por `gerar_relatorio.py`) |
| `resultados/comparacao_modelos.csv` | tabela com todas as métricas por modelo |
| `resultados/comparacao_modelos.png` | erro de pose por horizonte |
| `resultados/erro_vs_oclusao.png` | erro em função do tempo sob oclusão |
| `resultados/avaliacao_modelos.txt` | resumo em texto |
| `resultados/video_previsoes_pose.mp4` | previsões de cada modelo sobre o vídeo real |


## Reprodutibilidade

- Seed fixa (`42`) em Python, NumPy e PyTorch.
- Subconjuntos de treino/teste sorteados com a mesma seed.
- Após instalar, congele as versões exatas com `pip freeze > requirements.txt`.
