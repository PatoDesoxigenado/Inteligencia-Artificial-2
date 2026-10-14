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
- `video_previsao_metricas.txt` — erro da previsão (em pixels) comparado ao movimento constante
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
| `--detector` | mobilenet | `mobilenet` (rápido), `resnet50` (mais preciso, lento) ou `fundo` (sem download) |
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

- **Gravar o seu**: celular fixo, ~15-20 s, uma pessoa andando e passando atrás de algum obstáculo.
- **Pexels Videos** (pexels.com/videos) e **Pixabay Videos** (pixabay.com/videos): vídeos gratuitos;
  prefira câmera fixa e uma única pessoa. Confira a licença de cada vídeo.
- **MOTChallenge** (motchallenge.net): pedestres com muita oclusão (vêm como sequências de imagens;
  converta em vídeo com `ffmpeg`).

## Reprodutibilidade

- Seed fixa (`42`) em Python, NumPy e PyTorch.
- Subconjuntos de treino/teste sorteados com a mesma seed.
- Após instalar, congele as versões exatas com `pip freeze > requirements.txt`.
