# Inteligência Artificial 2 — Geração de Exemplos de Visão Computacional

Pipeline híbrido de visão computacional: uma **CNN pré-treinada (ResNet18)** extrai
features das imagens e um **Random Forest** faz a classificação.

```
Imagens (CIFAR-10) → ResNet18 (sem a última camada) → vetor de 512 features → Random Forest → classe
```

## Estrutura

```
├── extrair_features.py   # Etapa 1: baixa o CIFAR-10 e extrai features com a CNN
├── treinar_rf.py         # Etapa 2: treina/avalia o Random Forest (+ baseline)
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

## Opções úteis

```powershell
# Teste rápido (menos imagens e entrada menor)
python extrair_features.py --n-treino 1000 --n-teste 200 --tamanho 112

# Mais dados
python extrair_features.py --n-treino 10000 --n-teste 2000

# Variar hiperparâmetros do Random Forest
python treinar_rf.py --n-arvores 500 --max-profundidade 20
```

## Saídas (pasta `resultados/`)

- `relatorio.txt` — acurácia do RF, baseline e relatório por classe
- `matriz_confusao.png`
- `importancia_features.png`
- `random_forest.joblib` — modelo treinado (não versionado)

## Reprodutibilidade

- Seed fixa (`42`) em Python, NumPy e PyTorch.
- Subconjuntos de treino/teste sorteados com a mesma seed.
- Após instalar, congele as versões exatas com `pip freeze > requirements.txt`.
