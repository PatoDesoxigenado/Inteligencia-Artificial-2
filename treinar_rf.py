"""Etapa 2: treina um Random Forest sobre as features extraídas pela CNN e avalia.

Também treina uma Regressão Logística como baseline para comparação.
Saídas em ./resultados: matriz de confusão, relatório e modelo treinado.
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # salva figuras sem precisar de janela
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=str, default="features/cifar10_resnet18.npz")
    parser.add_argument("--n-arvores", type=int, default=200)
    parser.add_argument("--max-profundidade", type=int, default=None)
    parser.add_argument("--saida", type=str, default="resultados")
    args = parser.parse_args()

    dados = np.load(args.features, allow_pickle=False)
    X_treino, y_treino = dados["X_treino"], dados["y_treino"]
    X_teste, y_teste = dados["X_teste"], dados["y_teste"]
    classes = [str(c) for c in dados["classes"]]
    print(f"Treino: {X_treino.shape} | Teste: {X_teste.shape}")

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    # --- Random Forest ---
    print("\nTreinando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=args.n_arvores,
        max_depth=args.max_profundidade,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_treino, y_treino)
    pred_rf = rf.predict(X_teste)
    acc_rf = accuracy_score(y_teste, pred_rf)

    # --- Baseline: Regressão Logística ---
    print("Treinando baseline (Regressão Logística)...")
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_treino, y_treino)
    acc_lr = accuracy_score(y_teste, lr.predict(X_teste))

    relatorio = classification_report(y_teste, pred_rf, target_names=classes, digits=3)
    resumo = (
        f"Acurácia Random Forest (CNN + RF): {acc_rf:.3f}\n"
        f"Acurácia Regressão Logística (baseline): {acc_lr:.3f}\n\n"
        f"Relatório do Random Forest:\n{relatorio}"
    )
    print("\n" + resumo)
    (saida / "relatorio.txt").write_text(resumo, encoding="utf-8")

    # --- Matriz de confusão ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_teste, pred_rf, display_labels=classes, xticks_rotation=45, ax=ax, colorbar=False
    )
    ax.set_title(f"CNN + Random Forest (acurácia = {acc_rf:.3f})")
    fig.tight_layout()
    fig.savefig(saida / "matriz_confusao.png", dpi=150)
    plt.close(fig)

    # --- Importância das 20 features mais relevantes ---
    topo = np.argsort(rf.feature_importances_)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(20), rf.feature_importances_[topo])
    ax.set_xticks(range(20))
    ax.set_xticklabels([str(i) for i in topo], rotation=90)
    ax.set_xlabel("Índice da feature (0-511)")
    ax.set_ylabel("Importância")
    ax.set_title("Top 20 features mais importantes")
    fig.tight_layout()
    fig.savefig(saida / "importancia_features.png", dpi=150)
    plt.close(fig)

    joblib.dump(rf, saida / "random_forest.joblib")
    print(f"Arquivos salvos em ./{saida}/")


if __name__ == "__main__":
    main()
