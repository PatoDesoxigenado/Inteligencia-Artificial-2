"""Etapa 1: extrai features de imagens do CIFAR-10 com uma ResNet18 pré-treinada.

O CIFAR-10 é baixado automaticamente para a pasta ./data na primeira execução.
As features (vetores de 512 dimensões) são salvas em ./features/cifar10_resnet18.npz
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42


def fixar_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extrair(modelo, loader, device):
    """Passa as imagens pela CNN (sem gradiente) e devolve (X, y)."""
    X, y = [], []
    with torch.no_grad():
        for i, (imagens, rotulos) in enumerate(loader, start=1):
            feats = modelo(imagens.to(device)).cpu().numpy()
            X.append(feats)
            y.append(rotulos.numpy())
            if i % 10 == 0 or i == len(loader):
                print(f"  lote {i}/{len(loader)}", flush=True)
    return np.vstack(X), np.concatenate(y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-treino", type=int, default=5000, help="imagens de treino (máx. 50000)")
    parser.add_argument("--n-teste", type=int, default=1000, help="imagens de teste (máx. 10000)")
    parser.add_argument("--tamanho", type=int, default=224, help="lado da imagem na entrada da CNN (menor = mais rápido)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--saida", type=str, default="features/cifar10_resnet18.npz")
    args = parser.parse_args()

    fixar_seeds(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    tfm = transforms.Compose([
        transforms.Resize(args.tamanho),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("Carregando CIFAR-10 (baixa para ./data se necessário)...")
    treino = datasets.CIFAR10("data", train=True, download=True, transform=tfm)
    teste = datasets.CIFAR10("data", train=False, download=True, transform=tfm)

    # Subconjuntos reprodutíveis (mesma seed => mesmas imagens)
    rng = np.random.default_rng(SEED)
    idx_treino = np.sort(rng.choice(len(treino), args.n_treino, replace=False))
    idx_teste = np.sort(rng.choice(len(teste), args.n_teste, replace=False))

    loader_treino = DataLoader(Subset(treino, idx_treino), batch_size=args.batch_size, num_workers=0)
    loader_teste = DataLoader(Subset(teste, idx_teste), batch_size=args.batch_size, num_workers=0)

    # ResNet18 pré-treinada no ImageNet, sem a última camada (vira extrator de features)
    cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    cnn.fc = torch.nn.Identity()
    cnn.eval().to(device)

    print("Extraindo features de treino...")
    X_treino, y_treino = extrair(cnn, loader_treino, device)
    print("Extraindo features de teste...")
    X_teste, y_teste = extrair(cnn, loader_teste, device)

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        saida,
        X_treino=X_treino, y_treino=y_treino,
        X_teste=X_teste, y_teste=y_teste,
        classes=np.array(treino.classes),
    )
    print(f"Pronto! Salvo em {saida} | treino: {X_treino.shape} | teste: {X_teste.shape}")


if __name__ == "__main__":
    main()
