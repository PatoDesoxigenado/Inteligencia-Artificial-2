"""Treina os modelos temporais (LSTM, GRU, Transformer, T-GNN) no dataset montado.

Protocolo (idêntico para todos os modelos):
  - entrada : T quadros passados da pose observada (x, y, visível, presente)
  - saída   : H quadros futuros; a rede prevê o RESIDUAL sobre a base de movimento
              constante (a mesma base já calculada no dataset)
  - perda  : MSE com máscara de supervisão (só os alvos confiáveis entram)
  - o EKF NÃO é treinado (é modelo de processo) e o RF é treinado como baseline

O treino usa o conjunto TREINO (cenas sintéticas + trecho inicial do vídeo real) e a
validação usa o conjunto de VALIDAÇÃO, sempre sem olhar o futuro do vídeo.

Uso:
  python treinar_modelos_seq.py --dataset dados/dataset.npz --saida resultados/modelos
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from modelos_temporais import (EKF, GRUPose, LSTMPose, N_ARTICULACOES, RFPose,
                                TGNNPose, TransformerPose)


def carregar(caminho):
    d = np.load(caminho, allow_pickle=True)
    grupos = {}
    for g in ("treino", "validacao", "teste"):
        if f"{g}_entrada" in d:
            grupos[g] = {k: d[f"{g}_{k}"] for k in
                         ("entrada", "alvo", "sup", "vis", "ocl", "base", "t0", "presente_ini")}
    cfg = json.loads(str(d["config"])) if "config" in d else {}
    return grupos, cfg


def perda_mascarada(residuo, alvo, base, mascara):
    """MSE com máscara: só os alvos confiáveis (sup=1) entram na perda."""
    dif = (residuo + base - alvo) ** 2
    m = mascara[..., None]
    return (dif * m).sum() / np.maximum(m.sum() * 2, 1.0)


def treinar_rede(nome, modelo, dados_tr, dados_va, epochs, lr, batch, device, seed,
                 verbose=True, paciencia=12):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    modelo.to(device)

    ent = torch.tensor(dados_tr["entrada"], dtype=torch.float32)
    res_alvo = torch.tensor(dados_tr["alvo"] - dados_tr["base"], dtype=torch.float32)
    sup = torch.tensor(dados_tr["sup"], dtype=torch.float32)

    ent_va = torch.tensor(dados_va["entrada"], dtype=torch.float32)
    alvo_va = torch.tensor(dados_va["alvo"], dtype=torch.float32)
    base_va = torch.tensor(dados_va["base"], dtype=torch.float32)
    sup_va = torch.tensor(dados_va["sup"], dtype=torch.float32)

    # LOSS HUBER (suavemente L1): o MSE puro é dominado por poucas amostras ruins
    # (no vídeo real, a pessoa às vezes fica pequena na imagem e a base de movimento
    # constante erra muito) e o gradiente passa a atender só elas. Com o Huber, os
    # outliers contribuem de forma limitada e o modelo ainda aprende o caso comum.
    BETA = 0.5   # em unidades de altura do corpo

    def perda_huber(residuo, alvo_residuo, mascara):
        dif = (residuo - alvo_residuo).abs()
        quad = torch.clamp(dif, max=BETA)
        lin = dif - quad
        per = 0.5 * quad ** 2 + BETA * lin
        m = mascara[..., None]
        return (per * m).sum() / (m.sum() * 2 + 1e-8)

    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    n = len(ent)
    passos = max(1, n // batch)
    # O OneCycleLR precisa saber QUANTAS épocas realmente vão rodar: se o schedule
    # foi montado para 60 épocas mas o treino para na 16, a taxa de aprendizado nunca
    # "anneal" (cai) e a validação não converge. Por isso o schedule usa o mesmo
    # número de épocas do treino.
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * passos,
                                                pct_start=0.3)
    historico = []
    melhor = (float("inf"), None, -1)
    sem_melhoria = 0

    for ep in range(epochs):
        modelo.train()
        perm = torch.randperm(n)
        tot = 0.0
        for p in range(passos):
            idx = perm[p * batch:(p + 1) * batch]
            opt.zero_grad()
            r = modelo.forward(ent[idx].to(device))
            perda = perda_huber(r, res_alvo[idx].to(device), sup[idx].to(device))
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(perda)
        tot /= passos

        modelo.eval()
        with torch.no_grad():
            r = modelo.forward(ent_va.to(device)).cpu()
            pv = float(perda_huber(r, alvo_va - base_va, sup_va))
        historico.append({"epoca": ep + 1, "perda_treino": tot, "perda_validacao": pv})
        if pv < melhor[0] - 1e-6:
            melhor = (pv, {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}, ep + 1)
            sem_melhoria = 0
        else:
            sem_melhoria += 1
        if verbose:
            marca = " *" if sem_melhoria == 0 else ""
            print(f"    época {ep + 1:3d}/{epochs}  treino {tot:.5f}  validação {pv:.5f}{marca}",
                  flush=True)

        # EARLY STOPPING: com poucos dados, treinar demais faz a validação PIORAR
        # (overfitting). Se a validação não melhora há 'paciencia' épocas, o treino
        # para e os melhores pesos são restaurados.
        if sem_melhoria >= paciencia:
            if verbose:
                print(f"    early stopping na época {ep + 1} "
                      f"(sem melhora há {paciencia} épocas)", flush=True)
            break

    modelo.load_state_dict(melhor[1])
    return modelo, historico, melhor[0], melhor[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dados/dataset.npz")
    p.add_argument("--saida", default="resultados/modelos")
    p.add_argument("--epochs", type=int, default=35,
                   help="número de épocas (o OneCycleLR é montado para este número, "
                        "então a taxa de aprendizado realmente termina no valor mínimo; "
                        "com ~1,3 mil amostras o melhor ponto costuma ser na época 15-25)")
    p.add_argument("--paciencia", type=int, default=12,
                   help="épocas sem melhora na validação para interromper o treino "
                        "(rede de segurança; o corte normal é o fim do schedule)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--semente", type=int, default=7)
    p.add_argument("--modelos", default="lstm,gru,transformer,tgnn",
                   help="quais treinar: lstm,gru,transformer,tgnn")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    grupos, cfg = carregar(args.dataset)
    T, H = cfg.get("T", 25), cfg.get("H", 50)
    tr, va = grupos["treino"], grupos["validacao"]
    print(f"Treino: {len(tr['entrada'])} amostras | Validação: {len(va['entrada'])} amostras "
          f"| T={T} H={H}")

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    salvos = {}

    fabricas = {
        "lstm": lambda: LSTMPose(T, H),
        "gru": lambda: GRUPose(T, H),
        "transformer": lambda: TransformerPose(T, H),
        "tgnn": lambda: TGNNPose(T, H),
    }
    for nome in [m.strip() for m in args.modelos.split(",") if m.strip()]:
        print(f"\nTreinando {nome}...")
        t0 = time.time()
        modelo, hist, melhor, ep_melhor = treinar_rede(
            nome, fabricas[nome](), tr, va, args.epochs, args.lr, args.batch, device,
            args.semente, paciencia=args.paciencia)
        torch.save({"estado": modelo.state_dict(), "T": T, "H": H}, saida / f"{nome}.pt")
        salvos[nome] = {"historico": hist, "melhor_validacao": melhor,
                        "melhor_epoca": ep_melhor, "epocas_treino": len(hist),
                        "segundos": time.time() - t0,
                        "parametros": sum(p.numel() for p in modelo.parameters())}
        print(f"  -> melhor validação {melhor:.5f} (época {ep_melhor}/{len(hist)}) "
              f"em {time.time() - t0:.0f} s")

    # ---- baseline RF (mesmo espaço de pose) ----
    print("\nTreinando RF (baseline clássico)...")
    rf = RFPose(H)
    rf.treinar(tr["entrada"], tr["alvo"], tr["base"], tr["sup"])
    import joblib
    joblib.dump(rf, saida / "rf.joblib")
    salvos["rf"] = {"historico": [], "melhor_validacao": None, "parametros": 0,
                    "segundos": 0.0}
    print(f"  -> {len(rf.modelos)} horizontes treinados")

    (saida / "treinamento.json").write_text(json.dumps(salvos, indent=2), encoding="utf-8")
    print(f"\nModelos salvos em {saida}")


if __name__ == "__main__":
    main()
