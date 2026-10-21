"""Modelos temporais para prever a POSE de uma pessoa sob oclusão de câmera.

Todos os modelos compartilham a MESMA interface, para a comparação ser justa:

    entrada : (B, T, J, 4)  -> pose observada normalizada (x, y, visível, pessoa_presente)
    saída   : (B, H, J, 2)  -> pose futura prevista (mesmo referencial da entrada)

Modelos implementados
---------------------
  EKF          Filtro de Kalman Estendido (modelo de processo, sem treino): estado
               [x, y, vx, vy, log_h] do alvo, com atualização por medição quando a
               pessoa está visível e apenas previsão durante a oclusão.
  LSTM         LSTM sobre a sequência de poses.
  GRU          GRU sobre a sequência de poses.
  Transformer  Transformer temporal (encoder + decoder) sobre a sequência de poses.
  T-GNN        Temporal GNN (torch-geometric): o ESQUELETO é o grafo (13 nós, arestas
               = ossos) e o tempo entra como dimensão de sequência; a propagação
               espacial (GNN) e a temporal (GRU por nó) se combinam.

Baseline extra
--------------
  RF           Random Forest por horizonte (o modelo do pipeline da Etapa 3, adaptado
               para a pose), para comparar com a abordagem "clássica" já validada.

Observação importante: em todos os modelos a saída é um RESIDUAL sobre a base de
movimento constante (a mesma ideia do "modo residual" do processar_video.py): o modelo
aprende apenas a CORREÇÃO da base, o que estabiliza o treino com poucos dados.
"""
import math

import numpy as np

# ----------------------------------------------------------------------------------
# Configuração comum
# ----------------------------------------------------------------------------------
N_ARTICULACOES = 13
# arestas do esqueleto (índices canônicos) usadas pelo T-GNN
ARESTAS_ESQUELETO = [
    (0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 9), (9, 11), (8, 10), (10, 12),
    (7, 8),
]


# ----------------------------------------------------------------------------------
# EKF — Filtro de Kalman Estendido (não precisa de treino)
# ----------------------------------------------------------------------------------
class EKF:
    """Filtro de Kalman ESTENDIDO para a pose de uma pessoa (modelo de processo).

    Estado (6 dimensões):
        x = [tx, ty, vx, vy, log_s, w]
        tx, ty  -> posição do tronco (centro do corpo) em pixels
        vx, vy  -> velocidade do tronco (px/quadro)
        log_s   -> log da ESCALA do corpo (altura cabeça->quadril, em px)
        w       -> taxa de variação da escala (a pessoa pode se aproximar da câmera)

    O FILTER É EXTENDIDO porque o modelo de MEDIÇÃO é não linear: a medição de cada
    articulação visível j é

        z_j = t + s * o_j ,     s = exp(log_s)

    e o Jacobiano H depende de s. O estado da câmera é integrado ao modelo do corpo,
    então a "forma" (os offsets o_j de cada articulação em relação ao tronco) é
    memorizada pelo filtro a partir das articulações que ele viu.

    Quando a pessoa está escondida não há medição: o filtro só PREDIZ, e a incerteza
    cresce (é o comportamento clássico do Kalman Filter sob oclusão).
    """

    # Parâmetros calibrados no conjunto de VALIDAÇÃO (busca em grade sobre
    # q_pos, q_vel, r_pos e p_vel0). O conjunto de teste não foi usado para ajustar.
    def __init__(self, q_pos=1e-2, q_vel=5e-2, q_escala=1e-2, r_pos=16.0, r_escala=4.0,
                 vel_max=8.0, damp=1.0, p_vel0=4.0):
        self.q_pos, self.q_vel, self.q_escala = q_pos, q_vel, q_escala
        self.r_pos, self.r_escala = r_pos, r_escala
        self.vel_max, self.damp, self.p_vel0 = vel_max, damp, p_vel0
        self.reset()

    def reset(self):
        self.x = None
        self.P = np.eye(6) * 10.0
        self.o = np.zeros((N_ARTICULACOES, 2), np.float32)   # forma (offsets) do corpo
        self.o_ok = np.zeros(N_ARTICULACOES, bool)
        self.o_media = np.zeros((N_ARTICULACOES, 2), np.float32)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _tronco(pose, mask):
        m = np.asarray(mask) > 0
        idx = [i for i in (1, 2, 7, 8) if m[i]]
        if not idx:
            return None
        return pose[idx].mean(axis=0).astype(np.float64)

    def _forma_inicial(self, pose, mask):
        """Estima tronco, escala e offsets a partir da primeira visão boa."""
        t = self._tronco(pose, mask)
        if t is None:
            return None
        m = np.asarray(mask) > 0
        quadris = [i for i in (7, 8) if m[i]]
        if not quadris:
            return None
        cquad = pose[quadris].mean(axis=0)
        if m[0]:
            s = float(np.linalg.norm(pose[0] - cquad))
        else:
            vis = [i for i in range(N_ARTICULACOES) if m[i]]
            if not vis:
                return None
            s = float(np.mean([np.linalg.norm(pose[i] - t) for i in vis])) * 1.6
        if s < 1e-3:
            return None
        o = (pose - t) / s
        return t, s, o.astype(np.float32)

    # ---------------------------------------------------------------- filtro
    def passo(self, pose=None, mask=None, dt=1.0):
        """Processa um quadro: atualiza com a medição (se houver) ou só prevê."""
        if pose is None or mask is None:
            if self.x is not None:
                self._predict(dt)
            return

        m = np.asarray(mask) > 0
        if self.x is None:
            info = self._forma_inicial(pose, mask)
            if info is None:
                return
            t, s, o = info
            self.x = np.array([t[0], t[1], 0.0, 0.0, math.log(s), 0.0])
            self.P = np.diag([self.r_pos, self.r_pos, self.p_vel0, self.p_vel0, 0.05, 0.01])
            self.o = o
            self.o_ok = m.copy()
            self.o_media = o.copy()
            return

        self._predict(dt)
        self._update(pose, mask)

    def _predict(self, dt):
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 4] = self.damp
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.diag(
            [self.q_pos, self.q_pos, self.q_vel, self.q_vel, self.q_escala, 1e-3]) * dt
        v = self.x[2:4]
        nv = float(np.hypot(v[0], v[1]))
        if nv > self.vel_max:
            self.x[2:4] = v * (self.vel_max / nv)
        self.x[4] = float(np.clip(self.x[4], -6.0, 6.0))

    def _update(self, pose, mask):
        m = np.asarray(mask) > 0
        s = math.exp(float(self.x[4]))
        # medições: as articulações vistas, no modelo do corpo
        idx = [j for j in range(N_ARTICULACOES) if m[j] and self.o_ok[j]]
        if not idx:
            return
        t = self.x[0:2]
        H = np.zeros((2 * len(idx), 6))
        z = np.zeros(2 * len(idx))
        for k, j in enumerate(idx):
            o = self.o[j]
            z[2 * k] = pose[j, 0]
            z[2 * k + 1] = pose[j, 1]
            H[2 * k, 0] = 1.0
            H[2 * k, 1] = 0.0
            H[2 * k, 4] = s * o[0]
            H[2 * k + 1, 0] = 0.0
            H[2 * k + 1, 1] = 1.0
            H[2 * k + 1, 4] = s * o[1]
        y = z - (np.repeat(t[None, :], len(idx), axis=0).reshape(-1) +
                 (s * self.o[idx]).reshape(-1))
        R = np.diag([self.r_pos, self.r_pos] * len(idx))
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.x[4] = float(np.clip(self.x[4], -6.0, 6.0))

        # memoriza a forma vista (média móvel por articulação)
        t2 = self.x[0:2]
        s2 = math.exp(float(self.x[4]))
        for j in range(N_ARTICULACOES):
            if m[j]:
                novo = ((pose[j] - t2) / s2).astype(np.float32)
                if not self.o_ok[j]:
                    self.o_media[j] = novo
                else:
                    self.o_media[j] = 0.7 * self.o_media[j] + 0.3 * novo
                self.o[j] = self.o_media[j]
                self.o_ok[j] = True

    # ---------------------------------------------------------------- previsão
    def prever(self, horizonte):
        """Previsão de (tronco, escala) 'horizonte' quadros à frente."""
        if self.x is None:
            return None
        F = np.eye(6)
        F[0, 2] = horizonte
        F[1, 3] = horizonte
        F[4, 4] = self.damp ** horizonte
        xf = F @ self.x
        return float(xf[0]), float(xf[1]), math.exp(float(np.clip(xf[4], -6.0, 6.0)))

    def prever_pose(self, horizonte):
        """Pose prevista: tronco previsto + a FORMA memorizada, escalada pela
        escala prevista."""
        if self.x is None:
            return None
        prev = self.prever(horizonte)
        if prev is None:
            return None
        cx, cy, esc = prev
        out = np.zeros((N_ARTICULACOES, 2), np.float32)
        for j in range(N_ARTICULACOES):
            if not self.o_ok[j]:
                continue
            out[j] = (cx + self.o[j, 0] * esc, cy + self.o[j, 1] * esc)
        return out


def centro_tronco(pose, mask):
    """Centro do tronco (ombros + quadris visíveis): o ponto do corpo que se move
    de forma mais estável (o quadril isolado oscila a cada passada)."""
    m = np.asarray(mask) > 0
    tronco = [i for i in (1, 2, 7, 8) if m[i]]
    if not tronco:
        return None
    return pose[tronco].mean(axis=0).astype(np.float32)


def forma_relativa(pose, mask):
    """Offsets de cada articulação em relação ao centro do tronco, normalizados pela
    altura. É a "forma do corpo" usada pelo EKF para reconstruir a pose."""
    c = centro_tronco(pose, mask)
    if c is None:
        return None
    cabeca = [i for i in (0,) if mask[i] > 0]
    if not cabeca:
        return None
    altura = float(np.linalg.norm(pose[cabeca[0]] - c))
    if altura < 1e-3:
        return None
    return ((pose - c) / altura).astype(np.float32)


def forma_parcial(pose, mask):
    """Forma por articulação, com a mesma normalização de `forma_relativa`, mas
    devolvendo também quais articulações são confiáveis neste quadro."""
    f = forma_relativa(pose, mask)
    if f is None:
        return None, None
    m = np.asarray(mask) > 0
    return f, m


# ----------------------------------------------------------------------------------
# Modelos de rede (PyTorch)
# ----------------------------------------------------------------------------------
def _torch():
    import torch
    return torch


class ModeloSequencial:
    """Base comum: entrada (B, T, J, 4) -> saída (B, H, J, 2) (residual sobre a base)."""

    def __init__(self, T, H, J=N_ARTICULACOES):
        self.T, self.H, self.J = T, H, J

    # subclasses implementam `forward`

    def to(self, device):
        """Envia os parâmetros para o dispositivo (compatível com torch.nn.Module)."""
        self.rede.to(device)
        self.device = device
        return self

    def train(self, modo=True):
        self.rede.train(modo)
        return self

    def eval(self):
        self.rede.eval()
        return self

    def state_dict(self):
        return self.rede.state_dict()

    def load_state_dict(self, estado):
        self.rede.load_state_dict(estado)

    def parameters(self):
        return self.rede.parameters()
    def prever(self, entrada, base):
        import torch
        with torch.no_grad():
            resid = self.forward(entrada)
            return (base + resid).numpy()


class LSTMPose(ModeloSequencial):
    """LSTM sobre a sequência de poses (quadro a quadro, com as 13 articulações
    achatadas e a máscara) + cabeça que produz o residual da pose futura."""

    def __init__(self, T, H, J=N_ARTICULACOES, oculto=192, camadas=2, dropout=0.15):
        super().__init__(T, H, J)
        torch = _torch()
        self.rede = torch.nn.Module()
        entrada_dim = J * 4
        self.rede.lstm = torch.nn.LSTM(entrada_dim, oculto, camadas, batch_first=True,
                                       dropout=dropout if camadas > 1 else 0.0)
        self.rede.cabeca = torch.nn.Sequential(
            torch.nn.Linear(oculto, oculto),
            torch.nn.LayerNorm(oculto),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(oculto, H * J * 2),
        )

    def forward(self, entrada):
        torch = _torch()
        B, T, J, _ = entrada.shape
        x = entrada.reshape(B, T, J * 4)
        out, _ = self.rede.lstm(x)
        z = out[:, -1]                                  # último estado
        r = self.rede.cabeca(z).view(B, self.H, J, 2)
        return r


class GRUPose(ModeloSequencial):
    """GRU sobre a sequência de poses."""

    def __init__(self, T, H, J=N_ARTICULACOES, oculto=192, camadas=2, dropout=0.15):
        super().__init__(T, H, J)
        torch = _torch()
        self.rede = torch.nn.Module()
        self.rede.gru = torch.nn.GRU(J * 4, oculto, camadas, batch_first=True,
                                     dropout=dropout if camadas > 1 else 0.0)
        self.rede.cabeca = torch.nn.Sequential(
            torch.nn.Linear(oculto, oculto),
            torch.nn.LayerNorm(oculto),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(oculto, H * J * 2),
        )

    def forward(self, entrada):
        torch = _torch()
        B, T, J, _ = entrada.shape
        x = entrada.reshape(B, T, J * 4)
        out, _ = self.rede.gru(x)
        z = out[:, -1]
        r = self.rede.cabeca(z).view(B, self.H, J, 2)
        return r


class TransformerPose(ModeloSequencial):
    """Transformer temporal: encoder sobre os quadros (com codificação posicional) e
    decoder que consulta a memória para cada horizonte futuro."""

    def __init__(self, T, H, J=N_ARTICULACOES, d=128, cabeças=4, blocos=3, ff=256, dropout=0.15):
        super().__init__(T, H, J)
        torch = _torch()
        self.rede = torch.nn.Module()
        self.rede.proj = torch.nn.Linear(J * 4, d)
        self.rede.pos = torch.nn.Parameter(torch.zeros(1, T, d))
        torch.nn.init.normal_(self.rede.pos, std=0.02)
        camada_enc = torch.nn.TransformerEncoderLayer(d, cabeças, ff, dropout, batch_first=True,
                                                       norm_first=True, activation="gelu")
        self.rede.encoder = torch.nn.TransformerEncoder(camada_enc, blocos)
        # consultas de saída (um token por horizonte futuro)
        self.rede.consultas = torch.nn.Parameter(torch.randn(1, H, d) * 0.02)
        self.rede.dec = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(d, cabeças, ff, dropout, batch_first=True,
                                             norm_first=True, activation="gelu"), 2)
        self.rede.cabeca = torch.nn.Sequential(
            torch.nn.Linear(d, ff), torch.nn.LayerNorm(ff), torch.nn.GELU(),
            torch.nn.Linear(ff, J * 2))

    def forward(self, entrada):
        torch = _torch()
        B, T, J, _ = entrada.shape
        x = self.rede.proj(entrada.reshape(B, T, J * 4)) + self.rede.pos[:, :T]
        memoria = self.rede.encoder(x)
        q = self.rede.consultas.expand(B, -1, -1)
        y = self.rede.dec(q, memoria)
        r = self.rede.cabeca(y).view(B, self.H, J, 2)
        return r


class TGNNPose(ModeloSequencial):
    """T-GNN (Temporal Graph Neural Network) com torch-geometric.

    O grafo é o ESQUELETO: 13 nós (articulações) e arestas = ossos. Para cada quadro
    da entrada, os nós recebem as features da articulação e a informação temporal
    (a pose dos quadros anteriores é processada por uma GRU por nó, compartilhada).
    A propagação espacial (GNN sobre os ossos) e a temporal (GRU) se combinam antes da
    cabeça que prevê a pose futura.
    """

    def __init__(self, T, H, J=N_ARTICULACOES, d=96, blocos=2, dropout=0.15):
        super().__init__(T, H, J)
        torch = _torch()
        from torch_geometric.nn import GCNConv

        class _Bloco(torch.nn.Module):
            def __init__(self, d, dropout):
                super().__init__()
                self.conv1 = GCNConv(d, d)
                self.conv2 = GCNConv(d, d)
                self.norm1 = torch.nn.LayerNorm(d)
                self.norm2 = torch.nn.LayerNorm(d)
                self.drop = torch.nn.Dropout(dropout)

            def forward(self, x, edge_index):
                x = self.norm1(x + self.drop(torch.nn.functional.relu(self.conv1(x, edge_index))))
                x = self.norm2(x + self.drop(torch.nn.functional.relu(self.conv2(x, edge_index))))
                return x

        self.rede = torch.nn.Module()
        self.rede.proj = torch.nn.Linear(4, d)             # (x, y, visível, presente)
        self.rede.gru = torch.nn.GRU(d, d, batch_first=True)  # temporal, por nó
        self.rede.blocos = torch.nn.ModuleList([_Bloco(d, dropout) for _ in range(blocos)])
        self.rede.cabeca = torch.nn.Sequential(
            torch.nn.Linear(d, d), torch.nn.LayerNorm(d), torch.nn.GELU(),
            torch.nn.Linear(d, H * 2))

        ei = []
        for a, b in ARESTAS_ESQUELETO:
            ei.append([a, b])
            ei.append([b, a])
        self.edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        self.j = J

    def forward(self, entrada):
        torch = _torch()
        from torch_geometric.utils import to_dense_adj
        B, T, J, F = entrada.shape
        d = self.rede.proj.out_features

        # 1) temporal por nó: cada articulação tem sua própria série
        x = self.rede.proj(entrada)                       # (B, T, J, d)
        x = x.permute(0, 2, 1, 3).reshape(B * J, T, d)
        x, _ = self.rede.gru(x)                           # (B*J, T, d)
        seq = x.reshape(B, J, T, d)

        # 2) espacial por quadro: GNN sobre o esqueleto, em cada quadro
        adj = to_dense_adj(self.edge_index, max_num_nodes=J)[0]   # (J, J)
        saidas = []
        for t in range(T):
            h = seq[:, :, t, :].reshape(B * J, d)
            for bloco in self.rede.blocos:
                h = bloco(h, self.edge_index)
            saidas.append(h.reshape(B, J, d))
        h = torch.stack(saidas, dim=1)                   # (B, T, J, d)

        # 3) junta o estado temporal final com o espacial do último quadro
        z = h[:, -1] + seq[:, :, -1, :]                  # (B, J, d)
        r = self.rede.cabeca(z).view(B, J, self.H, 2).permute(0, 2, 1, 3)
        return r


# ----------------------------------------------------------------------------------
# Baseline: Random Forest por horizonte (adaptado do pipeline da Etapa 3)
# ----------------------------------------------------------------------------------
class RFPose:
    """Random Forest por horizonte: prevê o residual da pose futura a partir de
    atributos de movimento (posição, velocidade, oclusão). É o "clássico" já validado
    no processo de Etapa 3, agora no espaço de pose."""

    def __init__(self, H, J=N_ARTICULACOES, horizontes=None, n_arvores=100, min_leaf=3, semente=42):
        self.H, self.J = H, J
        self.horizontes = horizontes if horizontes is not None else list(range(1, H + 1))
        self.n_arvores, self.min_leaf, self.semente = n_arvores, min_leaf, semente
        self.modelos = {}

    @staticmethod
    def _atributos(entrada):
        """entrada: (T, J, 4) -> vetor de atributos (posição/velocidade/oclusão)."""
        T, J, _ = entrada.shape
        vis = entrada[:, :, 2]
        pres = entrada[:, :, 3]
        ultimo = entrada[-1]
        quadris = entrada[-1, 7:9, :2].mean(axis=0)
        altura = float(np.linalg.norm(entrada[-1, 0, :2] - quadris)) if vis[-1].sum() >= 4 else 0.0
        f = [quadris[0], quadris[1], altura, vis[-1].mean(), pres[-1].mean()]
        # velocidade do quadril nos últimos k quadros (normalizada pela altura)
        for k in (2, 5, 10):
            if T > k:
                q0 = entrada[-k - 1, 7:9, :2].mean(axis=0)
                f += [(quadris[0] - q0[0]) / max(altura, 1e-3),
                      (quadris[1] - q0[1]) / max(altura, 1e-3)]
            else:
                f += [0.0, 0.0]
        # variabilidade da pose (a pessoa está andando?) e tempo desde a última visão
        f.append(float(vis[-1].sum()))
        ult_vis = np.where(pres > 0)[0]
        f.append(float(T - 1 - ult_vis[-1]) if len(ult_vis) else float(T))
        return np.array(f, np.float32)

    def treinar(self, entradas, alvos, bases, mascaras, val=None):
        """Um RF por horizonte: cada um aprende o residual de pose (13x2) naquele
        horizonte a partir dos atributos de movimento."""
        from sklearn.ensemble import RandomForestRegressor
        X = np.array([self._atributos(e) for e in entradas])
        Y = (alvos - bases).astype(np.float32)     # (N, H, J, 2)
        M = mascaras.astype(np.float32)            # (N, H, J)
        self.modelos = {}
        for h in self.horizontes:
            k = h - 1
            m_k = M[:, k].mean(axis=1) > 0
            if m_k.sum() < 8:
                continue
            alvo_k = Y[:, k].reshape(len(X), -1)   # (N, J*2)
            rf = RandomForestRegressor(n_estimators=self.n_arvores, min_samples_leaf=self.min_leaf,
                                       random_state=self.semente, n_jobs=-1)
            rf.fit(X[m_k], alvo_k[m_k])
            self.modelos[h] = rf

    def prever(self, entrada, base):
        """Previsão para UMA amostra."""
        return self.prever_lote(entrada[None], base[None])[0]

    def prever_lote(self, entradas, bases):
        """Previsão para um lote (uma chamada de predict por horizonte, não por amostra)."""
        X = np.array([self._atributos(e) for e in entradas])
        out = bases.copy()
        for h, rf in self.modelos.items():
            pred = rf.predict(X).reshape(len(X), self.J, 2)
            out[:, h - 1] = bases[:, h - 1] + pred
        return out

    def __getstate__(self):
        return self.__dict__.copy()
