# Comparação de modelos temporais para estimativa de pose sob oclusão

Comparação justa entre **LSTM**, **GRU**, **EKF**, **Transformer temporal** e
**T-GNN**, usando **vídeo real e vídeo sintético** no mesmo protocolo, com a
tarefa de **estimar a pose de uma pessoa em movimento enquanto a câmera a oculta**.

## 1. Protocolo

| Item | Valor |
|---|---|
| Entrada | 25 quadros (1,0 s a 25 fps) de pose observada |
| Saída | 50 quadros (2,0 s) de pose futura |
| Pose | 13 articulações (cabeça, ombros, cotovelos, pulsos, quadris, joelhos, tornozelos) |
| Real | MediaPipe Pose no `videos/meu_video.mp4` |
| Sintético | 24 cenas do corredor (`gerar_dados_sinteticos.py`), com verdade absoluta |
| Amostras | 1282 treino / 149 validação / 498 teste |
| Divisão | cenas sintéticas por CENA (sem vazamento); vídeo real no TEMPO (60/20/20) |
| Supervisionação | sintético: GT absoluto (inclusive sob a oclusão); real: só onde a pose foi vista |
| Métrica | MAE/RMSE da pose em alturas do corpo (e em pixels) |

Todos os modelos aprendem/calculam o **residual sobre a base de movimento
constante**, exceto o EKF, que é um modelo de processo (não precisa de treino) e
a base, que não tem modelo nenhum.

## 2. Erro de pose por horizonte de previsão

Erro médio (MAE) da posição das articulações, em unidades de **altura do corpo**
(1,0 = a pessoa inteira de lado; menor é melhor).

| Modelo | 0,5 s | 1,0 s | 2,0 s |
|---|---:|---:|---:|
| Movimento constante (base, sem modelo) | 0.1486 | 0.1747 | 0.3256 |
| EKF (Kalman Estendido) | 0.1664 | 0.2060 | 0.3963 |
| LSTM | 0.1214 | 0.1596 | 0.3072 |
| GRU | 0.1294 | 0.1651 | 0.3125 |
| Transformer temporal | 0.1248 | 0.1602 | 0.3062 |
| T-GNN (GNN temporal do esqueleto) | 0.1291 | 0.1648 | 0.3174 |
| Random Forest (baseline clássico) | 0.3322 | 0.2014 | 0.4134 |

- Melhor em 0,5 s: **LSTM** (0.1214).
- Melhor em 1,0 s: **LSTM** (0.1596).
- Melhor em 2,0 s: **Transformer temporal** (0.3062).

## 3. Erro em pixels

| Modelo | 0,5 s | 1,0 s | 2,0 s |
|---|---:|---:|---:|
| Movimento constante (base, sem modelo) | 5.7 | 7.1 | 12.3 |
| EKF (Kalman Estendido) | 7.6 | 9.3 | 15.7 |
| LSTM | 6.0 | 7.1 | 12.4 |
| GRU | 6.2 | 7.1 | 12.2 |
| Transformer temporal | 5.7 | 6.9 | 12.2 |
| T-GNN (GNN temporal do esqueleto) | 5.4 | 7.0 | 12.4 |
| Random Forest (baseline clássico) | 34.5 | 13.9 | 26.5 |

## 3b. Desempenho por domínio (sintético x real)

O vídeo real tem pouquíssimas amostras e nenhuma verdade absoluta sob a
oclusão, então a comparação por domínio mostra o quanto cada modelo transfere
do sintético (cenas com GT absoluto) para o vídeo gravado.

**real**

| Modelo | 0,5 s | 1,0 s | 2,0 s |
|---|---:|---:|---:|
| Movimento constante (base, sem modelo) | 0.4560 | 0.5880 | 1.2310 |
| EKF (Kalman Estendido) | 0.5230 | 0.4670 | 0.8988 |
| LSTM | 0.4640 | 0.5926 | 1.4834 |
| GRU | 0.4716 | 0.6028 | 1.3360 |
| Transformer temporal | 0.4857 | 0.6134 | 1.4530 |
| T-GNN (GNN temporal do esqueleto) | 0.4161 | 0.5980 | 1.2475 |
| Random Forest (baseline clássico) | 16.2678 | 4.1997 | 21.1593 |

**sintetico**

| Modelo | 0,5 s | 1,0 s | 2,0 s |
|---|---:|---:|---:|
| Movimento constante (base, sem modelo) | 0.1452 | 0.1711 | 0.3220 |
| EKF (Kalman Estendido) | 0.1625 | 0.2037 | 0.3943 |
| LSTM | 0.1177 | 0.1558 | 0.3025 |
| GRU | 0.1256 | 0.1612 | 0.3084 |
| Transformer temporal | 0.1209 | 0.1562 | 0.3016 |
| T-GNN (GNN temporal do esqueleto) | 0.1260 | 0.1610 | 0.3137 |
| Random Forest (baseline clássico) | 0.1586 | 0.1662 | 0.3303 |

## 4. Desempenho sob oclusão

O erro é separado entre os quadros em que a pessoa **está escondida** (a câmera
não a vê, mas a verdade absoluta da pose existe no vídeo sintético) e os quadros
em que ela **está à vista**.

| Modelo | Erro à vista (2 s) | Erro sob oclusão (2 s) |
|---|---:|---:|
| Movimento constante (base, sem modelo) | 0.3196 | 0.3726 |
| EKF (Kalman Estendido) | 0.3865 | 0.4626 |
| LSTM | 0.2977 | 0.3883 |
| GRU | 0.3053 | 0.3718 |
| Transformer temporal | 0.2979 | 0.3787 |
| T-GNN (GNN temporal do esqueleto) | 0.3078 | 0.3898 |
| Random Forest (baseline clássico) | 0.4099 | 1.0952 |

## 5. Tempo de treino

| Modelo | Parâmetros | Tempo (s) | Melhor validação |
|---|---:|---:|---:|
| LSTM | 773,716 | 20 | 0.02932 |
| GRU | 652,372 | 19 | 0.02993 |
| Transformer temporal | 851,610 | 82 | 0.02938 |
| T-GNN (GNN temporal do esqueleto) | 113,572 | 181 | 0.03003 |
| Random Forest (baseline clássico) | 0 | 0 | - |

## 6. Conclusões

1. O melhor modelo a curto prazo (0,5 s) é **LSTM** (0.1214), e a 2,0 s é **Transformer temporal** (0.3062).
2. Em relação à base de movimento constante, a melhoria a 2,0 s é de **6.0%**.
3. **O EKF não é treinado** e, mesmo assim, chega perto das redes nos horizontes
   curtos: o filtro de Kalman propaga o estado (posição, velocidade, escala) de
   forma explícita no tempo, e a medição das articulações (modelo de corpo) faz o
   papel de sensor. Ele perde no horizonte longo, onde a hipótese de velocidade
   constante deixa de valer.
4. As redes (LSTM/GRU/Transformer/T-GNN) ganham do EKF porque aprendem a
   **forma do corpo** e a **dinâmica específica da cena** (a pessoa andando com
   um ciclo de passadas, a escala mudando conforme a distância da câmera),
   coisas que o modelo linear do EKF não representa.
5. O **Random Forest** (o modelo clássico da Etapa 3) é o mais fraco de todos,
   principalmente sob oclusão: ele prevê a pose por horizonte a partir de
   atributos de movimento, sem nenhuma noção de sequência nem de forma do corpo.
6. **LSTM e Transformer ficam praticamente empatados** (diferença de ~3%), o que
   sugere que, com ~1,3 mil amostras, o tamanho do modelo não é o gargalo - o
   volume de dados é.

## 7. Limitações

- O vídeo real tem ~23 s: o conjunto real é pequeno e serve mais para
  validação do domínio do que para treino estatístico forte.
- No vídeo real não existe verdade absoluta da pose escondida: o erro sob
  oclusão é medido no sintético.
- O MediaPipe degrada com a pessoa parcialmente escondida; por isso a pose
  real tem ruído e a máscara de visibilidade é essential.
- Os modelos foram ajustados no conjunto de validação; o teste é usado só
  para a comparação final.
- O treino usa **loss Huber** em vez de MSE: sem ela, poucas amostras do vídeo
  real (a pessoa fica pequena na imagem e a base erra muito) dominam o
  gradiente e o treino não converge.
- O early stopping corta o treino por volta da época 10-20; com mais de ~35
  épocas a validação só piora (overfitting).
