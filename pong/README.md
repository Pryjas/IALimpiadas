# Bot de Pong com IA

Bot inteligente que joga Pong usando Deep Q-Learning (DQN).

## Estrutura do Projeto

- `bot.py` - Implementações dos bots (Random, Tracker e AI)
- `envpong.py` - Ambiente do jogo Pong
- `simple_model.py` - Rede neural (DQN)
- `treinar_bot.py` - Script de treinamento
- `treinar_intensivo.py` - Treinamento com múltiplas opções
- `testar_bot.py` - Teste do bot treinado
- `pongPlayGUI.py` - Jogo com interface gráfica
- `pongPlayNOGUI.py` - Jogo sem interface
- `model/` - Modelos treinados salvos

## Como Usar

### 1. Treinar o Bot

```bash
python treinar_bot.py
```
O treinamento irá executar 1 milhão de passos (aproximadamente 30-40 minutos).

### 2. Testar o Bot

```bash
python testar_bot.py
```

### 3. Jogar Visualmente

```bash
# Bot AI vs Tracker
python pongPlayGUI.py ai tracker

# Bot AI vs AI
python pongPlayGUI.py ai ai

# Humano vs AI (Use W/S e UP/DOWN)
python pongPlayGUI.py human ai
```

## Arquitetura da IA

**Tipo:** Deep Q-Network (DQN)

**Estado (4 features):**
- Posição Y do paddle
- Posição X da bola
- Posição Y da bola
- Ângulo da bola

**Rede Neural:**
- Input: 4 neurônios
- Hidden 1: 256 neurônios (ReLU)
- Hidden 2: 128 neurônios (ReLU)
- Output: 3 neurônios (DOWN, STILL, UP)

**Treinamento:**
- Método: Self-play com Replay Memory
- Otimizador: Adam (lr=0.001)
- Exploração: Epsilon-greedy (1.0 → 0.05)
- Recompensas: +1.0 (rebater), -1.0 (deixar passar), ±0.1 (aproximar/afastar)

## Dependências

```bash
pip install torch numpy arcade gym
```

## Observações

- Modelos são salvos a cada 10.000 passos em `model/`
- O modelo final é salvo como `pong_model_final.pth`
- Use múltiplos workers (--workers) para treinar mais rápido
- Recomendado: 1-2 milhões de passos para bom desempenho
