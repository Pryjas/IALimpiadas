# QWOP Bot

Bot autônomo para o jogo QWOP que alcança ~53 metros usando sequência otimizada por algoritmo genético.

## Arquivos

- `bot.py` - Controlador do bot
- `envqwop.py` - Ambiente de simulação
- `qwop.py` - Motor do jogo
- `character.py` - Física do personagem
- `best_sequence.pkl` - Sequência otimizada (150 ações)
- `train_bot.py` - Script de treinamento (Island Model GA)

## Funcionamento

O bot executa uma sequência fixa de 150 ações que foi otimizada por **Island Model Genetic Algorithm**:

**Ações disponíveis:**
- 0 = Coxa esquerda
- 1 = Coxa direita
- 2 = Panturrilha esquerda
- 3 = Panturrilha direita

Cada ação é mantida por 6 steps (60ms) e a sequência é repetida em loop.

**Não é adaptativo** - apenas executa o script otimizado.

## Resultados

- Distância: **~53 metros**
- Método: Island Model Genetic Algorithm
- População: 60 indivíduos em 4 ilhas
- Gerações: 100

## Retreino

```bash
python train_bot.py
```