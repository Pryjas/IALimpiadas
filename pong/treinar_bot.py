import torch
import numpy as np
import math
import random
from collections import deque
from envpong import PongEnv, PongLogic
from simple_model import PongNet, SimpleTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Treinando em: {device}")


def extrair_estado(obs, jogador=1):
    vel_x = obs[10]
    vel_y = obs[11]
    angulo = math.atan2(vel_y, vel_x) / math.pi
    
    if jogador == 1:
        paddle_y = obs[1]
        ball_x = obs[8]
        ball_y = obs[9]
    else:
        paddle_y = obs[5]
        ball_x = 1 - obs[8]
        ball_y = obs[9]
        angulo = math.atan2(vel_y, -vel_x) / math.pi
    
    return [paddle_y, ball_x, ball_y, angulo]


def indice_para_acao(indice):
    if indice == 0:
        return PongLogic.PaddleMove.DOWN
    elif indice == 1:
        return PongLogic.PaddleMove.STILL
    else:
        return PongLogic.PaddleMove.UP


def calcular_recompensa(obs_nova, obs_antiga):
    reward_p1 = 0.0
    reward_p2 = 0.0
    pontuou = False
    
    vel_x_antiga = obs_antiga[10]
    vel_x_nova = obs_nova[10]
    
    if vel_x_antiga < 0 and vel_x_nova > 0:
        reward_p1 = 1.0
        pontuou = True
    
    if vel_x_antiga > 0 and vel_x_nova < 0:
        reward_p2 = 1.0
        pontuou = True
    
    if obs_nova[8] <= 0.05:
        reward_p1 = -1.0
        pontuou = True
    
    if obs_nova[8] >= 0.95:
        reward_p2 = -1.0
        pontuou = True
    
    if not pontuou:
        dist_antiga_p1 = abs(obs_antiga[1] - obs_antiga[9])
        dist_nova_p1 = abs(obs_nova[1] - obs_nova[9])
        if dist_nova_p1 < dist_antiga_p1:
            reward_p1 += 0.1
        else:
            reward_p1 -= 0.05
        
        dist_antiga_p2 = abs(obs_antiga[5] - obs_antiga[9])
        dist_nova_p2 = abs(obs_nova[5] - obs_nova[9])
        if dist_nova_p2 < dist_antiga_p2:
            reward_p2 += 0.1
        else:
            reward_p2 -= 0.05
    
    return reward_p1, reward_p2, pontuou


class ReplayMemory:
    
    def __init__(self, capacidade=50000):
        self.memoria = deque(maxlen=capacidade)
    
    def adicionar(self, estado, acao, recompensa, prox_estado, done):
        self.memoria.append((estado, acao, recompensa, prox_estado, done))
    
    def amostrar(self, tamanho_batch):
        batch = random.sample(self.memoria, tamanho_batch)
        
        estados, acoes, recompensas, prox_estados, dones = zip(*batch)
        
        return (list(estados), list(acoes), list(recompensas), 
                list(prox_estados), list(dones))
    
    def __len__(self):
        return len(self.memoria)


def treinar(num_passos=100000, batch_size=64, epsilon_inicial=1.0, num_workers=4):
    torch.set_num_threads(num_workers)
    
    env = PongEnv(debugPrint=False)
    modelo = PongNet().to(device)
    treinador = SimpleTrainer(modelo, learning_rate=0.001)
    
    print(f"Usando {torch.get_num_threads()} threads de CPU")
    print(f"Dispositivo: {device}")
    
    memoria = ReplayMemory(capacidade=50000)
    
    epsilon = epsilon_inicial
    epsilon_min = 0.05
    epsilon_decay = 0.99995
    
    total_recompensa_p1 = 0
    total_recompensa_p2 = 0
    rebatidas = 0
    episodio = 0
    
    obs, _ = env.reset()
    
    print("\n=== Iniciando Treinamento ===\n")
    print("Progresso será mostrado a cada 1000 passos\n")
    
    for passo in range(1, num_passos + 1):
        
        estado_p1 = extrair_estado(obs, jogador=1)
        estado_p2 = extrair_estado(obs, jogador=2)
        
        if random.random() < epsilon:
            acao_p1_idx = random.randint(0, 2)
            acao_p2_idx = random.randint(0, 2)
        else:
            with torch.no_grad():
                estado_p1_tensor = torch.FloatTensor(estado_p1).to(device)
                estado_p2_tensor = torch.FloatTensor(estado_p2).to(device)
                
                q_values_p1 = modelo(estado_p1_tensor)
                q_values_p2 = modelo(estado_p2_tensor)
                
                acao_p1_idx = q_values_p1.argmax().item()
                acao_p2_idx = q_values_p2.argmax().item()
        
        acao_p1 = indice_para_acao(acao_p1_idx)
        acao_p2 = indice_para_acao(acao_p2_idx)
        
        obs_antiga = obs
        obs, _, _, _, _ = env.step(acao_p1, acao_p2)
        
        reward_p1, reward_p2, pontuou = calcular_recompensa(obs, obs_antiga)
        
        prox_estado_p1 = extrair_estado(obs, jogador=1)
        prox_estado_p2 = extrair_estado(obs, jogador=2)
        
        memoria.adicionar(estado_p1, acao_p1_idx, reward_p1, prox_estado_p1, pontuou)
        memoria.adicionar(estado_p2, acao_p2_idx, reward_p2, prox_estado_p2, pontuou)
        
        total_recompensa_p1 += reward_p1
        total_recompensa_p2 += reward_p2
        if reward_p1 > 0 or reward_p2 > 0:
            rebatidas += 1
        
        if len(memoria) >= batch_size:
            batch = memoria.amostrar(batch_size)
            treinador.train_step(*batch)
        
        if passo % 10 == 0 and len(memoria) >= batch_size * 2:
            batch_grande = memoria.amostrar(batch_size * 2)
            treinador.train_step(*batch_grande)
        
        if pontuou:
            obs, _ = env.reset()
            episodio += 1
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if passo % 1000 == 0:
            avg_reward_p1 = total_recompensa_p1 / 1000
            avg_reward_p2 = total_recompensa_p2 / 1000
            
            print(f"Passo {passo:6d} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Rebatidas: {rebatidas:3d} | "
                  f"Episódios: {episodio:3d} | "
                  f"Reward P1: {avg_reward_p1:+.2f} | "
                  f"Reward P2: {avg_reward_p2:+.2f}")
            
            total_recompensa_p1 = 0
            total_recompensa_p2 = 0
            rebatidas = 0
            episodio = 0
        
        if passo % 10000 == 0:
            modelo.save(f'model/pong_model_{passo}.pth')
            print(f"  → Modelo salvo em passo {passo}")
    
    modelo.save('model/pong_model_final.pth')
    print("\n=== Treinamento Completo ===")
    print("Modelo final salvo em: model/pong_model_final.pth")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Treinar bot de Pong')
    parser.add_argument('--passos', type=int, default=100000, 
                        help='Número de passos de treinamento')
    parser.add_argument('--batch', type=int, default=64,
                        help='Tamanho do batch')
    parser.add_argument('--workers', type=int, default=4,
if __name__ == "__main__":
    treinar(num_passos=1000000, batch_size=128, num_workers=8)