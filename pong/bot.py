from envpong import PongLogic
import random
import math
import os

try:
    import torch
    from simple_model import PongNet
    PYTORCH_DISPONIVEL = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    PYTORCH_DISPONIVEL = False
    print("PyTorch não disponível. Bots usarão estratégias simples.")


class BotRight:
    def __init__(self, env):
        self.env = env
        self.obs = None
    
    def act(self):
        action = random.choice([PongLogic.PaddleMove.DOWN, PongLogic.PaddleMove.STILL, PongLogic.PaddleMove.UP])  
        return action
    
    def observe(self, obs):
        self.obs = obs


class BotLeft:
    def __init__(self, env):
        self.env = env
        self.obs = [0]*len(env.observation_space.sample())
    
    def act(self):
        p1y = self.obs[1]
        bally = self.obs[9]
        
        if p1y < bally:
            action = PongLogic.PaddleMove.UP
        else:
            action = PongLogic.PaddleMove.DOWN
            
        return action
    
    def observe(self, obs):
        self.obs = obs


class BotAI_P1:
    def __init__(self, env, model_path='model/pong_model_final.pth'):
        self.env = env
        self.obs = [0]*len(env.observation_space.sample())
        
        if not PYTORCH_DISPONIVEL:
            print("AVISO: PyTorch não disponível. Use BotLeft ao invés de BotAI_P1")
            return
        
        self.model = PongNet().to(device)
        if os.path.exists(model_path):
            self.model.load(model_path)
            print(f"Modelo carregado de: {model_path}")
        else:
            print(f"AVISO: Modelo não encontrado em {model_path}")
            print("Treine primeiro usando: python treinar_bot.py")
    
    def act(self):
        if not PYTORCH_DISPONIVEL:
            p1y = self.obs[1]
            bally = self.obs[9]
            return PongLogic.PaddleMove.UP if p1y < bally else PongLogic.PaddleMove.DOWN
        
        estado = self._extrair_estado(jogador=1)
        
        with torch.no_grad():
            estado_tensor = torch.FloatTensor(estado).to(device)
            q_values = self.model(estado_tensor)
            acao_idx = q_values.argmax().item()
        
        return self._indice_para_acao(acao_idx)
    
    def observe(self, obs):
        self.obs = obs
    
    def _extrair_estado(self, jogador):
        vel_x = self.obs[10]
        vel_y = self.obs[11]
        angulo = math.atan2(vel_y, vel_x) / math.pi
        
        if jogador == 1:
            return [self.obs[1], self.obs[8], self.obs[9], angulo]
        else:
            angulo = math.atan2(vel_y, -vel_x) / math.pi
            return [self.obs[5], 1 - self.obs[8], self.obs[9], angulo]
    
    def _indice_para_acao(self, indice):
        if indice == 0:
            return PongLogic.PaddleMove.DOWN
        elif indice == 1:
            return PongLogic.PaddleMove.STILL
        else:
            return PongLogic.PaddleMove.UP


class BotAI_P2:
    def __init__(self, env, model_path='model/pong_model_final.pth'):
        self.env = env
        self.obs = [0]*len(env.observation_space.sample())
        
        if not PYTORCH_DISPONIVEL:
            print("AVISO: PyTorch não disponível. Use BotLeft ao invés de BotAI_P2")
            return
        
        self.model = PongNet().to(device)
        if os.path.exists(model_path):
            self.model.load(model_path)
            print(f"Modelo carregado de: {model_path}")
        else:
            print(f"AVISO: Modelo não encontrado em {model_path}")
            print("Treine primeiro usando: python treinar_bot.py")
    
    def act(self):
        if not PYTORCH_DISPONIVEL:
            p2y = self.obs[5]
            bally = self.obs[9]
            return PongLogic.PaddleMove.UP if p2y < bally else PongLogic.PaddleMove.DOWN
        
        estado = self._extrair_estado(jogador=2)
        
        with torch.no_grad():
            estado_tensor = torch.FloatTensor(estado).to(device)
            q_values = self.model(estado_tensor)
            acao_idx = q_values.argmax().item()
        
        return self._indice_para_acao(acao_idx)
    
    def observe(self, obs):
        self.obs = obs
    
    def _extrair_estado(self, jogador):
        vel_x = self.obs[10]
        vel_y = self.obs[11]
        angulo = math.atan2(vel_y, vel_x) / math.pi
        
        if jogador == 1:
            return [self.obs[1], self.obs[8], self.obs[9], angulo]
        else:
            angulo = math.atan2(vel_y, -vel_x) / math.pi
            return [self.obs[5], 1 - self.obs[8], self.obs[9], angulo]
    
    def _indice_para_acao(self, indice):
        if indice == 0:
            return PongLogic.PaddleMove.DOWN
        elif indice == 1:
            return PongLogic.PaddleMove.STILL
        else:
            return PongLogic.PaddleMove.UP