import arcade
from envpong import PongGUIEnv
from bot import BotRight, BotLeft, BotAI_P1, BotAI_P2
import os
import time
import threading
import sys

def runLoop(env, player1_type='ai', player2_type='tracker'):
    
    if player1_type == 'ai':
        if not os.path.exists('model/pong_model_final.pth'):
            print("⚠️  Modelo não encontrado! Usando Tracker Bot para P1")
            player1 = BotLeft(env)
        else:
            player1 = BotAI_P1(env)
            print("✅ Player 1: Bot AI")
    elif player1_type == 'tracker':
        player1 = BotLeft(env)
        print("✅ Player 1: Tracker Bot")
    elif player1_type == 'random':
        player1 = BotRight(env)
        print("✅ Player 1: Random Bot")
    else:
        player1 = None
        print("✅ Player 1: Humano (teclas W/S)")
    
    if player2_type == 'ai':
        if not os.path.exists('model/pong_model_final.pth'):
            print("⚠️  Modelo não encontrado! Usando Tracker Bot para P2")
            player2 = BotLeft(env)
        else:
            player2 = BotAI_P2(env)
            print("✅ Player 2: Bot AI")
    elif player2_type == 'tracker':
        player2 = BotLeft(env)
        print("✅ Player 2: Tracker Bot")
    elif player2_type == 'random':
        player2 = BotRight(env)
        print("✅ Player 2: Random Bot")
    else:
        player2 = None
        print("✅ Player 2: Humano (teclas UP/DOWN)")
    
    obs_state = env.game.states[-1]
    obs = env.getInputs(obs_state)
    
    if player1:
        player1.observe(obs)
    if player2:
        player2.observe(obs)
    
    for i in range(2000):
        if player1:
            actionp1 = player1.act()
        else:
            actionp1 = env.player1action
        
        if player2:
            actionp2 = player2.act()
        else:
            actionp2 = env.player2action
         
        obs, reward, done, truncated, info = env.step(actionp1, actionp2)
        
        if player1:
            player1.observe(obs)
        if player2:
            player2.observe(obs)
        
        time.sleep(env.game.dt)
        
def main():
    
    player1_type = sys.argv[1] if len(sys.argv) > 1 else 'ai'
    player2_type = sys.argv[2] if len(sys.argv) > 2 else 'tracker'
    
    print("\n" + "="*60)
    print("🎮 PONG - Modo Visual")
    print("="*60)
    print(f"Configuração: {player1_type.upper()} vs {player2_type.upper()}")
    print("-"*60)
    
    env = PongGUIEnv()
    
    threading.Thread(target=runLoop, args=(env, player1_type, player2_type)).start()
    
    arcade.run()
        

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)

    main()