from envpong import PongEnv
from bot import BotAI_P1, BotLeft, BotRight
import os

def testar_bot():
    if not os.path.exists('model/pong_model_final.pth'):
        print("Modelo não encontrado!")
        print("Execute primeiro: python treinar_bot.py")
        return
    
    print("\n=== Testando Bot com IA ===\n")

    env = PongEnv(debugPrint=True)
    
    # Teste 1: AI vs Tracker Bot
    print("Teste 1: Bot AI (P1) vs Tracker Bot (P2)")
    print("-" * 50)
    
    bot_ai = BotAI_P1(env)
    bot_tracker = BotLeft(env)
    
    obs, _ = env.reset()
    bot_ai.observe(obs)
    bot_tracker.observe(obs)
    
    for i in range(500):
        acao1 = bot_ai.act()
        acao2 = bot_tracker.act()
        
        obs, _, _, _, _ = env.step(acao1, acao2)
        
        bot_ai.observe(obs)
        bot_tracker.observe(obs)
    
    score_ai = env.game.states[-1].player1Score
    score_tracker = env.game.states[-1].player2Score
    
    print(f"\nResultado: AI {score_ai} x {score_tracker} Tracker")
    if score_ai > score_tracker:
        print("Bot AI venceu!")
    elif score_ai < score_tracker:
        print("Tracker Bot venceu")
    else:
        print("Empate")
    
    print("\n" + "="*50)
    print("Teste 2: Bot AI (P1) vs Random Bot (P2)")
    print("-" * 50)
    
    bot_ai = BotAI_P1(env)
    bot_random = BotRight(env)
    
    obs, _ = env.reset()
    bot_ai.observe(obs)
    bot_random.observe(obs)
    
    for i in range(500):
        acao1 = bot_ai.act()
        acao2 = bot_random.act()
        
        obs, _, _, _, _ = env.step(acao1, acao2)
        
        bot_ai.observe(obs)
        bot_random.observe(obs)
    
    score_ai = env.game.states[-1].player1Score
    score_random = env.game.states[-1].player2Score
    
    print(f"\nResultado: AI {score_ai} x {score_random} Random")
    if score_ai > score_random:
        print("Bot AI VENCEU!")
    elif score_ai < score_random:
        print("Random Bot venceu")
    else:
        print("Empate")
    
    print("\n" + "="*50)
    print("Testes finalizados.")

if __name__ == "__main__":
    testar_bot()
