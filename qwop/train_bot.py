from envqwop import QWOPEnv
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def avaliar_sequencia(sequence, max_steps=5000):
    # Testa uma sequência e retorna a distância alcançada
    env = QWOPEnv(screen=False)
    
    MOVEMENT_PERSIST = 6
    n = len(sequence) * MOVEMENT_PERSIST
    x = np.zeros((n, 4))
    
    # Converte ações discretas (0-3) em forças
    for (i, action) in enumerate(sequence):
        idx_start = i * MOVEMENT_PERSIST
        idx_end = idx_start + MOVEMENT_PERSIST
        
        if action == 0:
            x[idx_start:idx_end, 0] = 1
            x[idx_start:idx_end, 1] = -1
        elif action == 1:
            x[idx_start:idx_end, 0] = -1
            x[idx_start:idx_end, 1] = 1
        elif action == 2:
            x[idx_start:idx_end, 2] = 1
            x[idx_start:idx_end, 3] = -1
        elif action == 3:
            x[idx_start:idx_end, 2] = -1
            x[idx_start:idx_end, 3] = 1
    
    obs, _ = env.reset()
    max_reward = 0
    total_progress = 0
    sustained_progress = 0
    
    # Testa com 5 repetições da sequência
    for cycle in range(5):
        cycle_start = max_reward
        for i in range(min(len(x), max_steps)):
            obs, reward, done, truncated, info = env.step(x[i])
            if reward > max_reward:
                max_reward = reward
        
        cycle_progress = max_reward - cycle_start
        total_progress += cycle_progress
        if cycle > 0:
            sustained_progress += cycle_progress
    
    # Fitness = distância + consistência
    fitness = (max_reward * 1.0 + 
               sustained_progress * 0.3 + 
               (total_progress / (len(x) * 5)) * 0.5)
    
    return fitness

def avaliar_wrapper(seq):
    return avaliar_sequencia(seq.tolist())

def algoritmo_genetico(num_islands=4, pop_per_island=15, generations=100, sequence_length=150):
    print(f"Treinamento iniciado: {num_islands * pop_per_island} indivíduos, {generations} gerações\n")
    
    
    # Inicializa ilhas com padrões diferentes
    islands = []
    seeds = [
        [2, 1, 2, 1, 1, 1, 0, 3, 2, 1, 2, 1, 2, 1, 3, 3],
        [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
        [0, 1, 0, 1, 2, 3, 2, 3],
        [2, 3, 2, 3, 1, 0, 1, 0],
    ]
    
    for island_id in range(num_islands):
        island = []
        seed = seeds[island_id % len(seeds)]
        
        for _ in range(pop_per_island):
            seq = np.random.randint(0, 4, sequence_length)
            if len(seed) <= sequence_length:
                seq[:len(seed)] = seed
            island.append(seq)
        
        islands.append(island)
    
    global_best = None
    global_best_fitness = 0
    migration_interval = 20
    
    # Evolução por gerações
    for gen in range(generations):
        # Avalia todos os indivíduos em paralelo
        all_individuals = []
        for island in islands:
            all_individuals.extend(island)
        
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            all_fitnesses = list(executor.map(avaliar_wrapper, all_individuals))
        
        # Divide resultados de volta para as ilhas
        island_fitnesses = []
        idx = 0
        for island in islands:
            island_fitnesses.append(all_fitnesses[idx:idx+len(island)])
            idx += len(island)
        
        # Atualiza melhor global
        for island_id, (island, fitnesses) in enumerate(zip(islands, island_fitnesses)):
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > global_best_fitness:
                global_best_fitness = fitnesses[best_idx]
                global_best = island[best_idx].copy()
                print(f"Geração {gen+1}: Novo recorde de {global_best_fitness:.2f}m")
        
        # Evolui cada ilha
        new_islands = []
        for island_id, (island, fitnesses) in enumerate(zip(islands, island_fitnesses)):
            # Mantém os melhores (elitismo)
            elite_count = max(2, pop_per_island // 5)
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            new_island = [island[i].copy() for i in elite_indices]
            
            # Gera novos indivíduos (cruzamento)
            while len(new_island) < pop_per_island:
                # Seleciona pais por torneio
                t1, t2 = np.random.choice(len(island), 2, replace=False)
                parent1 = island[t1] if fitnesses[t1] > fitnesses[t2] else island[t2]
                
                t1, t2 = np.random.choice(len(island), 2, replace=False)
                parent2 = island[t1] if fitnesses[t1] > fitnesses[t2] else island[t2]
                
                # Cruzamento
                point = np.random.randint(1, sequence_length)
                child = np.concatenate([parent1[:point], parent2[point:]])
                
                # Mutação
                mutation_rate = 0.1
                mask = np.random.random(sequence_length) < mutation_rate
                child[mask] = np.random.randint(0, 4, mask.sum())
                
                new_island.append(child)
            
            new_islands.append(new_island[:pop_per_island])
        
        islands = new_islands
        
        # Migração entre ilhas (silenciosa)
        if (gen + 1) % migration_interval == 0:
            for i in range(num_islands):
                next_island = (i + 1) % num_islands
                donor = islands[i][0].copy()
                islands[next_island][-1] = donor
        
        # Busca local no melhor (silenciosa)
        if (gen + 1) % 10 == 0 and global_best is not None:
            for _ in range(5):
                neighbor = global_best.copy()
                pos = np.random.randint(0, sequence_length)
                neighbor[pos] = np.random.randint(0, 4)
                
                fitness = avaliar_sequencia(neighbor.tolist())
                if fitness > global_best_fitness:
                    global_best = neighbor
                    global_best_fitness = fitness
    
    # Salva melhor sequência
    with open('best_sequence.pkl', 'wb') as f:
        pickle.dump(global_best.tolist(), f)
    
    print(f"\nTreinamento concluído. Distancia alcançada: {global_best_fitness:.2f}m")
    
    return global_best, global_best_fitness

if __name__ == "__main__":
    best_seq, best_fitness = algoritmo_genetico(
        num_islands=4,
        pop_per_island=15,
        generations=100,
        sequence_length=150
    )