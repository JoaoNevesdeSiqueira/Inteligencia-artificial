import os
import pandas as pd
import random
import csv
import time  

# Função que lê os dados do arquivo da mochila
# Aqui pegamos pesos, valores e a capacidade da mochila
def ler_instancia(filepath):
    df = pd.read_csv(filepath).dropna(subset=["Peso"])
    
    # Pegamos a capacidade da mochila
    capacidade = int(df[df["Item"].str.contains("Capacidade")]["Peso"].values[0])
    
    # Tiramos a linha da capacidade para ficar só os itens
    df = df[~df["Item"].str.contains("Capacidade")]
    
    # Transformamos as colunas em listas
    pesos = df["Peso"].astype(int).tolist()
    valores = df["Valor"].astype(int).tolist()
    
    return pesos, valores, capacidade

# Função para avaliar o indivíduo
# Aqui vemos se ele respeita a capacidade da mochila
def fitness(individuo, pesos, valores, capacidade):
    peso_total = sum([gene * peso for gene, peso in zip(individuo, pesos)])
    valor_total = sum([gene * valor for gene, valor in zip(individuo, valores)])
    
    # Se o peso for maior que a capacidade, não pode, então valor é 0
    if peso_total > capacidade:
        return 0
    else:
        return valor_total

# Tipos de crossover
# Aqui fazemos o cruzamento de dois pais para gerar um filho

# Crossover de 1 ponto: cortamos o pai e colamos um pedaço do outro
def crossover_1p(pai1, pai2):
    ponto = random.randint(1, len(pai1) - 1)
    return pai1[:ponto] + pai2[ponto:]

# Crossover de 2 pontos: cortamos em dois lugares
def crossover_2p(pai1, pai2):
    p1, p2 = sorted(random.sample(range(len(pai1)), 2))
    return pai1[:p1] + pai2[p1:p2] + pai1[p2:]

# Crossover uniforme: sorteamos gene a gene de qual pai vem
def crossover_uniforme(pai1, pai2):
    return [pai1[i] if random.random() < 0.5 else pai2[i] for i in range(len(pai1))]

# Mutação: invertemos um gene com uma certa chance
def mutar(individuo, taxa_mutacao):
    return [1 - gene if random.random() < taxa_mutacao else gene for gene in individuo]

# Inicialização aleatória da população
def populacao_aleatoria(tam_individuo, tam_populacao):
    return [[random.randint(0, 1) for _ in range(tam_individuo)] for _ in range(tam_populacao)]

# Inicialização com heurística: tenta colocar o máximo de itens possível
def populacao_heuristica(pesos, capacidade, tam_populacao):
    populacao = []
    for _ in range(tam_populacao):
        individuo = [0] * len(pesos)
        peso_acumulado = 0
        indices = list(range(len(pesos)))
        random.shuffle(indices)
        for i in indices:
            if peso_acumulado + pesos[i] <= capacidade:
                individuo[i] = 1
                peso_acumulado += pesos[i]
        populacao.append(individuo)
    return populacao

# Função principal do Algoritmo Genético
def algoritmo_genetico(pesos, valores, capacidade, tam_populacao, geracoes, crossover_fn, taxa_mutacao, tipo_init, usar_convergencia=False):
    
    n = len(pesos)
    
    # Inicializamos a população
    if tipo_init == "aleatoria":
        populacao = populacao_aleatoria(n, tam_populacao)
    else:
        populacao = populacao_heuristica(pesos, capacidade, tam_populacao)
    
    melhor_fitness = 0
    estagnado = 0  
    for _ in range(geracoes):
        
        # Ordenamos a população pelo fitness
        populacao = sorted(populacao, key=lambda ind: fitness(ind, pesos, valores, capacidade), reverse=True)
        
        nova_populacao = populacao[:2]  # Mantemos os 2 melhores (elitismo)
        
        while len(nova_populacao) < tam_populacao:
            # Seleção dos pais (entre os 10 melhores)
            pais = random.sample(populacao[:10], 2)
            
            # Cruzamos os pais
            filho = crossover_fn(pais[0], pais[1])
            
            # Mutamos o filho
            filho = mutar(filho, taxa_mutacao)
            
            nova_populacao.append(filho)
        
        populacao = nova_populacao
        
        # Critério de parada por convergência
        atual_melhor = fitness(populacao[0], pesos, valores, capacidade)
        
        if usar_convergencia:
            if atual_melhor == melhor_fitness:
                estagnado += 1
            else:
                estagnado = 0
                melhor_fitness = atual_melhor
            if estagnado >= 10:  # Paramos se não melhorar em 10 gerações
                break

    melhor_individuo = max(populacao, key=lambda ind: fitness(ind, pesos, valores, capacidade))
    return melhor_individuo, fitness(melhor_individuo, pesos, valores, capacidade)

# Configurações de operadores
crossovers = {
    "1_ponto": crossover_1p,
    "2_pontos": crossover_2p,
    "uniforme": crossover_uniforme
}
mutacoes = {
    "baixa": 0.01,
    "media": 0.05,
    "alta": 0.1
}
inicializacoes = ["aleatoria", "heuristica"]
criterios_parada = [False, True]  # False: gerações fixas, True: convergência

# Aqui guardamos os resultados
resultados = []

print("\n--- TESTANDO CONFIGURAÇÕES ---")

for i in range(1, 11):
    caminho = f"knapsack_{i}.csv"
    
    if not os.path.exists(caminho):
        print(f"Arquivo não encontrado: {caminho}")
        continue
    
    pesos, valores, capacidade = ler_instancia(caminho)
    
    for nome_cross, func_cross in crossovers.items():
        for nome_mut, taxa_mut in mutacoes.items():
            for init in inicializacoes:
                for parada in criterios_parada:
                    
                    # Medimos o tempo de execução
                    inicio = time.time()
                    
                    solucao, valor = algoritmo_genetico(
                        pesos, valores, capacidade,
                        tam_populacao=50,
                        geracoes=100,
                        crossover_fn=func_cross,
                        taxa_mutacao=taxa_mut,
                        tipo_init=init,
                        usar_convergencia=parada
                    )
                    
                    fim = time.time()
                    tempo_execucao = fim - inicio  # Calculamos o tempo
                    
                    peso_total = sum([gene * peso for gene, peso in zip(solucao, pesos)])
                    
                    resultados.append([
                        i, nome_cross, nome_mut, init, 
                        "convergencia" if parada else "fixo", 
                        valor, peso_total, capacidade,
                        round(tempo_execucao, 4)
                    ])
                    
                    print(f"Instância {i}, crossover={nome_cross}, mutação={nome_mut}, init={init}, parada={'convergencia' if parada else 'fixo'} => Valor = {valor}, Peso = {peso_total}/{capacidade}, Tempo = {tempo_execucao:.4f} seg")

# Salva os resultados em CSV
with open("resultados_ag_mochila.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Instancia", "Crossover", "Mutacao", "Inicializacao", "Criterio_Parada", "Valor", "Peso_Total", "Capacidade", "Tempo"])
    writer.writerows(resultados)

print("\nResultados salvos em 'resultados_ag_mochila.csv'")
