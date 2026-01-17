# --- Imports de Pacotes ---
# Requer instalação: import Pkg; Pkg.add(["Graphs", "JuMP", "Gurobi", "MathOptInterface"])
# 
# VERSÃO COM BELLMAN-FORD (MCNF_BF.jl)
# Este arquivo é uma versão modificada do geradorMCNF.jl que usa o algoritmo de Bellman-Ford
# quando há custos reduzidos negativos durante o pricing. Isso é necessário porque o Dijkstra
# não funciona corretamente com custos negativos, e os custos reduzidos podem ser negativos
# mesmo quando os custos originais são positivos (devido aos multiplicadores duais).
#
using Random, Graphs, Printf, JuMP, Gurobi, MathOptInterface
const MOI = MathOptInterface

# --- Estruturas de Dados ---

"""
Representa uma mercadoria (commodity) que precisa ser transportada.
"""
struct Commodity
    id::Int
    source::Int
    sink::Int
    demand::Float64
    # Armazenar todos os sources e sinks para modelagem correta
    sources::Dict{Int, Float64}  # node -> supply (supply > 0)
    sinks::Dict{Int, Float64}    # node -> demand (demand > 0)
end

"""
Representa a instância completa do problema MCNFP.
"""
struct MCNFPInstance
    graph::SimpleDiGraph          # A topologia da rede
    capacities::Dict{Edge, Float64} # Capacidade de cada arco u -> v
    costs::Dict{Tuple{Int, Edge}, Float64}  # Custo unitário de fluxo: (commodity_id, edge) -> custo
    commodities::Vector{Commodity}  # Lista de mercadorias
end

# --- Função Geradora ---

"""
    generate_mcnfp_instance(; num_nodes, num_commodities, density, seed, max_cap, max_cost, max_demand)

Gera uma instância aleatória para o Multi-Commodity Network Flow Problem.

# Argumentos
- `num_nodes`: Número de nós na rede.
- `num_commodities`: Número de mercadorias (pares origem-destino).
- `density`: Probabilidade de existir uma aresta entre dois nós (0.0 a 1.0).
- `seed`: Semente para o gerador de números aleatórios (reprodutibilidade).
"""
function generate_mcnfp_instance(;
    num_nodes::Int=10,
    num_commodities::Int=3,
    density::Float64=0.3,
    seed::Int=42,
    max_cap::Float64=100.0,
    max_cost::Float64=20.0,
    max_demand::Float64=10.0
    )
    # 1. Configurar a semente aleatória
    Random.seed!(seed)

    # 2. Gerar a topologia do grafo (Direcionado)
    # Erdős-Rényi: cria arestas com probabilidade p = density
    g = erdos_renyi(num_nodes, density, is_directed=true)

    # Garante que não há self-loops (u -> u)
    for i in 1:num_nodes
        if has_edge(g, i, i)
            rem_edge!(g, i, i)
        end
    end

    # (Opcional) Garantir conectividade mínima:
    # Adiciona um ciclo para garantir que o grafo não seja totalmente desconexo,
    # embora isso não garanta viabilidade para todas as commodities.
    for i in 1:(num_nodes-1)
        add_edge!(g, i, i+1)
    end
    add_edge!(g, num_nodes, 1)

    # 3. Gerar atributos dos Arcos (Capacidade e Custo)
    capacities = Dict{Edge, Float64}()
    costs = Dict{Tuple{Int, Edge}, Float64}()

    for e in edges(g)
        # Capacidade aleatória entre 10% do max e o max
        capacities[e] = round(rand() * (max_cap - (max_cap*0.1)) + (max_cap*0.1), digits=0)
        
        # Custo aleatório por commodity (para instâncias aleatórias, usamos o mesmo custo para todas as commodities)
        base_cost = round(rand() * max_cost, digits=0)
        for k in 1:num_commodities
            costs[(k, e)] = base_cost
        end
    end

    # 4. Gerar Commodities (Vários Sources e Sinks)
    commodities = Vector{Commodity}()
    
    for k in 1:num_commodities
        s = rand(1:num_nodes)
        t = rand(1:num_nodes)

        # Garante que source != sink
        while s == t
            t = rand(1:num_nodes)
        end

        demand = round(rand() * max_demand + 1.0, digits=0)
        # Para instâncias geradas aleatoriamente, criar sources e sinks simples
        sources_dict = Dict(s => demand)
        sinks_dict = Dict(t => demand)
        push!(commodities, Commodity(k, s, t, demand, sources_dict, sinks_dict))
    end

    return MCNFPInstance(g, capacities, costs, commodities)
end

# --- Função Auxiliar para Exibição ---

function print_instance_summary(inst::MCNFPInstance)
    println("=== Resumo da Instância MCNFP ===")
    println("Nós: ", nv(inst.graph))
    println("Arcos: ", ne(inst.graph))
    println("Mercadorias: ", length(inst.commodities))
    println("-"^30)
    
    println("Mercadorias (Amostra):")
    for c in first(inst.commodities, 5) # Mostra até 5
        @printf("  ID %d: %d -> %d (Demanda: %.2f)\n", c.id, c.source, c.sink, c.demand)
    end
    if length(inst.commodities) > 5
        println("  ...")
    end

    println("-"^30)
    println("Arcos (Amostra):")
    count = 0
    for e in edges(inst.graph)
        count += 1
        cap = inst.capacities[e]
        # Mostrar custos para todas as commodities
        cost_strs = String[]
        for comm in inst.commodities
            cost = get(inst.costs, (comm.id, e), 0.0)
            push!(cost_strs, "k$(comm.id)=$(cost)")
        end
        costs_display = join(cost_strs, ", ")
        @printf("  %d -> %d | Cap: %.2f | Custos: %s\n", src(e), dst(e), cap, costs_display)
        if count >= 5 break end
    end
end

# --- Exemplo de Uso ---

# Gerar uma instância com seed fixa (sempre gerará os mesmos dados)
instancia = generate_mcnfp_instance(
    num_nodes=8, 
    num_commodities=4, 
    density=0.4, 
    seed=123
)

print_instance_summary(instancia)

"""
    check_feasibility_lp(inst::MCNFPInstance; verbose=true)

Constrói e resolve um modelo de Programação Linear para verificar se a instância
do MCNFP é viável (possível de satisfazer todas as demandas respeitando as capacidades).
Retorna `true` se viável, `false` caso contrário.
"""
function check_feasibility_lp(inst::MCNFPInstance; verbose=true)
    # 1. Inicializar o Modelo com o solver Gurobi
    model = Model(Gurobi.Optimizer)
    
    if !verbose
        set_silent(model)
    end

    # Atalhos para facilitar a leitura
    g = inst.graph
    comms = inst.commodities
    num_k = length(comms)

    # 2. Variáveis de Decisão
    # x[k, u, v]: Fluxo da mercadoria k no arco u->v
    # Usamos um dicionário ou array esparso mapeando (k, edge) -> variavel
    @variable(model, x[k=1:num_k, e=edges(g)] >= 0)

    # 3. Restrições de Capacidade (Bundle Constraints)
    # A soma do fluxo de todas as mercadorias em um arco não pode exceder a capacidade
    for e in edges(g)
        @constraint(model, sum(x[k, e] for k in 1:num_k) <= inst.capacities[e])
    end

    # 4. Restrições de Conservação de Fluxo (Flow Conservation)
    # Para cada mercadoria k e cada nó v: (Fluxo Sai) - (Fluxo Entra) = Balanço
    # IMPORTANTE: Modelar TODOS os sources e sinks de cada commodity
    for (k_idx, commodity) in enumerate(comms)
        for v in vertices(g)
            # Calcular o balanço líquido esperado no nó v
            # Considerando TODOS os sources e sinks desta commodity
            net_demand = 0.0
            
            # Se v é um source, adicionar o supply (positivo = gera fluxo)
            if haskey(commodity.sources, v)
                net_demand += commodity.sources[v]
            end
            
            # Se v é um sink, subtrair a demanda (negativo = consome fluxo)
            if haskey(commodity.sinks, v)
                net_demand -= commodity.sinks[v]
            end
            # Nós de passagem (transbordo) têm net_demand = 0.0

            # Fluxo saindo de v (para vizinhos out)
            flow_out = @expression(model, sum(x[k_idx, Edge(v, neighbor)] for neighbor in outneighbors(g, v)))
            
            # Fluxo entrando em v (de vizinhos in)
            flow_in = @expression(model, sum(x[k_idx, Edge(neighbor, v)] for neighbor in inneighbors(g, v)))

            # Restrição: Sai - Entra = Demanda Líquida
            @constraint(model, flow_out - flow_in == net_demand)
        end
    end

    # 5. Função Objetivo
    # Para checar viabilidade, qualquer objetivo serve. 
    # Vamos minimizar o custo total para achar a solução mais barata se for viável.
    @objective(model, Min, 
        sum(get(inst.costs, (k, e), 0.0) * x[k, e] for k in 1:num_k, e in edges(g))
    )

    # 6. Resolver
    optimize!(model)

    # 7. Verificar Status
    status = termination_status(model)
    
    is_feasible = (status == MOI.OPTIMAL)

    if verbose
        println("\n=== Resultado da Verificação LP ===")
        println("Status do Solver: ", status)
        
        if is_feasible
            total_cost = objective_value(model)
            println("✅ Instância VIÁVEL.")
            @printf("Custo Mínimo Total: %.2f\n", total_cost)
        else
            println("❌ Instância INVIÁVEL (Capacidades insuficientes ou grafo desconexo).")
        end
        println("-"^30)
    end

    return is_feasible
end

# --- Função de Geração de Colunas ---
# IMPORTANTE: Esta seção requer os imports no topo do arquivo:
#   using Random, Graphs, Printf, JuMP, Gurobi, MathOptInterface
#   const MOI = MathOptInterface
# Se você copiar apenas esta seção, certifique-se de incluir os imports acima.

"""
    edge_in_path(e::Edge, path::Vector{Edge})::Bool

Verifica se um arco está presente em um caminho.
"""
function edge_in_path(e::Edge, path::Vector{Edge})::Bool
    for path_edge in path
        if src(path_edge) == src(e) && dst(path_edge) == dst(e)
            return true
        end
    end
    return false
end

"""
    paths_equal(path1::Vector{Edge}, path2::Vector{Edge})::Bool

Verifica se dois caminhos são iguais (mesmos arcos na mesma ordem).
"""
function paths_equal(path1::Vector{Edge}, path2::Vector{Edge})::Bool
    if length(path1) != length(path2)
        return false
    end
    for i in 1:length(path1)
        if src(path1[i]) != src(path2[i]) || dst(path1[i]) != dst(path2[i])
            return false
        end
    end
    return true
end

"""
    dijkstra_shortest_path(g::SimpleDiGraph, costs::Dict{Edge, Float64}, source::Int, target::Int)

Encontra o caminho mais curto de source para target usando o algoritmo de Dijkstra.
Retorna (distância, caminho como lista de nós, caminho como lista de arcos).
Se não houver caminho, retorna (Inf, [], []).
"""
function dijkstra_shortest_path(g::SimpleDiGraph, costs::Dict{Edge, Float64}, source::Int, target::Int)
    num_nodes = nv(g)
    dist = fill(Inf, num_nodes)
    prev = fill(0, num_nodes)
    visited = falses(num_nodes)
    
    dist[source] = 0.0
    
    for _ in 1:num_nodes
        # Encontrar o nó não visitado com menor distância
        u = 0
        min_dist = Inf
        for v in 1:num_nodes
            if !visited[v] && dist[v] < min_dist
                min_dist = dist[v]
                u = v
            end
        end
        
        if u == 0 || min_dist == Inf
            break  # Não há mais nós alcançáveis
        end
        
        visited[u] = true
        
        # Se chegamos ao destino, podemos parar
        if u == target
            break
        end
        
        # Relaxar arestas saindo de u
        for neighbor in outneighbors(g, u)
            if !visited[neighbor]
                e = Edge(u, neighbor)
                if haskey(costs, e)
                    alt = dist[u] + costs[e]
                    if alt < dist[neighbor]
                        dist[neighbor] = alt
                        prev[neighbor] = u
                    end
                end
            end
        end
    end
    
    # Reconstruir o caminho
    if !isfinite(dist[target]) || dist[target] == Inf
        return (Inf, Int[], Edge[])
    end
    
    path_nodes = Int[]
    path_edges = Edge[]
    u = target
    
    while u != 0
        pushfirst!(path_nodes, u)
        if prev[u] != 0
            pushfirst!(path_edges, Edge(prev[u], u))
        end
        u = prev[u]
    end
    
    return (dist[target], path_nodes, path_edges)
end

"""
    bellman_ford_shortest_path(g::SimpleDiGraph, costs::Dict{Edge, Float64}, source::Int, target::Int)

Encontra o caminho mais curto de source para target usando o algoritmo de Bellman-Ford.
Funciona mesmo com custos negativos (mas não com ciclos negativos alcançáveis).
Retorna (distância, caminho como lista de nós, caminho como lista de arcos).
Se não houver caminho, retorna (Inf, [], []).
Se houver ciclo negativo alcançável, retorna (-Inf, [], []).
"""
function bellman_ford_shortest_path(g::SimpleDiGraph, costs::Dict{Edge, Float64}, source::Int, target::Int)
    num_nodes = nv(g)
    dist = fill(Inf, num_nodes)
    prev = fill(0, num_nodes)
    
    dist[source] = 0.0
    
    # Relaxar arestas até num_nodes - 1 vezes
    for _ in 1:(num_nodes - 1)
        improved = false
        for e in edges(g)
            u = src(e)
            v = dst(e)
            if isfinite(dist[u]) && haskey(costs, e)
                alt = dist[u] + costs[e]
                if alt + 1e-12 < dist[v]  # Tolerância numérica
                    dist[v] = alt
                    prev[v] = u
                    improved = true
                end
            end
        end
        if !improved
            break  # Convergência antecipada
        end
    end
    
    # Verificar ciclos negativos alcançáveis
    negative_cycle = false
    for e in edges(g)
        u = src(e)
        v = dst(e)
        if isfinite(dist[u]) && haskey(costs, e)
            if dist[u] + costs[e] + 1e-12 < dist[v]
                # Ciclo negativo detectado
                # Verificar se o ciclo é alcançável do source e alcança o target
                negative_cycle = true
                break
            end
        end
    end
    
    if negative_cycle
        # Verificar se o ciclo negativo afeta o caminho para o target
        # Se sim, retornar -Inf
        return (-Inf, Int[], Edge[])
    end
    
    # Reconstruir o caminho
    if !isfinite(dist[target]) || dist[target] == Inf
        return (Inf, Int[], Edge[])
    end
    
    path_nodes = Int[]
    path_edges = Edge[]
    u = target
    
    # Verificar se há ciclo no caminho (proteção contra loops infinitos)
    visited = falses(num_nodes)
    while u != 0
        if visited[u]
            # Ciclo detectado no caminho
            return (-Inf, Int[], Edge[])
        end
        visited[u] = true
        pushfirst!(path_nodes, u)
        if prev[u] != 0
            pushfirst!(path_edges, Edge(prev[u], u))
        end
        u = prev[u]
    end
    
    return (dist[target], path_nodes, path_edges)
end

"""
    has_negative_costs(costs::Dict{Edge, Float64})

Verifica se há custos negativos no dicionário de custos.
"""
function has_negative_costs(costs::Dict{Edge, Float64})
    for (e, cost) in costs
        if cost < -1e-12  # Tolerância numérica
            return true
        end
    end
    return false
end

"""
    edge_in_path(e::Edge, path::Vector{Edge})

Verifica se um arco está presente em um caminho.
"""
function edge_in_path(e::Edge, path::Vector{Edge})
    return e in path
end

"""
    paths_equal(path1::Vector{Edge}, path2::Vector{Edge})

Verifica se dois caminhos são iguais (mesmos arcos na mesma ordem).
"""
function paths_equal(path1::Vector{Edge}, path2::Vector{Edge})
    if length(path1) != length(path2)
        return false
    end
    for i in 1:length(path1)
        if path1[i] != path2[i]
            return false
        end
    end
    return true
end

"""
    solve_mcnfp_column_generation(inst::MCNFPInstance; max_iterations=1000, verbose=true, tolerance=1e-6)

Resolve o problema MCNFP usando geração de colunas (column generation) simplificado.

A abordagem usa formulação por caminhos:
- Modelo Mestre: Variáveis λ[p,k] representam o fluxo da commodity k no caminho p
- Inicia com variáveis dummy para garantir viabilidade
- Adiciona colunas enquanto custos reduzidos são negativos

IMPORTANTE: Esta versão usa Bellman-Ford quando há custos reduzidos negativos durante o pricing,
pois o Dijkstra não funciona corretamente com custos negativos. Os custos reduzidos podem ser
negativos mesmo quando os custos originais são positivos (devido aos multiplicadores duais).

# Argumentos
- `inst`: Instância do problema MCNFP
- `max_iterations`: Número máximo de iterações do algoritmo
- `verbose`: Se true, imprime log simplificado
- `tolerance`: Tolerância para considerar custo reduzido negativo

# Retorna
- `optimal`: true se encontrou solução ótima
- `objective_value`: Valor da função objetivo
- `solution`: Dicionário mapeando (commodity_id, path_id) -> fluxo
- `columns_generated`: Número total de colunas geradas
"""
function solve_mcnfp_column_generation(inst::MCNFPInstance; 
                                       max_iterations::Int=10000, 
                                       verbose::Bool=true,
                                       tolerance::Float64=1e-6)
    
    g = inst.graph
    commodities = inst.commodities
    num_k = length(commodities)
    
    # Estrutura para armazenar log completo
    log_lines = String[]
    
    function log_print(msgs...)
        msg = string(msgs...)
        push!(log_lines, msg)
        if verbose
            println(msg)
        end
    end
    
    function log_printf(fmt::String, args...)
        msg = Printf.format(Printf.Format(fmt), args...)
        push!(log_lines, msg)
        if verbose
            print(msg)
        end
    end
    
    # Estruturas para armazenar caminhos (colunas)
    # IMPORTANTE: Armazenar source e sink de cada caminho para modelagem correta
    paths = Vector{Vector{Edge}}[]
    path_sources = Vector{Int}[]  # path_sources[k][p] = source do caminho p da commodity k
    path_sinks = Vector{Int}[]    # path_sinks[k][p] = sink do caminho p da commodity k
    path_costs = Vector{Float64}[]
    for k in 1:num_k
        push!(paths, Vector{Edge}[])
        push!(path_sources, Int[])
        push!(path_sinks, Int[])
        push!(path_costs, Float64[])
    end
    
    # Função auxiliar BFS para encontrar caminhos
    function bfs_path(g::SimpleDiGraph, source::Int, target::Int)
        if source == target
            return (0.0, [source], Edge[])
        end
        queue = [source]
        prev = fill(0, nv(g))
        visited = falses(nv(g))
        visited[source] = true
        
        while !isempty(queue)
            u = popfirst!(queue)
            if u == target
                path_nodes = Int[]
                path_edges = Edge[]
                v = target
                while v != 0
                    pushfirst!(path_nodes, v)
                    if prev[v] != 0
                        pushfirst!(path_edges, Edge(prev[v], v))
                    end
                    v = prev[v]
                end
                return (length(path_edges), path_nodes, path_edges)
            end
            for neighbor in outneighbors(g, u)
                if !visited[neighbor]
                    visited[neighbor] = true
                    prev[neighbor] = u
                    push!(queue, neighbor)
                end
            end
        end
        return (Inf, Int[], Edge[])
    end
    
    # CORREÇÃO: Garantir que todas as commodities tenham pelo menos um caminho inicial
    # IMPORTANTE: Encontrar caminhos para TODOS os pares source-sink, não apenas o representativo
    log_print("\n=== Inicializando Caminhos Iniciais ===")
    for (k_idx, comm) in enumerate(commodities)
        # Construir dicionário de custos para esta commodity específica
        commodity_costs = Dict{Edge, Float64}()
        for e in edges(g)
            commodity_costs[e] = get(inst.costs, (comm.id, e), 1.0)
        end
        
        # Encontrar caminhos para TODOS os pares source-sink desta commodity
        paths_found = 0
        for source_node in keys(comm.sources)
            for sink_node in keys(comm.sinks)
                # Tentar encontrar caminho usando Dijkstra com custos desta commodity
                dist, _, path_edges = dijkstra_shortest_path(g, commodity_costs, source_node, sink_node)
                
                # Se não encontrou, tentar BFS
                if !isfinite(dist) || dist == Inf || isempty(path_edges)
                    dist, _, path_edges = bfs_path(g, source_node, sink_node)
                end
                
                if isfinite(dist) && dist < Inf && !isempty(path_edges)
                    # Verificar se este caminho já existe
                    path_exists = false
                    for existing_path in paths[k_idx]
                        if paths_equal(existing_path, path_edges)
                            path_exists = true
                            break
                        end
                    end
                    
                    if !path_exists
                        push!(paths[k_idx], path_edges)
                        push!(path_sources[k_idx], source_node)
                        push!(path_sinks[k_idx], sink_node)
                        cost = sum(get(inst.costs, (comm.id, e), 0.0) for e in path_edges)
                        push!(path_costs[k_idx], cost)
                        paths_found += 1
                        # Log todos os caminhos iniciais (limitado a 5 para não poluir)
                        if paths_found <= 5
                            log_printf("  Commodity %d: Caminho inicial %d -> %d (custo: %.2f)\n", 
                                      comm.id, source_node, sink_node, cost)
                        end
                    end
                end
            end
        end
        
        if paths_found == 0
            # Fallback: usar caminho do source/sink representativo
            dist, _, path_edges = dijkstra_shortest_path(g, commodity_costs, comm.source, comm.sink)
            if !isfinite(dist) || dist == Inf || isempty(path_edges)
                dist, _, path_edges = bfs_path(g, comm.source, comm.sink)
            end
            if isfinite(dist) && dist < Inf && !isempty(path_edges)
                push!(paths[k_idx], path_edges)
                push!(path_sources[k_idx], comm.source)
                push!(path_sinks[k_idx], comm.sink)
                cost = sum(get(inst.costs, (comm.id, e), 0.0) for e in path_edges)
                push!(path_costs[k_idx], cost)
                log_printf("  Commodity %d: Caminho inicial (fallback) %d -> %d (custo: %.2f)\n", 
                          comm.id, comm.source, comm.sink, cost)
            else
                log_printf("  ⚠️  Commodity %d: Sem caminho viável\n", comm.id)
            end
        end
    end
    
    # Verificar se todas as commodities têm caminhos
    commodities_without_paths = [k for k in 1:num_k if length(paths[k]) == 0]
    if !isempty(commodities_without_paths)
        log_print("\n⚠️  AVISO: Algumas commodities não têm caminhos iniciais.")
        log_print("  O modelo pode ser inviável devido a grafo desconexo.")
    end
    
    # Inicializar com variáveis dummy para garantir viabilidade
    # Cada commodity terá uma variável dummy com custo alto (M grande)
    M = 10000.0  # Penalidade para variáveis dummy
    
    iteration = 0
    total_columns = sum(length(p) for p in paths)
    best_bound = Inf  # Melhor limite dual (lower bound)
    
    log_print("\n=== Iniciando Geração de Colunas ===")
    log_printf("Colunas iniciais: %d\n", total_columns)
    
    while iteration < max_iterations
        iteration += 1
        
        # ===== RESOLVER MODELO MESTRE RESTRITO (RMP) =====
        master = Model(Gurobi.Optimizer)
        if !verbose
            set_silent(master)
        end
        
        # Variáveis: λ[p,k] = fluxo da commodity k no caminho p
        λ = Dict{Tuple{Int, Int}, JuMP.VariableRef}()
        dummy_vars = Dict{Int, JuMP.VariableRef}()  # Variáveis dummy por commodity
        
        # Criar variáveis para caminhos existentes
        for k in 1:num_k
            for p in 1:length(paths[k])
                λ[(k, p)] = @variable(master, base_name="λ[$k,$p]", lower_bound=0.0)
            end
            # Criar variável dummy se não há caminhos (para garantir viabilidade)
            if length(paths[k]) == 0
                dummy_vars[k] = @variable(master, base_name="dummy[$k]", lower_bound=0.0)
            end
        end
        
        # Restrições de convexidade por SOURCE (k,s): soma dos caminhos que começam em s = supply de s
        # IMPORTANTE: Criar restrições para TODOS os sources, mesmo se não há caminhos ainda
        source_constraints = Dict{Tuple{Int, Int}, JuMP.ConstraintRef}()  # (k_idx, source_node) -> constraint
        source_dummy_vars = Dict{Tuple{Int, Int}, JuMP.VariableRef}()  # (k_idx, source_node) -> dummy variable
        for (k_idx, comm) in enumerate(commodities)
            for source_node in keys(comm.sources)
                supply = comm.sources[source_node]
                # Somar todos os caminhos desta commodity que começam neste source
                terms = JuMP.VariableRef[]
                for p in 1:length(paths[k_idx])
                    if path_sources[k_idx][p] == source_node
                        if haskey(λ, (k_idx, p))
                            push!(terms, λ[(k_idx, p)])
                        end
                    end
                end
                # Criar restrição sempre, usando dummy se necessário
                if !isempty(terms)
                    expr = @expression(master, sum(terms))
                    source_constraints[(k_idx, source_node)] = @constraint(master, expr == supply)
                else
                    # Se não há caminhos ainda, usar variável dummy para garantir viabilidade
                    if !haskey(source_dummy_vars, (k_idx, source_node))
                        source_dummy_vars[(k_idx, source_node)] = @variable(master, base_name="source_dummy[$k_idx,$source_node]", lower_bound=0.0)
                    end
                    source_constraints[(k_idx, source_node)] = @constraint(master, source_dummy_vars[(k_idx, source_node)] == supply)
                end
            end
        end
        
        # Restrições de convexidade por SINK (k,t): soma dos caminhos que terminam em t = demanda de t
        # IMPORTANTE: Criar restrições para TODOS os sinks, mesmo se não há caminhos ainda
        sink_constraints = Dict{Tuple{Int, Int}, JuMP.ConstraintRef}()  # (k_idx, sink_node) -> constraint
        sink_dummy_vars = Dict{Tuple{Int, Int}, JuMP.VariableRef}()  # (k_idx, sink_node) -> dummy variable
        demand_duals = Dict{Tuple{Int, Int}, Float64}()  # (k_idx, sink_node) -> dual value
        for (k_idx, comm) in enumerate(commodities)
            for sink_node in keys(comm.sinks)
                demand = comm.sinks[sink_node]
                # Somar todos os caminhos desta commodity que terminam neste sink
                terms = JuMP.VariableRef[]
                for p in 1:length(paths[k_idx])
                    if path_sinks[k_idx][p] == sink_node
                        if haskey(λ, (k_idx, p))
                            push!(terms, λ[(k_idx, p)])
                        end
                    end
                end
                # Criar restrição sempre, usando dummy se necessário
                if !isempty(terms)
                    expr = @expression(master, sum(terms))
                    sink_constraints[(k_idx, sink_node)] = @constraint(master, expr == demand)
                else
                    # Se não há caminhos ainda, usar variável dummy para garantir viabilidade
                    if !haskey(sink_dummy_vars, (k_idx, sink_node))
                        sink_dummy_vars[(k_idx, sink_node)] = @variable(master, base_name="sink_dummy[$k_idx,$sink_node]", lower_bound=0.0)
                    end
                    sink_constraints[(k_idx, sink_node)] = @constraint(master, sink_dummy_vars[(k_idx, sink_node)] == demand)
                end
            end
        end
        
        # Manter restrições de demanda agregadas para compatibilidade (mas não usá-las se temos restrições por sink)
        demand_constraints = Dict{Int, JuMP.ConstraintRef}()
        for (k_idx, comm) in enumerate(commodities)
            if length(paths[k_idx]) > 0 && isempty(sink_constraints)  # Só usar se não temos restrições por sink
                expr = @expression(master, sum(λ[(k_idx, p)] for p in 1:length(paths[k_idx])))
                demand_constraints[k_idx] = @constraint(master, expr == comm.demand)
            else
                # Usar variável dummy (não contribui para capacidade, mas satisfaz demanda)
                if !haskey(dummy_vars, k_idx)
                    dummy_vars[k_idx] = @variable(master, base_name="dummy[$k_idx]", lower_bound=0.0)
                end
                demand_constraints[k_idx] = @constraint(master, dummy_vars[k_idx] == comm.demand)
            end
        end
        
        # Restrições de capacidade com variáveis de folga para garantir viabilidade
        capacity_constraints = Dict{Edge, JuMP.ConstraintRef}()
        slack_vars = Dict{Edge, JuMP.VariableRef}()  # Variáveis de folga por arco
        for e in edges(g)
            terms = JuMP.VariableRef[]
            for k in 1:num_k
                for p in 1:length(paths[k])
                    if haskey(λ, (k, p)) && edge_in_path(e, paths[k][p])
                        push!(terms, λ[(k, p)])
                    end
                end
            end
            # Criar variável de folga para garantir viabilidade
            slack_vars[e] = @variable(master, base_name="slack[$(src(e))->$(dst(e))]", lower_bound=0.0)
            if !isempty(terms)
                expr = @expression(master, sum(terms))
                capacity_constraints[e] = @constraint(master, expr - slack_vars[e] <= inst.capacities[e])
            else
                capacity_constraints[e] = @constraint(master, -slack_vars[e] <= inst.capacities[e])
            end
        end
        
        # Função objetivo (inclui penalidade para variáveis dummy e de folga)
        obj_terms = JuMP.VariableRef[]
        obj_coeffs = Float64[]
        
        for (k, p) in keys(λ)
            push!(obj_terms, λ[(k, p)])
            push!(obj_coeffs, path_costs[k][p])
        end
        
        for (k, var) in dummy_vars
            push!(obj_terms, var)
            push!(obj_coeffs, M)
        end
        
        # Adicionar penalidade para variáveis dummy de source e sink
        for ((k_idx, node), var) in source_dummy_vars
            push!(obj_terms, var)
            push!(obj_coeffs, M)
        end
        for ((k_idx, node), var) in sink_dummy_vars
            push!(obj_terms, var)
            push!(obj_coeffs, M)
        end
        
        # Adicionar penalidade para variáveis de folga (mas menor que dummy)
        slack_penalty = M / 10.0  # Penalidade menor que dummy, mas ainda alta
        for (e, var) in slack_vars
            push!(obj_terms, var)
            push!(obj_coeffs, slack_penalty)
        end
        
        if !isempty(obj_terms)
            obj_expr = @expression(master, sum(obj_coeffs[i] * obj_terms[i] for i in 1:length(obj_terms)))
            @objective(master, Min, obj_expr)
        else
            @objective(master, Min, 0.0)
        end
        
        optimize!(master)
        status = termination_status(master)
        
        if status != MOI.OPTIMAL
            log_printf("Iter %d: Modelo inviável ou não resolvido (status: %s)\n", iteration, status)
            
            # Primeiro, tentar encontrar caminhos para commodities sem caminhos
            paths_found_this_iter = 0
            for (k_idx, comm) in enumerate(commodities)
                if length(paths[k_idx]) == 0
                    dist, _, path_edges = bfs_path(g, comm.source, comm.sink)
                    if isfinite(dist) && dist < Inf && !isempty(path_edges)
                        push!(paths[k_idx], path_edges)
                        push!(path_sources[k_idx], comm.source)
                        push!(path_sinks[k_idx], comm.sink)
                        cost = sum(get(inst.costs, (comm.id, e), 0.0) for e in path_edges)
                        push!(path_costs[k_idx], cost)
                        total_columns += 1
                        paths_found_this_iter += 1
                        log_printf("  ➕ Caminho encontrado para commodity %d: %d -> %d\n", 
                                  comm.id, comm.source, comm.sink)
                    end
                end
            end
            
            if paths_found_this_iter > 0
                continue  # Reiniciar iteração com novos caminhos
            end
            
            # Com variáveis de folga, o modelo deve ser sempre viável agora
            # Mas se ainda está inviável, pode ser um problema mais fundamental
            # Continuar para próxima iteração (as variáveis de folga devem tornar viável)
            continue
            
            # Se não encontrou novos caminhos e modelo está inviável, adicionar diagnóstico
            if iteration == 1 || iteration == 10 || iteration % 100 == 0
                # Calcular diagnóstico detalhado
                arc_usage_estimate = Dict{Edge, Float64}()
                for e in edges(g)
                    arc_usage_estimate[e] = 0.0
                end
                
                for (k_idx, comm) in enumerate(commodities)
                    if length(paths[k_idx]) > 0
                        for e in paths[k_idx][1]
                            arc_usage_estimate[e] += comm.demand
                        end
                    end
                end
                
                overloaded_arcs = [e for e in edges(g) 
                                   if arc_usage_estimate[e] > inst.capacities[e] + 1e-6]
                
                if !isempty(overloaded_arcs)
                    log_printf("  📊 Diagnóstico: %d arcos sobrecarregados (exemplo: ", length(overloaded_arcs))
                    # Mostrar alguns exemplos
                    count = 0
                    for e in overloaded_arcs
                        if count < 3
                            log_printf("%d->%d (uso=%.1f, cap=%.1f) ", 
                                      src(e), dst(e), arc_usage_estimate[e], inst.capacities[e])
                            count += 1
                        end
                    end
                    log_print("...)")
                    
                    # Verificar se há caminhos alternativos disponíveis
                    commodities_using_overloaded = Int[]
                    for (k_idx, comm) in enumerate(commodities)
                        if length(paths[k_idx]) > 0
                            if any(e in overloaded_arcs for e in paths[k_idx][1])
                                push!(commodities_using_overloaded, k_idx)
                            end
                        end
                    end
                    log_printf("  📊 %d commodities usam arcos sobrecarregados\n", length(commodities_using_overloaded))
                else
                    log_print("  📊 Nenhum arco individualmente sobrecarregado (pode ser conflito combinado)")
                end
                
                if iteration == 1
                    log_print("  💡 Tentando encontrar caminhos alternativos para reduzir conflitos de capacidade...")
                end
            end
            
            continue
        end
        
        obj_value_with_penalties = objective_value(master)
        
        # Calcular valor objetivo REAL (sem penalidades de folga e dummy)
        # Apenas somar os custos dos caminhos usados
        real_obj_value = 0.0
        for (k, p) in keys(λ)
            flow = value(λ[(k, p)])
            # Incluir todos os fluxos, mesmo que pequenos (para precisão numérica)
            if flow > -tolerance  # Permitir valores ligeiramente negativos devido a erros numéricos
                real_obj_value += path_costs[k][p] * max(flow, 0.0)
            end
        end
        
        # Usar o valor real para comparações e relatórios
        obj_value = real_obj_value
        best_bound = min(best_bound, obj_value)
        
        # Verificar se há variáveis de folga sendo usadas
        slack_usage = Dict{Edge, Float64}()
        total_slack = 0.0
        for (e, var) in slack_vars
            slack_val = value(var)
            slack_usage[e] = slack_val
            total_slack += slack_val
        end
        
        if total_slack > 1e-6 && iteration <= 5
            log_printf("  ⚠️  Variáveis de folga ativas: total=%.2f (indicando conflitos de capacidade)\n", total_slack)
        end
        
        # ===== RESOLVER SUBPROBLEMAS DE PRICING =====
        # Obter multiplicadores duais das restrições de capacidade
        # Com variáveis de folga, o multiplicador dual reflete o custo de usar capacidade vs folga
        duals = Dict{Edge, Float64}()
        for e in edges(g)
            if haskey(capacity_constraints, e)
                dual_val = dual(capacity_constraints[e])
                duals[e] = isnan(dual_val) ? 0.0 : dual_val
            else
                duals[e] = 0.0
            end
        end
        
        # Obter multiplicadores duais das restrições de SOURCE (k,s)
        source_duals = Dict{Tuple{Int, Int}, Float64}()  # (k_idx, source_node) -> dual value
        for (k_idx, comm) in enumerate(commodities)
            for source_node in keys(comm.sources)
                if haskey(source_constraints, (k_idx, source_node))
                    dual_val = dual(source_constraints[(k_idx, source_node)])
                    source_duals[(k_idx, source_node)] = isnan(dual_val) ? 0.0 : dual_val
                else
                    source_duals[(k_idx, source_node)] = 0.0
                end
            end
        end
        
        # Obter multiplicadores duais das restrições de sink (demanda por sink)
        # Se não temos restrições por sink, usar restrições agregadas
        demand_duals = Dict{Tuple{Int, Int}, Float64}()  # (k_idx, sink_node) -> dual value
        for (k_idx, comm) in enumerate(commodities)
            for sink_node in keys(comm.sinks)
                if haskey(sink_constraints, (k_idx, sink_node))
                    dual_val = dual(sink_constraints[(k_idx, sink_node)])
                    demand_duals[(k_idx, sink_node)] = isnan(dual_val) ? 0.0 : dual_val
                else
                    demand_duals[(k_idx, sink_node)] = 0.0
                end
            end
            # Fallback: se não temos restrições por sink, usar restrição agregada
            if isempty(sink_constraints) && haskey(demand_constraints, k_idx)
                dual_val = dual(demand_constraints[k_idx])
                # Distribuir o dual igualmente entre os sinks (aproximação)
                for sink_node in keys(comm.sinks)
                    demand_duals[(k_idx, sink_node)] = (isnan(dual_val) ? 0.0 : dual_val) / length(comm.sinks)
                end
            end
        end
        
        # Calcular custos reduzidos e encontrar novos caminhos
        new_columns_found = false
        min_reduced_cost = Inf
        
        # Se há folgas ativas, tentar encontrar caminhos que reduzam o uso de folgas
        # Penalizando arcos que têm folgas ativas
        if total_slack > 1e-6 && iteration <= 20
            # Ajustar custos reduzidos para penalizar arcos com folgas ativas
            for e in edges(g)
                if haskey(slack_usage, e) && slack_usage[e] > 1e-6
                    # Penalizar arcos com folgas ativas para encorajar caminhos alternativos
                    # O multiplicador dual já reflete isso, mas podemos aumentar a penalidade
                    # Isso é feito implicitamente pelo dual, mas podemos reforçar
                end
            end
        end
        
        for (k_idx, comm) in enumerate(commodities)
            # Calcular custos reduzidos: c_reduzido[e] = c^k[e] - π[e]
            # O custo reduzido de um caminho p é: sum(c_reduzido[e] for e in p) - pi_k
            # onde c^k[e] é o custo da commodity k no arco e
            # π[e] é o multiplicador dual da restrição de capacidade
            # e pi_k é o multiplicador dual da restrição de demanda
            reduced_costs = Dict{Edge, Float64}()
            for e in edges(g)
                # Usar custo específico desta commodity
                commodity_cost = get(inst.costs, (comm.id, e), 0.0)
                reduced_costs[e] = commodity_cost - get(duals, e, 0.0)
            end
            
            # Verificar se há custos negativos (reduzidos podem ser negativos mesmo com custos originais positivos)
            has_negative = has_negative_costs(reduced_costs)
            
            # IMPORTANTE: Encontrar caminhos para TODOS os pares source-sink com custo reduzido negativo
            # Adicionar TODOS os caminhos com custo reduzido negativo, não apenas o melhor
            for source_node in keys(comm.sources)
                for sink_node in keys(comm.sinks)
                    # Encontrar caminho de menor custo reduzido
                    # Usar Bellman-Ford se houver custos negativos, caso contrário usar Dijkstra (mais eficiente)
                    if has_negative
                        dist, _, path_edges = bellman_ford_shortest_path(g, reduced_costs, source_node, sink_node)
                    else
                        dist, _, path_edges = dijkstra_shortest_path(g, reduced_costs, source_node, sink_node)
                    end
                    
                    if isfinite(dist) && dist < Inf && !isempty(path_edges)
                        # Custo reduzido do caminho = custo do caminho (com custos reduzidos dos arcos) 
                        # - multiplicador dual do source - multiplicador dual do sink
                        source_dual = get(source_duals, (k_idx, source_node), 0.0)
                        sink_dual = get(demand_duals, (k_idx, sink_node), 0.0)
                        reduced_cost_path = sum(reduced_costs[e] for e in path_edges) - source_dual - sink_dual
                        
                        min_reduced_cost = min(min_reduced_cost, reduced_cost_path)
                        
                        # Verificar se este caminho já existe
                        path_exists = false
                        for existing_path in paths[k_idx]
                            if paths_equal(existing_path, path_edges)
                                path_exists = true
                                break
                            end
                        end
                        
                        # Adicionar se custo reduzido é negativo OU se não há caminho ainda para este par source-sink
                        # IMPORTANTE: Adicionar caminhos para TODOS os pares source-sink, não apenas os com custo reduzido negativo
                        has_path_for_pair = false
                        for p in 1:length(paths[k_idx])
                            if path_sources[k_idx][p] == source_node && path_sinks[k_idx][p] == sink_node
                                has_path_for_pair = true
                                break
                            end
                        end
                        
                        if !path_exists && (reduced_cost_path < -tolerance || !has_path_for_pair)
                            push!(paths[k_idx], path_edges)
                            push!(path_sources[k_idx], source_node)
                            push!(path_sinks[k_idx], sink_node)
                            # Usar custo específico desta commodity
                            cost_path = sum(get(inst.costs, (comm.id, e), 0.0) for e in path_edges)
                            push!(path_costs[k_idx], cost_path)
                            new_columns_found = true
                            total_columns += 1
                            if verbose && iteration <= 10
                                log_printf("  ➕ Nova coluna para commodity %d: %d -> %d, custo=%.2f, custo_reduzido=%.4f\n", 
                                          comm.id, source_node, sink_node, cost_path, reduced_cost_path)
                            end
                        end
                    end
                end
            end
        end
        
        # Calcular gap (diferença relativa entre primal e dual)
        gap = Inf
        if status == MOI.OPTIMAL && isfinite(obj_value) && isfinite(min_reduced_cost)
            # Se min_reduced_cost >= 0, estamos no ótimo (gap = 0)
            if min_reduced_cost >= -tolerance
                gap = 0.0
            else
                # Gap relativo aproximado: quando há custo reduzido negativo,
                # o gap é aproximadamente o valor absoluto do custo reduzido mínimo
                # dividido pelo valor objetivo (gap relativo)
                gap = abs(min_reduced_cost) / max(abs(obj_value), 1.0)
            end
        end
        
        # Log simplificado
        gap_str = isfinite(gap) ? @sprintf("%.6f", gap) : "N/A"
        log_printf("Iter %d: Obj=%.4f, Gap=%s, Cols=%d, MinRC=%.4f\n", 
                   iteration, obj_value, gap_str, total_columns, min_reduced_cost)
        
        # Critério de parada: nenhuma coluna com custo reduzido negativo
        if !new_columns_found && min_reduced_cost >= -tolerance
            log_print("\n✅ Ótimo alcançado! Nenhuma coluna com custo reduzido negativo.")
            
            # Construir solução final e calcular valor objetivo REAL
            solution = Dict{Tuple{Int, Int}, Float64}()
            final_obj_real = 0.0
            for k in 1:num_k
                for p in 1:length(paths[k])
                    if haskey(λ, (k, p))
                        flow = value(λ[(k, p)])
                        if flow > tolerance
                            solution[(k, p)] = flow
                        end
                        # Calcular valor objetivo real incluindo todos os fluxos
                        if flow > -tolerance
                            final_obj_real += path_costs[k][p] * max(flow, 0.0)
                        end
                    end
                end
            end
            
            # Gerar arquivo de resumo com valor objetivo real
            generate_summary_file(inst, log_lines, true, final_obj_real, 
                                 solution, total_columns, iteration, paths, path_sources, path_sinks)
            
            return (optimal=true, objective_value=final_obj_real, solution=solution, columns_generated=total_columns)
        end
    end
    
    # Máximo de iterações atingido
    log_print("\n⚠️  Número máximo de iterações atingido.")
    
    # Resolver modelo final
    log_print("\n=== Resolvendo Modelo Final ===")
    master_final = Model(Gurobi.Optimizer)
    if !verbose
        set_silent(master_final)
    end
    
    λ_final = Dict{Tuple{Int, Int}, JuMP.VariableRef}()
    for k in 1:num_k
        for p in 1:length(paths[k])
            λ_final[(k, p)] = @variable(master_final, base_name="λ_final[$k,$p]", lower_bound=0.0)
        end
    end
    
    # Restrições de convexidade por SOURCE (k,s)
    for (k_idx, comm) in enumerate(commodities)
        for source_node in keys(comm.sources)
            supply = comm.sources[source_node]
            terms = JuMP.VariableRef[]
            for p in 1:length(paths[k_idx])
                if path_sources[k_idx][p] == source_node
                    if haskey(λ_final, (k_idx, p))
                        push!(terms, λ_final[(k_idx, p)])
                    end
                end
            end
            if !isempty(terms)
                @constraint(master_final, sum(terms) == supply)
            end
        end
    end
    
    # Restrições de convexidade por SINK (k,t)
    for (k_idx, comm) in enumerate(commodities)
        for sink_node in keys(comm.sinks)
            demand = comm.sinks[sink_node]
            terms = JuMP.VariableRef[]
            for p in 1:length(paths[k_idx])
                if path_sinks[k_idx][p] == sink_node
                    if haskey(λ_final, (k_idx, p))
                        push!(terms, λ_final[(k_idx, p)])
                    end
                end
            end
            if !isempty(terms)
                @constraint(master_final, sum(terms) == demand)
            end
        end
    end
    
    # Adicionar variáveis de folga também no modelo final para garantir viabilidade
    slack_final = Dict{Edge, JuMP.VariableRef}()
    for e in edges(g)
        slack_final[e] = @variable(master_final, base_name="slack_final[$(src(e))->$(dst(e))]", lower_bound=0.0)
    end
    
    for e in edges(g)
        terms = JuMP.VariableRef[]
        for k in 1:num_k
            for p in 1:length(paths[k])
                if haskey(λ_final, (k, p)) && edge_in_path(e, paths[k][p])
                    push!(terms, λ_final[(k, p)])
                end
            end
        end
        if !isempty(terms)
            expr = @expression(master_final, sum(terms))
            @constraint(master_final, expr - slack_final[e] <= inst.capacities[e])
        else
            @constraint(master_final, -slack_final[e] <= inst.capacities[e])
        end
    end
    
    # Função objetivo: minimizar custo real (sem penalidade de folga no modelo final)
    # Mas ainda incluímos folgas para garantir viabilidade
    slack_penalty_final = 1000.0  # Penalidade menor no modelo final
    if !isempty(λ_final)
        obj_expr = @expression(master_final,
            sum(path_costs[k][p] * λ_final[(k, p)] 
                for (k, p) in keys(λ_final)) +
            sum(slack_penalty_final * slack_final[e] for e in keys(slack_final))
        )
        @objective(master_final, Min, obj_expr)
    else
        @objective(master_final, Min, sum(slack_penalty_final * slack_final[e] for e in keys(slack_final)))
    end
    
    optimize!(master_final)
    final_status = termination_status(master_final)
    final_obj_with_penalties = final_status == MOI.OPTIMAL ? objective_value(master_final) : Inf
    
    # Calcular valor objetivo REAL (sem penalidades de folga)
    final_obj_real = 0.0
    solution = Dict{Tuple{Int, Int}, Float64}()
    if final_status == MOI.OPTIMAL
        for k in 1:num_k
            for p in 1:length(paths[k])
                if haskey(λ_final, (k, p))
                    flow = value(λ_final[(k, p)])
                    if flow > tolerance
                        solution[(k, p)] = flow
                    end
                    # Calcular valor objetivo real incluindo todos os fluxos
                    if flow > -tolerance
                        final_obj_real += path_costs[k][p] * max(flow, 0.0)
                    end
                end
            end
        end
    end
    
    if final_status == MOI.OPTIMAL
        log_printf("✅ Modelo final resolvido com sucesso. Objetivo (real): %.4f\n", final_obj_real)
    else
        log_printf("❌ Modelo final não resolvido. Status: %s\n", final_status)
    end
    
    # Gerar arquivo de resumo com valor objetivo real
    generate_summary_file(inst, log_lines, false, final_obj_real, 
                         solution, total_columns, iteration, paths, path_sources, path_sinks; final_status=final_status)
    
    return (optimal=false, objective_value=final_obj_real, solution=solution, columns_generated=total_columns)
end

"""
    generate_summary_file(inst, log_lines, optimal, obj_value, solution, columns_generated, iterations, paths, path_sources, path_sinks; final_status=nothing)

Gera um arquivo TXT com o resumo da execução do algoritmo CGA.
"""
function generate_summary_file(inst::MCNFPInstance, log_lines::Vector{String},
                               optimal::Bool, obj_value::Float64, 
                               solution::Dict{Tuple{Int, Int}, Float64},
                               columns_generated::Int, iterations::Int,
                               paths::Vector{Vector{Vector{Edge}}},
                               path_sources::Vector{Vector{Int}},
                               path_sinks::Vector{Vector{Int}};
                               final_status=nothing)
    
    filename = "CGA_Summary.txt"
    open(filename, "w") do file
        # Cabeçalho
        println(file, "="^80)
        println(file, "RESUMO DA EXECUÇÃO DO ALGORITMO DE GERAÇÃO DE COLUNAS (CGA)")
        println(file, "="^80)
        println(file, "")
        
        # Resumo da instância usando print_instance_summary
        println(file, "=== RESUMO DA INSTÂNCIA ===")
        println(file, "Nós: ", nv(inst.graph))
        println(file, "Arcos: ", ne(inst.graph))
        println(file, "Mercadorias: ", length(inst.commodities))
        println(file, "-"^30)
        println(file, "")
        
        println(file, "Mercadorias:")
        for c in inst.commodities
            println(file, Printf.format(Printf.Format("  ID %d: %d -> %d (Demanda: %.2f)"), c.id, c.source, c.sink, c.demand))
        end
        println(file, "")
        
        println(file, "-"^30)
        println(file, "Arcos (Amostra - primeiros 10):")
        count = 0
        for e in edges(inst.graph)
            count += 1
            cap = inst.capacities[e]
            # Mostrar custos para todas as commodities
            cost_strs = String[]
            for comm in inst.commodities
                cost = get(inst.costs, (comm.id, e), 0.0)
                push!(cost_strs, "k$(comm.id)=$(cost)")
            end
            costs_display = join(cost_strs, ", ")
            println(file, Printf.format(Printf.Format("  %d -> %d | Cap: %.2f | Custos: %s"), src(e), dst(e), cap, costs_display))
            if count >= 10 break end
        end
        println(file, "")
        println(file, "="^80)
        println(file, "")
        
        # Log completo da execução
        println(file, "=== LOG COMPLETO DA EXECUÇÃO ===")
        println(file, "")
        for line in log_lines
            println(file, line)
        end
        println(file, "")
        println(file, "="^80)
        println(file, "")
        
        # Resumo final
        println(file, "=== RESULTADO FINAL ===")
        if optimal
            println(file, "✅ Solução ÓTIMA encontrada!")
        else
            if final_status !== nothing
                println(file, "⚠️  Solução não ótima (Status: $final_status)")
            else
                println(file, "⚠️  Solução não ótima (iterações máximas atingidas)")
            end
        end
        println(file, Printf.format(Printf.Format("Valor objetivo: %.4f"), obj_value))
        println(file, Printf.format(Printf.Format("Total de iterações: %d"), iterations))
        println(file, Printf.format(Printf.Format("Total de colunas geradas: %d"), columns_generated))
        println(file, "")
        
        if !isempty(solution)
            println(file, "Solução (fluxo por caminho):")
            for ((k, p), flow) in solution
                source = p <= length(path_sources[k]) ? path_sources[k][p] : 0
                sink = p <= length(path_sinks[k]) ? path_sinks[k][p] : 0
                path_str = if p <= length(paths[k]) && !isempty(paths[k][p])
                    join([Printf.format(Printf.Format("%d->%d"), src(e), dst(e)) for e in paths[k][p]], "->")
                else
                    "N/A"
                end
                println(file, Printf.format(Printf.Format("  Commodity %d, Caminho %d (%d->%d): fluxo = %.4f, rota: %s"), 
                          k, p, source, sink, flow, path_str))
            end
            println(file, "")
            
            # Calcular e mostrar fluxos por arco
            println(file, "Fluxos por arco (agregados por commodity):")
            arc_flows = Dict{Tuple{Int, Edge}, Float64}()
            
            # Inicializar todos os arcos com fluxo zero para todas as commodities
            for e in edges(inst.graph)
                for comm in inst.commodities
                    arc_flows[(comm.id, e)] = 0.0
                end
            end
            
            # Calcular fluxos agregados por arco
            for ((k, p), flow) in solution
                if k <= length(paths) && p <= length(paths[k]) && !isempty(paths[k][p])
                    for e in paths[k][p]
                        if haskey(arc_flows, (k, e))
                            arc_flows[(k, e)] += flow
                        end
                    end
                end
            end
            
            # Agrupar por arco e mostrar
            for e in edges(inst.graph)
                flow_strs = String[]
                for comm in inst.commodities
                    flow_val = get(arc_flows, (comm.id, e), 0.0)
                    if flow_val > 1e-6  # Apenas mostrar fluxos significativos
                        push!(flow_strs, Printf.format(Printf.Format("k%d=%.2f"), comm.id, flow_val))
                    end
                end
                if !isempty(flow_strs)
                    flows_display = join(flow_strs, ", ")
                    cap = inst.capacities[e]
                    println(file, Printf.format(Printf.Format("  %d -> %d | Cap: %.2f | Fluxos: %s"), 
                              src(e), dst(e), cap, flows_display))
                end
            end
        else
            println(file, "Nenhuma solução encontrada.")
        end
        println(file, "")
        println(file, "="^80)
    end
    
    println("\n📄 Arquivo de resumo gerado: $filename")
end


# 1. Gerar uma instância (vamos forçar uma seed que sabemos que pode ser difícil ou fácil)
# Densidade baixa (0.2) aumenta chance de inviabilidade (grafo desconexo)
println("Gerando instância...")
instancia = generate_mcnfp_instance(
    num_nodes=100, 
    num_commodities=10, 
    density=1.0, 
    seed=800,      # Mude o seed para testar diferentes cenários
    max_cap=100.0, 
    max_demand=10.0
)

print_instance_summary(instancia)

# 2. Checar a viabilidade com LP
is_possible = check_feasibility_lp(instancia)

# 3. (Opcional) Teste de sanidade
if is_possible
    println("Podemos prosseguir com algoritmos mais complexos.")
else
    println("Dica: Tente aumentar a densidade do grafo ou as capacidades.")
end

# --- Exemplo de Uso da Geração de Colunas ---
# Descomente o código abaixo para testar a função de geração de colunas

# println("\n" * "="^50)
# println("=== TESTE DE GERAÇÃO DE COLUNAS ===")
# println("="^50)
# 
# # Gerar uma instância menor para teste
# instancia_cg = generate_mcnfp_instance(
#     num_nodes=10, 
#     num_commodities=3, 
#     density=0.5, 
#     seed=42,
#     max_cap=100.0, 
#     max_demand=20.0
# )
# 
# print_instance_summary(instancia_cg)
# 
# # Resolver usando geração de colunas
# result = solve_mcnfp_column_generation(instancia_cg; verbose=true, max_iterations=100)
# 
# println("\n=== RESULTADO FINAL ===")
# if result.optimal
#     @printf("✅ Solução ÓTIMA encontrada!\n")
#     @printf("Valor objetivo: %.4f\n", result.objective_value)
#     @printf("Total de colunas geradas: %d\n", result.columns_generated)
#     println("\nSolução (fluxo por caminho):")
#     for ((k, p), flow) in result.solution
#         @printf("  Commodity %d, Caminho %d: fluxo = %.4f\n", k, p, flow)
#     end
# else
#     @printf("⚠️  Solução não ótima (iterações máximas atingidas)\n")
#     @printf("Valor objetivo: %.4f\n", result.objective_value)
#     @printf("Total de colunas geradas: %d\n", result.columns_generated)
# end
# end