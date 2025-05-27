# O Problema do Caixeiro Viajante (TSP)

O problema do caixeiro viajante (TSP) é encontrar o caminho de menor custo que visita um conjunto de (n) cidades exatamente uma vez e retorna à cidade inicial. Suponha que temos uma matriz de distâncias (D) onde (D[i][j]) representa a distância entre a cidade (i) e a cidade (j).

---

## Estrutura Modular

- modulo.py → São carregadas as bibliotecas, banco de dados/coordenadas das localidades. Além de plotar os pontos no mapa da região.
- held-karp.ipynb → Aplicação do algoritmo exato Hled-Karp para obtenção da melhor rota.
- held-karp_cluster.ipynb → Aplicação do algoritmo exato Hled-Karp para obtenção da melhor rota com o inclemento da clusterização das localidades.
- formigas.ipynb → Aplicação da heuristica colônia de formigas para obtenção da melhor rota.
- held-karp_cluster.ipynb → Aplicação da heurística colônia de formigas para obtenção da melhor rota com o inclemento da clusterização das localidades.

---

## Algoritmos utilizados na solução

### ✅ 1. Held Karp
- A ideia principal do algoritmo Held-Karp é usar programação dinâmica para evitar recalcular subproblemas repetidos. Em vez de calcular todas as permutações possíveis das cidades (força bruta), armazenamos as soluções parciais para subconjuntos de cidades e as reutilizamos.

### ✅ 2. Algoritmo de Colônia de Formigas (ACO)
- O método colônia de formigas (Ant Colony Optimization - ACO) é uma meta-heurística inspirada no comportamento de forrageamento das formigas. Esta técnica é amplamente utilizada para resolver problemas de otimização combinatória, como o problema do caixeiro viajante (Travelling Salesman Problem - TSP).
- 
### ✅ 3. Clusterização
- A clusterização, também conhecida como agrupamento ou clustering, é uma técnica de análise de dados que visa organizar elementos semelhantes em grupos (clusters). Essa técnica é utilizada em diversas áreas, como ciência de dados, marketing, e computação, para identificar padrões e estruturas subjacentes em grandes conjuntos de dados, a partir de determinado parâmetro chave utilizado para diferenciar os itens.
