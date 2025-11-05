# Se importan las bibliotecas necesarias
import sys
import math
from typing import Any, List, Tuple


# --- 1. Modelado del Problema: Grafo Ponderado COMPLETO (Objetivo 1) ---

class WeightedGraphMatrix:
    """
    Representa el mapa de la ciudad como un grafo ponderado usando
    una Matriz de Adyacencia. Almacena los COSTOS (pesos) de la ruta.
    """

    def __init__(self):
        # 'adj_matrix' almacenará los costos. Ej: matriz[A][B] = 25.0
        self.adj_matrix: List[List[float]] = []
        # 'nodes' es una lista que mapea un nombre (ej: 'Sede A') a un índice (ej: 1)
        self.nodes: List[Any] = []
        # 'size' es el número total de nodos (sedes)
        self.size: int = 0

    def get_index(self, value: Any) -> int:
        """Función auxiliar para obtener el índice (posición) de un nodo por su nombre."""
        try:
            return self.nodes.index(value)
        except ValueError:
            return -1

    def add_vertex(self, value: Any) -> None:
        """Añade un vértice (una sede) al grafo, expandiendo la matriz."""
        if value in self.nodes:
            return

        self.nodes.append(value)
        self.size += 1

        # Añadir nueva COLUMNA a todas las filas
        for row in self.adj_matrix:
            row.append(0.0)

        # Añadir nueva FILA completa
        self.adj_matrix.append([0.0] * self.size)

    def add_edge(self, vertex_1: Any, vertex_2: Any, cost: float, directed: bool = False):
        """
        Añade una arista (una ruta) con un costo específico.
        Si el costo es 0, significa que la ruta no existe o es insignificante.
        """
        if vertex_1 not in self.nodes:
            self.add_vertex(vertex_1)
        if vertex_2 not in self.nodes:
            self.add_vertex(vertex_2)

        pos_v1 = self.get_index(vertex_1)
        pos_v2 = self.get_index(vertex_2)

        # Asignamos el costo en la matriz: matriz[pos_v1][pos_v2] = costo
        self.adj_matrix[pos_v1][pos_v2] = cost

        if not directed:
            # Si la ruta es de doble vía (no dirigida), asignamos el mismo costo
            self.adj_matrix[pos_v2][pos_v1] = cost

    def print_matrix(self):
        """Función auxiliar para imprimir la matriz de costos de forma legible."""
        if self.size == 0:
            print("Grafo vacío")
            return

        header = "       " + " | ".join([f"{str(node)[:5]:^5}" for node in self.nodes])
        print(header)
        print("-" * len(header))

        for i in range(self.size):
            row_header = f"{str(self.nodes[i])[:5]:>5} |"
            row_data = " | ".join([f"{cost:^5.1f}" for cost in self.adj_matrix[i]])
            print(f"{row_header} {row_data}")


# --- 2. Algoritmo de Expansión Mínimo (MST) - Prim (Objetivo 2) ---
# Se mantiene sin cambios, ya que su objetivo es la conexión de la red.

class MSTFinder:
    """
    Implementa el Algoritmo de Prim para encontrar el Árbol de Expansión Mínimo (MST).
    (Red de conexión más barata).
    """

    def __init__(self, graph: WeightedGraphMatrix):
        self.graph = graph
        self.size = graph.size
        self.nodes = graph.nodes
        self.adj_matrix = graph.adj_matrix

    def _get_min_key_vertex(self, key_costs: List[float], mst_set: List[bool]) -> int:
        """Función auxiliar de Prim: Encuentra el vértice con el costo mínimo que aún no está en el MST."""
        min_cost = math.inf
        min_index = -1
        for v in range(self.size):
            if key_costs[v] < min_cost and not mst_set[v]:
                min_cost = key_costs[v]
                min_index = v
        return min_index

    def find_prim_mst(self) -> Tuple[List[Tuple[Any, Any, float]], float]:
        """Implementación del Algoritmo de Prim."""
        if self.size == 0:
            return [], 0.0

        key_costs = [math.inf] * self.size
        parent = [None] * self.size
        mst_set = [False] * self.size

        key_costs[0] = 0.0
        parent[0] = -1

        mst_edges = []
        total_cost = 0.0

        for _ in range(self.size):
            u_idx = self._get_min_key_vertex(key_costs, mst_set)

            if u_idx == -1: break

            mst_set[u_idx] = True

            for v_idx in range(self.size):
                cost = self.adj_matrix[u_idx][v_idx]

                # Lógica central: si es una arista válida y el costo es mejor que el actual...
                if 0 < cost < key_costs[v_idx] and not mst_set[v_idx]:
                    key_costs[v_idx] = cost
                    parent[v_idx] = u_idx

        for i in range(1, self.size):
            if parent[i] is not None and parent[i] != -1:
                u_node = self.nodes[parent[i]]
                v_node = self.nodes[i]
                cost = self.adj_matrix[parent[i]][i]
                mst_edges.append((u_node, v_node, cost))
                total_cost += cost

        return mst_edges, total_cost


# --- 3. Aplicación de Backtracking - Camino Mínimo (Objetivo 3) ---

class TSPBacktracker:
    """
    Resuelve el Problema del Camino Hamiltoniano Mínimo usando Backtracking.
    Busca la ruta más barata que VISITE TODAS las sedes sin regresar al Depot.
    """

    def __init__(self, graph: WeightedGraphMatrix):
        self.graph = graph
        self.size = graph.size
        self.nodes = graph.nodes
        self.adj_matrix = graph.adj_matrix

        self.min_cost = math.inf
        self.best_path = []
        self.visited = [False] * self.size

    def find_optimal_route(self, start_node: Any) -> Tuple[List[Any], float]:
        """Función pública que inicia la búsqueda de backtracking."""
        start_index = self.graph.get_index(start_node)
        if start_index == -1:
            return [], 0.0

        self.visited[start_index] = True
        current_path = [start_node]

        self._backtrack_helper(start_index, current_path, 0.0)

        return self.best_path, self.min_cost

    def _backtrack_helper(self, u_idx: int, current_path: List[Any], current_cost: float):
        """
        Función RECURSIVA de Backtracking.
        """

        # --- CASO BASE (FINAL DEL CAMINO) ---
        # Se cumple si hemos visitado todas las sedes (la ruta tiene el tamaño completo).
        if len(current_path) == self.size:

            # *** MODIFICACIÓN CLAVE: No se añade el costo de regreso al Depot. ***
            total_cost = current_cost

            # Si el costo de esta ruta (sin regreso) es el mejor hasta ahora...
            if total_cost < self.min_cost:
                self.min_cost = total_cost
                # Guardamos la ruta. Usamos .copy() para evitar que se modifique después.
                self.best_path = current_path.copy()

            return  # Terminamos esta rama

        # --- PODA (OPTIMIZACIÓN) ---
        # Si el costo acumulado de esta ruta ya es PEOR que la mejor ruta final encontrada...
        if current_cost >= self.min_cost:
            return  # Detenemos la exploración y retrocedemos (Backtrack)

        # --- PASO RECURSIVO (EXPLORACIÓN) ---
        # Iterar sobre todos los posibles *siguientes* destinos 'v'
        for v_idx in range(self.size):
            cost_u_v = self.adj_matrix[u_idx][v_idx]

            # Si hay una ruta Y aún no hemos visitado 'v'...
            if cost_u_v > 0 and not self.visited[v_idx]:
                # 1. "TOMAR DECISIÓN" (Avanzar)
                self.visited[v_idx] = True
                v_node_name = self.nodes[v_idx]
                current_path.append(v_node_name)

                # 2. "EXPLORAR" (Recursión)
                # Llamarse a sí misma, con el costo actualizado
                self._backtrack_helper(v_idx, current_path, current_cost + cost_u_v)

                # 3. "DESHACER DECISIÓN" (Backtrack)
                # Al regresar, se deshace la visita para que el bucle pueda probar otras opciones
                current_path.pop()
                self.visited[v_idx] = False


# ====================================================================
# --- EJECUCIÓN DEL ESCENARIO CON GRAFO COMPLETO ---
# ====================================================================

print("🚚 Iniciando Simulación de Entrega de Comidas (Camino Mínimo)")
print("=" * 50)

COSTO_POR_KM = 2.5

g = WeightedGraphMatrix()
sedes = ['Depot', 'Sede A', 'Sede B', 'Sede C', 'Sede D', 'Sede E']
for sede in sedes:
    g.add_vertex(sede)

# --- 1. CONEXIONES ORIGINALES (9 RUTAS) ---
# Estas son las rutas que definían el mapa inicial.
g.add_edge('Depot', 'Sede A', 10 * COSTO_POR_KM)  # 25.0
g.add_edge('Depot', 'Sede B', 15 * COSTO_POR_KM)  # 37.5
g.add_edge('Depot', 'Sede C', 30 * COSTO_POR_KM)  # 75.0

g.add_edge('Sede A', 'Sede B', 8 * COSTO_POR_KM)  # 20.0
g.add_edge('Sede A', 'Sede D', 20 * COSTO_POR_KM)  # 50.0
g.add_edge('Sede B', 'Sede C', 12 * COSTO_POR_KM)  # 30.0
g.add_edge('Sede B', 'Sede D', 18 * COSTO_POR_KM)  # 45.0
g.add_edge('Sede C', 'Sede E', 5 * COSTO_POR_KM)  # 12.5
g.add_edge('Sede D', 'Sede E', 22 * COSTO_POR_KM)  # 55.0

# --- 2. CONEXIONES FALTANTES PARA EL GRAFO COMPLETO (6 RUTAS) ---
# Estas se añaden para asegurar que el Backtracking pruebe todas las rutas posibles
# y que el regreso sea posible desde cualquier punto (aunque el costo de regreso se omita).
print("💡 Se han añadido las siguientes 6 conexiones para crear un Grafo Completo.")
g.add_edge('Depot', 'Sede D', 20 * COSTO_POR_KM)  # Costo 50.0 (Depot a D)
g.add_edge('Depot', 'Sede E', 22 * COSTO_POR_KM)  # Costo 55.0 (Depot a E)
g.add_edge('Sede A', 'Sede C', 25 * COSTO_POR_KM)  # Costo 62.5
g.add_edge('Sede A', 'Sede E', 30 * COSTO_POR_KM)  # Costo 75.0
g.add_edge('Sede B', 'Sede E', 17 * COSTO_POR_KM)  # Costo 42.5
g.add_edge('Sede C', 'Sede D', 25 * COSTO_POR_KM)  # Costo 62.5

print("\n🗺️  Mapa de Costos Completo (Matriz de Adyacencia):")
g.print_matrix()
print("\n" + "=" * 50)

# --- 3. Aplicación de Árbol de Expansión Mínimo (Objetivo 2) ---
print("🌐 Objetivo 2: Árbol de Expansión Mínimo (Algoritmo de Prim)")
print("   (Costo mínimo para *conectar* todas las sedes)")

mst_finder = MSTFinder(g)
mst_edges, mst_total_cost = mst_finder.find_prim_mst()

for u, v, cost in mst_edges:
    print(f"   - Conectar {u} <-> {v} (Costo: ${cost:.2f})")
print(f"\n   Costo Total de la Red Mínima (MST): ${mst_total_cost:.2f}")
print("\n" + "=" * 50)

# --- 4. Aplicación de Backtracking (Objetivo 3) ---
print("🚛 Objetivo 3: Ruta de Entrega Óptima (Backtracking - Camino Mínimo)")
print("   *** NOTA: Se omite el costo de regreso al Depot ***")

tsp_solver = TSPBacktracker(g)
best_path, min_cost = tsp_solver.find_optimal_route('Depot')

if min_cost == math.inf:
    print("   No se encontró una ruta válida que visite todas las sedes.")
else:
    path_str = " -> ".join(best_path)
    print(f"   Ruta Óptima Encontrada (6 paradas): {path_str}")
    print(f"   Costo Total de la Ruta (Sin regreso): ${min_cost:.2f}")

print("\n" + "=" * 50)
print("🏁 Simulación Finalizada.")
