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

    return mst_edges, total_cos