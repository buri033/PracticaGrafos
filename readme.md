# 🏭 Optimización de Rutas Logísticas con Grafos y Backtracking

## 📘 Descripción del Proyecto
Este proyecto implementa una solución algorítmica para la optimización de rutas de entrega en una red logística.  
El objetivo principal es determinar el camino más eficiente para un vehículo que debe visitar todas las sedes de la empresa, minimizando el costo total de operación.  
El proyecto utiliza la **Teoría de Grafos** para modelar la red de sedes (nodos) y rutas (aristas con costos).  
Se resuelven dos objetivos de optimización distintos:

### 🔹 Árbol de Expansión Mínimo (MST)
- Resuelve el problema de conectividad, encontrando la red de rutas más barata para garantizar que todas las sedes estén conectadas.  
- Se utiliza el **Algoritmo de Prim**.

### 🔹 Camino Hamiltoniano Mínimo
- Resuelve el problema de secuencia de viaje, encontrando la ruta más corta que visita todas las sedes sin necesidad de regresar al punto de partida (Depot).  
- Se utiliza un enfoque de **Backtracking** para explorar todas las combinaciones posibles.

---

## ⚙️ Cómo Ejecutar el Proyecto

### Requisitos
- Python 3.x.

### Ejecución
El código se encuentra en un único archivo Python.  
Para ejecutarlo, simplemente abre el archivo en tu editor o entorno de desarrollo (IDE) y ejecuta el script.  
El programa generará la siguiente salida en la consola:
- La **Matriz de Costos** del grafo completo.  
- El resultado de la **Red Mínima (MST)**.  
- La **Ruta Óptima de Entrega** y su **costo mínimo total**.

---

## 💡 Supuestos Asumidos
Para modelar y ejecutar el problema de manera efectiva, se asumieron los siguientes puntos:

- **Grafo Completo:** Se asume que existe una conexión directa entre cada par de sedes (Depot a todas las Sedes, y cada Sede entre sí).  
  Esta conectividad es fundamental para que el algoritmo de Backtracking explore todas las secuencias válidas.  
  Los costos de las rutas faltantes fueron estimados para completar la red.

- **Costo de la Ruta:** El costo de cada arista (ruta) es simétrico (no dirigido), lo que implica que el costo de A a B es igual al costo de B a A.  
  El costo se calcula como:  
<img width="197" height="31" alt="image" src="https://github.com/user-attachments/assets/d9c553b2-efbe-41ac-9379-8455895b7c6b" />


- **Objetivo de Entrega (Camino Mínimo):**  
  Se asume que la ruta óptima debe visitar todas las sedes, pero **NO** es obligatorio regresar al Depot.  
  Por lo tanto, el costo final del Objetivo 3 (Backtracking) solo incluye la suma de los tramos de entrega, sin añadir el costo de la vuelta a la base.


## 🎭 Equipo:
- Tomás Buriticá Jaramillo
- Juan Esteban Vallejo Hincapié



