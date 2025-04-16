
import time
import psutil
import heapq
from collections import deque
import os

MAZE_FILE = r"C:\Users\João.Neves\Downloads\labirinto1_pronto\maze1.txt"

def load_maze(path):
    with open(path, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    maze = [list(line) for line in lines]
    start, end = None, None
    for y, row in enumerate(maze):
        for x, val in enumerate(row):
            if val == 'S':
                start = (y, x)
            elif val == 'E':
                end = (y, x)
    return maze, start, end

def get_neighbors(pos, maze):
    y, x = pos
    neighbors = []
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]):
            if maze[ny][nx] in [' ', 'E']:
                neighbors.append((ny, nx))
    return neighbors

def reconstruct_path(came_from, end):
    path = []
    curr = end
    while curr in came_from:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path

def measure(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        end_mem = process.memory_info().rss
        mem_used = end_mem - start_mem
        return result, elapsed, mem_used
    return wrapper

@measure
def bfs(maze, start, end):
    queue = deque([start])
    came_from = {}
    visited = set([start])
    while queue:
        current = queue.popleft()
        if current == end:
            return reconstruct_path(came_from, end)
        for neighbor in get_neighbors(current, maze):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
    return []

@measure
def dfs(maze, start, end):
    stack = [start]
    came_from = {}
    visited = set([start])
    while stack:
        current = stack.pop()
        if current == end:
            return reconstruct_path(came_from, end)
        for neighbor in get_neighbors(current, maze):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
    return []

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

@measure
def astar(maze, start, end):
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            return reconstruct_path(came_from, end)
        for neighbor in get_neighbors(current, maze):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                priority = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current
    return []

@measure
def greedy(maze, start, end):
    open_set = [(heuristic(start, end), start)]
    came_from = {}
    visited = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            return reconstruct_path(came_from, end)
        visited.add(current)
        for neighbor in get_neighbors(current, maze):
            if neighbor not in visited:
                heapq.heappush(open_set, (heuristic(neighbor, end), neighbor))
                came_from[neighbor] = current
    return []

if __name__ == "__main__":
    maze, start, end = load_maze(MAZE_FILE)

    print("Executando algoritmos no maze1.txt:")

    for name, algo in [("BFS", bfs), ("DFS", dfs), ("A*", astar), ("Gulosa", greedy)]:
        path, tempo, memoria = algo(maze, start, end)
        print(f"{name}: Tempo = {tempo:.6f}s | Memória = {memoria/1024:.2f} KB | Passos = {len(path)}")
