import numpy as np

def generate_mesh(n: int = 50):
    """Generate a regular triangular mesh on [0,1]^2 square domain with hexagonal patterns.
    Each hexagon is composed of six equilateral triangles.
    
    Args:
        n (int): Number of hexagons along each axis (default: 50)
        
    Returns:
        tuple: (vertices, triangles, edges)
            - vertices: (N, 3) array of 3D vertex coordinates
            - triangles: (M, 3) array of vertex indices forming triangles
            - edges: (K, 2) array of vertex indices forming edges
    """
    # Calculate the spacing for equilateral triangles
    dx = 1.0 / n
    dy = dx * np.sqrt(3) / 2.0
    
    # Calculate number of rows needed
    n_rows = int(1.0 / dy) + 1
    n_cols = n + 1
    
    # Generate vertices
    vertices = []
    vertex_map = {}
    
    for i in range(n_rows):
        row_offset = (dx / 2) if (i % 2) == 1 else 0
        for j in range(n_cols):
            x = j * dx + row_offset
            y = i * dy
            vertex_map[(i, j)] = len(vertices)
            vertices.append([x, y, 0])
    
    vertices = np.array(vertices)
    
    # Generate triangles
    triangles = []
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            if (i, j) in vertex_map and (i, j+1) in vertex_map and (i+1, j) in vertex_map:
                if i % 2 == 0:
                    # Pointing up triangle
                    triangles.append([vertex_map[(i, j)], vertex_map[(i, j+1)], vertex_map[(i+1, j)]])
                    # Pointing down triangle (if vertices exist)
                    if (i+1, j+1) in vertex_map:
                        triangles.append([vertex_map[(i, j+1)], vertex_map[(i+1, j+1)], vertex_map[(i+1, j)]])
                else:
                    # Pointing down triangle
                    if (i+1, j+1) in vertex_map:
                        triangles.append([vertex_map[(i, j)], vertex_map[(i+1, j+1)], vertex_map[(i+1, j)]])
                    # Pointing up triangle (if vertices exist)
                    if j > 0 and (i, j-1) in vertex_map:
                        triangles.append([vertex_map[(i, j-1)], vertex_map[(i, j)], vertex_map[(i+1, j)]])
    
    triangles = np.array(triangles)
    
    # Generate edges
    edges = set()
    for triangle in triangles:
        # Add all edges of the triangle
        edges.add(tuple(sorted([triangle[0], triangle[1]])))
        edges.add(tuple(sorted([triangle[1], triangle[2]])))
        edges.add(tuple(sorted([triangle[2], triangle[0]])))
    
    edges = np.array(list(edges))
    
    return vertices, triangles, edges