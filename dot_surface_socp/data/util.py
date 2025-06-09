from numpy.linalg import norm as np_norm
import numpy as np
from pathlib import Path


def gaussian(v, c, scale):
    return np.exp(-np_norm(v - c) ** 2 / scale)

def mask(v, c, r):
    return np_norm(v - c) < r

def cut(cond, val):
    return np.where(cond, val, 0)

def weighted_norm(x: np.ndarray, weight: np.ndarray):
    return np.sqrt((x ** 2 * (weight / np.sum(weight))).sum())

def cut_off(x, sigma):
    """Define a cutoff function = 1 if x <= 0, = 0 if x>=sigma, smooth transition in between
    """
    x /= sigma
    if x <= 0.:
        return 1.
    elif x >= 1.:
        return 0.
    else:
        return (x - 1.) ** 2 * (x + 1) ** 2
    
def read_mesh(name_file: str, kwargs_generating_mesh: dict = None):
	"""Read mesh data from file.

	Supports two file formats:
	- .off: Standard OFF mesh file format
	- .py: Python script that contains generate_mesh() function to generate mesh data

	Args:
		name_file (str): 
			Path to the mesh file
		kwargs_generating_mesh (dict, optional): 
			Keyword arguments to pass to generate_mesh() function if name_file is a .py file.
		
	Returns:
		tuple: (vertices, triangles, edges) where:
			vertices: numpy array of shape (nVertices, 3)
			triangles: numpy array of shape (nTriangles, 3)
			edges: numpy array of shape (nEdges, 2)
			
	Raises:
		ValueError: If file format is not supported
	"""
	ext = Path(name_file).suffix

	if ext == ".off":
		vertices, triangles, edges = read_mesh_off(name_file)
	elif ext == ".py":
		# Execute the Python file and get mesh data
		module_name = Path(name_file).stem
		from importlib.util import spec_from_file_location, module_from_spec
		spec = spec_from_file_location(module_name, name_file)
		module = module_from_spec(spec)
		spec.loader.exec_module(module)

		if not hasattr(module, 'generate_mesh'):
			raise ValueError("Python file must contain generate_mesh() function")

		vertices, triangles, edges = module.generate_mesh(**kwargs_generating_mesh)
	else:
		raise ValueError(f"File format ({ext}) not supported")

	return vertices, triangles, edges

    
def read_mesh_off(name_file: str):
    """Read .off file and return vertices, triangles, and edges arrays.
    
    Args:
        name_file (str): Path to the .off file
        
    Returns:
        tuple: (vertices, triangles, edges) where:
            vertices: numpy array of shape (nVertices, 3)
            triangles: numpy array of shape (nTriangles, 3)
            edges: numpy array of shape (nEdges, 2)
            
    Raises:
        ValueError: If file is not a valid .off file
    """
    try:
        with open(name_file, 'r') as file_off:
            if file_off.readline().strip() != 'OFF':
                raise ValueError("Not a valid .off file")
            
            toRead = file_off.readline().split()
            if len(toRead) < 2:
                raise ValueError("Invalid file format: missing vertex/triangle counts")
                
            n_vertices = int(toRead[0])
            n_triangles = int(toRead[1])
            n_edges = 3 * n_triangles
            
            vertices = np.zeros((n_vertices, 3))
            triangles = np.zeros((n_triangles, 3), dtype=int)
            edges = np.zeros((n_edges, 2), dtype=int)
            
            # Read vertices and triangles
            counterV = 0
            counterT = 0
            
            for line in file_off:
                toRead = line.strip().split()
                if not toRead:  # Skip empty lines
                    continue
                    
                if toRead[0] == '3':
                    if len(toRead) < 4:
                        raise ValueError(f"Invalid triangle data at line {counterT + 1}")
                    
                    # Fill triangles
                    triangles[counterT] = [int(toRead[1]), int(toRead[2]), int(toRead[3])]
                    
                    # Fill edges
                    edges[3*counterT:3*counterT+3] = [
                        [int(toRead[1]), int(toRead[2])],
                        [int(toRead[2]), int(toRead[3])],
                        [int(toRead[3]), int(toRead[1])]
                    ]
                    
                    counterT += 1
                else:
                    if len(toRead) < 3:
                        raise ValueError(f"Invalid vertex data at line {counterV + 1}")
                    
                    # Fill vertices
                    vertices[counterV] = [float(toRead[0]), float(toRead[1]), float(toRead[2])]
                    counterV += 1
            
            if counterV != n_vertices:
                raise ValueError(f"Expected {n_vertices} vertices but found {counterV}")
            if counterT != n_triangles:
                raise ValueError(f"Expected {n_triangles} triangles but found {counterT}")
                
            return vertices, triangles, edges
    except (IOError, ValueError) as e:
        raise ValueError(f"Error reading .off file: {str(e)}")