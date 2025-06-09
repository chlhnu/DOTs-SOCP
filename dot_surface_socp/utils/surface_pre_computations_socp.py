""" Precompute the geometrics operators on a triangulated surface
"""

# Mathematical functions
import numpy as np
import scipy.sparse as scsp
from numpy import linalg as lin
from math import acos


def geometricQuantities(vertices, triangles, edges) :
	"""Compute the geometric quantities associated to the mesh
	"""
	n_triangles = triangles.shape[0]
	base_function = np.zeros((n_triangles, 3,3 ))

	area_triangles = np.zeros(n_triangles)
	angle_triangles = np.zeros((n_triangles, 3))

	for i in range(n_triangles) :
		v01 = vertices[ triangles[i,1],:] - vertices[ triangles[i,0],:]
		v12 = vertices[ triangles[i,2],:] - vertices[ triangles[i,1],:]
		v20 = vertices[ triangles[i,0],:] - vertices[ triangles[i,2],:]

		area_triangles[i] = lin.norm( np.cross(  v01, v12  )  ) / 2

		angle_triangles[i,0] = acos(  np.dot( v01, - v20  ) / (lin.norm(v01)*lin.norm(v20) ))
		angle_triangles[i,1] = acos(  np.dot( v12, - v01  ) / (lin.norm(v12)*lin.norm(v01) ))
		angle_triangles[i,2] = acos(  np.dot( v20, - v12  ) / (lin.norm(v20)*lin.norm(v12) ))

		base_function[i,0,:] = - v01 + ( v12 ) * (  np.dot( v01, v12 ) / np.dot( v12,v12  )    )
		base_function[i,1,:] = - v12 + ( v20 ) * (  np.dot( v12, v20 ) / np.dot( v20,v20  )    )
		base_function[i,2,:] = - v20 + ( v01 ) * (  np.dot( v20, v01 ) / np.dot( v01,v01  )    )

		base_function[i,0,:] /= lin.norm(base_function[i,0,:])**2
		base_function[i,1,:] /= lin.norm(base_function[i,1,:])**2
		base_function[i,2,:] /= lin.norm(base_function[i,2,:])**2

	return area_triangles, angle_triangles, base_function


def geometricMatrices(vertices, triangles, edges, area_triangles, angle_triangles, base_function) :
	"""Computes three geometric operators for the mesh

	Returns:
    	1. Laplacian matrix using cotangent weights - Maps scalar functions to scalar functions
    	2. Gradient matrix - Maps scalar functions defined on vertices to vector fields on triangles 
    	3. Divergence matrix - The negative adjoint of the gradient matrix, maps vector fields on triangles to scalar functions on vertices
	"""

	n_vertices = vertices.shape[0]
	n_triangles = triangles.shape[0]

	# Build gradient matrix
	gradient = scsp.coo_matrix((n_triangles*3, n_vertices))
	for id_coord in range(3):
		for id_vertex in range(3):
			row_indices = [3*i + id_coord for i in range(n_triangles)]
			col_indices = triangles[:, id_vertex]
			values = base_function[:, id_vertex, id_coord]
			gradient += scsp.coo_matrix((values, (row_indices, col_indices)), 
										shape=(n_triangles*3, n_vertices)).tocsr()

	# Build divergence matrix
	divergence = - gradient.transpose()

	# Build Laplacian matrix
	cotan_weights = 0.5 * np.divide(np.cos(angle_triangles), np.sin(angle_triangles))
	laplacian = scsp.coo_matrix((n_vertices, n_vertices))
	# 	For vertex 0
	laplacian += scsp.coo_matrix((cotan_weights[:,0], (triangles[:,1], triangles[:,2])), shape=(n_vertices, n_vertices)).tocsr()
	laplacian += scsp.coo_matrix((-cotan_weights[:,0], (triangles[:,1], triangles[:,1])), shape=(n_vertices, n_vertices)).tocsr()
	laplacian += scsp.coo_matrix((cotan_weights[:,0], (triangles[:,2], triangles[:,1])), shape=(n_vertices, n_vertices)).tocsr()
	laplacian += scsp.coo_matrix((-cotan_weights[:,0], (triangles[:,2], triangles[:,2])), shape=(n_vertices, n_vertices)).tocsr()
	# 	For vertex 1
	laplacian += scsp.coo_matrix((cotan_weights[:,1], (triangles[:,2], triangles[:,0])), shape=(n_vertices, n_vertices)).tocsr()
	laplacian += scsp.coo_matrix((-cotan_weights[:,1], (triangles[:,2], triangles[:,2])), shape=(n_vertices, n_vertices)).tocsr()
	laplacian += scsp.coo_matrix((cotan_weights[:,1], (triangles[:,0], triangles[:,2])), shape=(n_vertices, n_vertices)).tocsr()
	laplacian += scsp.coo_matrix((-cotan_weights[:,1], (triangles[:,0], triangles[:,0])), shape=(n_vertices, n_vertices)).tocsr()
	# 	For vertex 2
	laplacian += scsp.coo_matrix((cotan_weights[:,2], (triangles[:,0], triangles[:,1])), shape=(n_vertices, n_vertices)).tocsr()
	laplacian += scsp.coo_matrix((-cotan_weights[:,2], (triangles[:,1], triangles[:,1])), shape=(n_vertices, n_vertices)).tocsr()
	laplacian += scsp.coo_matrix((cotan_weights[:,2], (triangles[:,1], triangles[:,0])), shape=(n_vertices, n_vertices)).tocsr()
	laplacian += scsp.coo_matrix((-cotan_weights[:,2], (triangles[:,0], triangles[:,0])), shape=(n_vertices, n_vertices)).tocsr()

	return gradient, divergence, laplacian

def trianglesToVertices(vertices, triangles, area_triangles) :
	"""
	Compute matrices for triangle-to-vertex transformations.

	Returns:
		originTriangle: Matrix (nVertices x nTriangles*3) that maps triangle data to vertices
			- Input: Vector of size (nTriangles*3)
			- Output: Vector of size (nVertices)
			- Each entry (a,v) is weighted by triangle a's area and mapped to vertex v
			
		areaVertices: Vector of size (nVertices)
			- Contains sum of areas of all triangles containing each vertex
			
		vertexTriangles: Matrix (nTriangles*3 x nVertices) for vertex-to-triangle mapping
			- Input: Function defined on vertices
			- Output: Vector of size (nTriangles*3)
			- Maps vertex data to corresponding triangle-vertex pairs (a,v)

		areaVerticesTriangles: Vector of size (nTriangles*3)
			- Contains area of vertex corresponding to each triangle-vertex pair (a,v)
	"""
	n_vertices = vertices.shape[0]
	n_triangles = triangles.shape[0]

	vertex_of_triangles = []
	arrayArea = np.zeros(3*n_triangles)
	for i in range(3*n_triangles) :
		j = i % n_triangles
		k = i // n_triangles
		vertex_of_triangles.append( triangles[j,k] )
		arrayArea[i] = area_triangles[j]

	# Build the mapping: face -> vertex
	mapping_face_to_vertex = scsp.coo_matrix(( arrayArea , ( vertex_of_triangles , range(3*n_triangles)  )  ) , shape = (n_vertices,3*n_triangles)).tocsr()

	# Area of vertex
	area_vertices = mapping_face_to_vertex.dot( np.ones(  3*n_triangles )  )

	# Build the mapping: vertex -> face (without the weighting by the areas)
	mapping_vertex_to_face = scsp.coo_matrix(( np.ones(3*n_triangles) , ( range(3*n_triangles) , vertex_of_triangles  )  ) , shape = (3*n_triangles, n_vertices)).tocsr()

	# Area of vertex correspoinding to the triangle-vertex pair (a,v)
	area_vertices_triangles = area_vertices[vertex_of_triangles]

	return mapping_face_to_vertex, area_vertices, mapping_vertex_to_face, area_vertices_triangles
