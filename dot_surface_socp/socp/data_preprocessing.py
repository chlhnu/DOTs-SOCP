import trimesh
from dot_surface_socp.utils.type import GeometryData
from dot_surface_socp.utils.surface_pre_computations_socp import geometricQuantities, trianglesToVertices

def normalize_geometry(geometry: GeometryData, camera_config: dict = None):
    vertices, triangles = geometry["vertices"], geometry["triangles"]

    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=triangles.copy(), process=False)

    offset = - mesh.centroid
    mesh.vertices += offset

    scale_factor = 1.0 / (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).max()
    mesh.vertices *= scale_factor

    offset2 = - mesh.vertices.min(axis=0)
    mesh.vertices += offset2

    normalized_vertices = mesh.vertices.copy()
    normalized_triangles = mesh.faces.copy()
    normalized_edges = mesh.edges.copy()

    area_triangles, _, _ = geometricQuantities(normalized_vertices, normalized_triangles, normalized_edges)
    _, area_vertices, _, _ = trianglesToVertices(vertices, triangles, area_triangles)

    normalized_geometry = GeometryData(
        vertices=normalized_vertices,
        triangles=normalized_triangles,
        edges=normalized_edges,
        mu0=geometry["mu0"],
        mu1=geometry["mu1"],
        area_triangles=area_triangles,
        area_vertices=area_vertices
    )

    if camera_config is None:
        return normalized_geometry, scale_factor
    else:
        noramlized_camera = {}
        noramlized_camera["position"] = (camera_config["position"] + offset) * scale_factor + offset2
        noramlized_camera["focal_point"] = (camera_config["focal_point"] + offset) * scale_factor + offset2
        noramlized_camera["up"] = camera_config["up"]

        return normalized_geometry, scale_factor, noramlized_camera