"""
Task 1: Mesh Loading and Inspection
This module handles loading .obj files and extracting vertex statistics.
"""

import trimesh
import numpy as np
import open3d as o3d
from pathlib import Path


class MeshLoader:
    """Load and inspect 3D mesh files."""
    
    def __init__(self, mesh_path):
        """
        Initialize the mesh loader.
        
        Args:
            mesh_path: Path to the .obj file
        """
        self.mesh_path = Path(mesh_path)
        self.mesh = None
        self.vertices = None
        self.faces = None
        self.load_mesh()
        
    def load_mesh(self):
        """Load the mesh from file using trimesh."""
        try:
            self.mesh = trimesh.load(str(self.mesh_path))
            self.vertices = np.array(self.mesh.vertices, dtype=np.float64)
            self.faces = np.array(self.mesh.faces) if hasattr(self.mesh, 'faces') else None
            print(f"✓ Successfully loaded mesh: {self.mesh_path.name}")
        except Exception as e:
            print(f"✗ Error loading mesh {self.mesh_path}: {e}")
            raise
            
    def get_statistics(self):
        """
        Compute and return statistics about the mesh vertices.
        
        Returns:
            dict: Dictionary containing mesh statistics
        """
        if self.vertices is None:
            raise ValueError("No vertices loaded")
            
        stats = {
            'num_vertices': len(self.vertices),
            'num_faces': len(self.faces) if self.faces is not None else 0,
            'min_coords': self.vertices.min(axis=0),
            'max_coords': self.vertices.max(axis=0),
            'mean_coords': self.vertices.mean(axis=0),
            'std_coords': self.vertices.std(axis=0),
            'range_coords': self.vertices.max(axis=0) - self.vertices.min(axis=0)
        }
        
        return stats
    
    def print_statistics(self):
        """Print detailed statistics about the mesh."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print(f"MESH STATISTICS: {self.mesh_path.name}")
        print("="*70)
        print(f"Number of Vertices: {stats['num_vertices']:,}")
        print(f"Number of Faces: {stats['num_faces']:,}")
        print("\n" + "-"*70)
        print(f"{'Axis':<10} {'Min':>15} {'Max':>15} {'Mean':>15} {'Std Dev':>15}")
        print("-"*70)
        
        axes = ['X', 'Y', 'Z']
        for i, axis in enumerate(axes):
            print(f"{axis:<10} {stats['min_coords'][i]:>15.6f} "
                  f"{stats['max_coords'][i]:>15.6f} "
                  f"{stats['mean_coords'][i]:>15.6f} "
                  f"{stats['std_coords'][i]:>15.6f}")
        
        print("\n" + "-"*70)
        print(f"{'Axis':<10} {'Range':>15}")
        print("-"*70)
        for i, axis in enumerate(axes):
            print(f"{axis:<10} {stats['range_coords'][i]:>15.6f}")
        print("="*70 + "\n")
        
        return stats
    
    def visualize(self, title=None, save_path=None):
        """
        Visualize the mesh using Open3D.
        
        Args:
            title: Window title
            save_path: Optional path to save screenshot
        """
        try:
            # Create Open3D mesh
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
            
            if self.faces is not None:
                o3d_mesh.triangles = o3d.utility.Vector3iVector(self.faces)
            
            # Compute normals for better visualization
            o3d_mesh.compute_vertex_normals()
            
            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=title or f"Mesh: {self.mesh_path.name}")
            vis.add_geometry(o3d_mesh)
            
            # Set rendering options
            opt = vis.get_render_option()
            opt.mesh_show_back_face = True
            opt.mesh_show_wireframe = False
            
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            print(f"Warning: Could not visualize mesh: {e}")
    
    def get_vertices(self):
        """Return the vertex array."""
        return self.vertices.copy()
    
    def get_faces(self):
        """Return the faces array."""
        return self.faces.copy() if self.faces is not None else None


def load_all_meshes(directory):
    """
    Load all .obj files from a directory.
    
    Args:
        directory: Path to directory containing .obj files
        
    Returns:
        dict: Dictionary mapping filenames to MeshLoader objects
    """
    directory = Path(directory)
    mesh_files = list(directory.glob("*.obj"))
    
    if not mesh_files:
        raise ValueError(f"No .obj files found in {directory}")
    
    print(f"\nFound {len(mesh_files)} mesh files in {directory}")
    
    meshes = {}
    for mesh_file in mesh_files:
        try:
            loader = MeshLoader(mesh_file)
            meshes[mesh_file.stem] = loader
        except Exception as e:
            print(f"Failed to load {mesh_file.name}: {e}")
    
    return meshes


if __name__ == "__main__":
    # Test the loader
    import sys
    
    if len(sys.argv) > 1:
        mesh_path = sys.argv[1]
    else:
        mesh_path = "../samples/branch.obj"
    
    loader = MeshLoader(mesh_path)
    loader.print_statistics()
    
    # Uncomment to visualize
    # loader.visualize()
