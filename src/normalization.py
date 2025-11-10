"""
Task 2: Mesh Normalization and Quantization
This module implements different normalization techniques and quantization.
"""

import numpy as np
import trimesh
from pathlib import Path


class MeshNormalizer:
    """Normalize mesh vertices using different methods."""
    
    def __init__(self, vertices):
        """
        Initialize normalizer with vertex data.
        
        Args:
            vertices: Numpy array of shape (N, 3) containing vertex coordinates
        """
        self.original_vertices = vertices.copy()
        self.normalized_vertices = None
        self.normalization_params = {}
        
    def minmax_normalize(self, target_range=(0, 1)):
        """
        Apply Min-Max normalization.
        
        Brings all coordinates into the specified range [min, max].
        Formula: x' = (x - x_min) / (x_max - x_min) * (range_max - range_min) + range_min
        
        Args:
            target_range: Tuple of (min, max) for target range
            
        Returns:
            Normalized vertices
        """
        v_min = self.original_vertices.min(axis=0)
        v_max = self.original_vertices.max(axis=0)
        
        # Avoid division by zero
        range_vals = v_max - v_min
        range_vals[range_vals == 0] = 1.0
        
        # Normalize to [0, 1] first
        normalized = (self.original_vertices - v_min) / range_vals
        
        # Scale to target range
        range_min, range_max = target_range
        normalized = normalized * (range_max - range_min) + range_min
        
        self.normalized_vertices = normalized
        self.normalization_params = {
            'method': 'minmax',
            'v_min': v_min,
            'v_max': v_max,
            'target_range': target_range
        }
        
        return self.normalized_vertices
    
    def unit_sphere_normalize(self):
        """
        Apply Unit Sphere normalization.
        
        Centers the mesh at origin and scales so all vertices fit 
        within a sphere of radius 1.
        
        Returns:
            Normalized vertices
        """
        # Center at origin
        centroid = self.original_vertices.mean(axis=0)
        centered = self.original_vertices - centroid
        
        # Find maximum distance from origin
        max_distance = np.max(np.linalg.norm(centered, axis=1))
        
        # Avoid division by zero
        if max_distance == 0:
            max_distance = 1.0
        
        # Scale to unit sphere
        normalized = centered / max_distance
        
        self.normalized_vertices = normalized
        self.normalization_params = {
            'method': 'unit_sphere',
            'centroid': centroid,
            'max_distance': max_distance
        }
        
        return self.normalized_vertices
    
    def zscore_normalize(self):
        """
        Apply Z-Score normalization.
        
        Centers the mesh and scales by standard deviation.
        Formula: x' = (x - μ) / σ
        
        Returns:
            Normalized vertices
        """
        mean = self.original_vertices.mean(axis=0)
        std = self.original_vertices.std(axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        normalized = (self.original_vertices - mean) / std
        
        self.normalized_vertices = normalized
        self.normalization_params = {
            'method': 'zscore',
            'mean': mean,
            'std': std
        }
        
        return self.normalized_vertices
    
    def denormalize(self, normalized_vertices, params=None):
        """
        Reverse the normalization process.
        
        Args:
            normalized_vertices: Normalized vertex array
            params: Normalization parameters (uses stored params if None)
            
        Returns:
            Denormalized vertices
        """
        if params is None:
            params = self.normalization_params
        
        method = params['method']
        
        if method == 'minmax':
            # Reverse scaling from target range
            range_min, range_max = params['target_range']
            denorm = (normalized_vertices - range_min) / (range_max - range_min)
            
            # Reverse normalization
            v_min = params['v_min']
            v_max = params['v_max']
            range_vals = v_max - v_min
            range_vals[range_vals == 0] = 1.0
            
            denorm = denorm * range_vals + v_min
            
        elif method == 'unit_sphere':
            # Reverse scaling
            denorm = normalized_vertices * params['max_distance']
            # Reverse centering
            denorm = denorm + params['centroid']
            
        elif method == 'zscore':
            # Reverse z-score
            denorm = normalized_vertices * params['std'] + params['mean']
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return denorm


class MeshQuantizer:
    """Quantize mesh vertices into discrete bins."""
    
    def __init__(self, n_bins=1024):
        """
        Initialize quantizer.
        
        Args:
            n_bins: Number of bins for quantization
        """
        self.n_bins = n_bins
        
    def quantize(self, vertices, input_range=(0, 1)):
        """
        Quantize vertices to discrete bins.
        
        Args:
            vertices: Normalized vertex array (should be in specified range)
            input_range: Range of input values (default [0, 1])
            
        Returns:
            Quantized vertices as integer array
        """
        # Shift to [0, 1] if needed
        range_min, range_max = input_range
        normalized = (vertices - range_min) / (range_max - range_min)
        
        # Clip to ensure values are in valid range
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Quantize
        quantized = np.floor(normalized * (self.n_bins - 1)).astype(np.int32)
        
        # Ensure values are within bounds
        quantized = np.clip(quantized, 0, self.n_bins - 1)
        
        return quantized
    
    def dequantize(self, quantized_vertices, output_range=(0, 1)):
        """
        Reverse quantization to continuous values.
        
        Args:
            quantized_vertices: Integer array of quantized vertices
            output_range: Desired output range (default [0, 1])
            
        Returns:
            Dequantized vertices as float array
        """
        # Convert back to [0, 1]
        dequantized = quantized_vertices.astype(np.float64) / (self.n_bins - 1)
        
        # Scale to output range
        range_min, range_max = output_range
        dequantized = dequantized * (range_max - range_min) + range_min
        
        return dequantized


def save_mesh(vertices, faces, filepath, format='obj'):
    """
    Save mesh to file.
    
    Args:
        vertices: Vertex array
        faces: Face array
        filepath: Output file path
        format: File format ('obj' or 'ply')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(str(filepath))
    print(f"✓ Saved mesh to: {filepath}")


def process_mesh_pipeline(vertices, faces, method='minmax', n_bins=1024, 
                          output_dir=None, mesh_name='mesh'):
    """
    Complete pipeline: normalize -> quantize -> save.
    
    Args:
        vertices: Original vertex array
        faces: Face array
        method: Normalization method ('minmax' or 'unit_sphere')
        n_bins: Number of quantization bins
        output_dir: Directory to save outputs
        mesh_name: Name prefix for output files
        
    Returns:
        Dictionary containing all intermediate results
    """
    results = {
        'original_vertices': vertices.copy(),
        'method': method,
        'n_bins': n_bins
    }
    
    # Normalize
    normalizer = MeshNormalizer(vertices)
    
    if method == 'minmax':
        normalized = normalizer.minmax_normalize(target_range=(0, 1))
    elif method == 'unit_sphere':
        normalized = normalizer.unit_sphere_normalize()
        # Shift unit sphere to [0, 1] range for quantization
        normalized = (normalized + 1) / 2  # Map [-1, 1] to [0, 1]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    results['normalized_vertices'] = normalized
    results['normalization_params'] = normalizer.normalization_params
    
    # Quantize
    quantizer = MeshQuantizer(n_bins=n_bins)
    quantized = quantizer.quantize(normalized, input_range=(0, 1))
    results['quantized_vertices'] = quantized
    results['quantizer'] = quantizer
    
    # Save meshes if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        
        # Save normalized mesh
        norm_path = output_dir / 'normalized' / f"{mesh_name}_{method}_normalized.obj"
        save_mesh(normalized, faces, norm_path)
        
        # Save quantized mesh (convert back to float for visualization)
        quant_float = quantized.astype(np.float64) / (n_bins - 1)
        quant_path = output_dir / 'quantized' / f"{mesh_name}_{method}_quantized.obj"
        save_mesh(quant_float, faces, quant_path)
    
    return results


if __name__ == "__main__":
    # Test normalization
    test_vertices = np.random.randn(100, 3) * 10
    
    print("Testing Min-Max Normalization:")
    normalizer = MeshNormalizer(test_vertices)
    normalized = normalizer.minmax_normalize()
    print(f"Original range: [{test_vertices.min():.2f}, {test_vertices.max():.2f}]")
    print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    print("\nTesting Unit Sphere Normalization:")
    normalizer2 = MeshNormalizer(test_vertices)
    normalized2 = normalizer2.unit_sphere_normalize()
    print(f"Max distance from origin: {np.max(np.linalg.norm(normalized2, axis=1)):.6f}")
    
    print("\nTesting Quantization:")
    quantizer = MeshQuantizer(n_bins=1024)
    quantized = quantizer.quantize(normalized)
    dequantized = quantizer.dequantize(quantized)
    print(f"Quantization error: {np.mean(np.abs(normalized - dequantized)):.6f}")
