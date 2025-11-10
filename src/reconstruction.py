"""
Task 3: Reconstruction and Error Analysis
This module handles dequantization, denormalization, and error measurement.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import trimesh


class ErrorAnalyzer:
    """Analyze reconstruction errors between original and processed meshes."""
    
    def __init__(self, original_vertices, reconstructed_vertices):
        """
        Initialize error analyzer.
        
        Args:
            original_vertices: Original vertex array
            reconstructed_vertices: Reconstructed vertex array
        """
        self.original = original_vertices
        self.reconstructed = reconstructed_vertices
        
        if self.original.shape != self.reconstructed.shape:
            raise ValueError("Vertex arrays must have the same shape")
    
    def compute_mse(self):
        """
        Compute Mean Squared Error.
        
        Returns:
            float: MSE value
        """
        mse = np.mean((self.original - self.reconstructed) ** 2)
        return mse
    
    def compute_mae(self):
        """
        Compute Mean Absolute Error.
        
        Returns:
            float: MAE value
        """
        mae = np.mean(np.abs(self.original - self.reconstructed))
        return mae
    
    def compute_rmse(self):
        """
        Compute Root Mean Squared Error.
        
        Returns:
            float: RMSE value
        """
        rmse = np.sqrt(self.compute_mse())
        return rmse
    
    def compute_per_axis_error(self):
        """
        Compute errors per axis (X, Y, Z).
        
        Returns:
            dict: Dictionary with per-axis MSE, MAE, and RMSE
        """
        errors = {
            'mse': {},
            'mae': {},
            'rmse': {}
        }
        
        axes = ['X', 'Y', 'Z']
        for i, axis in enumerate(axes):
            diff = self.original[:, i] - self.reconstructed[:, i]
            errors['mse'][axis] = np.mean(diff ** 2)
            errors['mae'][axis] = np.mean(np.abs(diff))
            errors['rmse'][axis] = np.sqrt(errors['mse'][axis])
        
        return errors
    
    def compute_max_error(self):
        """
        Compute maximum error across all vertices.
        
        Returns:
            float: Maximum absolute error
        """
        return np.max(np.abs(self.original - self.reconstructed))
    
    def compute_relative_error(self):
        """
        Compute relative error (percentage).
        
        Returns:
            float: Mean relative error percentage
        """
        # Avoid division by zero
        original_norm = np.linalg.norm(self.original, axis=1)
        error_norm = np.linalg.norm(self.original - self.reconstructed, axis=1)
        
        valid_mask = original_norm > 1e-10
        if np.any(valid_mask):
            relative_errors = error_norm[valid_mask] / original_norm[valid_mask]
            return np.mean(relative_errors) * 100
        else:
            return 0.0
    
    def get_all_metrics(self):
        """
        Compute all error metrics.
        
        Returns:
            dict: Dictionary with all metrics
        """
        per_axis = self.compute_per_axis_error()
        
        metrics = {
            'mse': self.compute_mse(),
            'mae': self.compute_mae(),
            'rmse': self.compute_rmse(),
            'max_error': self.compute_max_error(),
            'relative_error_percent': self.compute_relative_error(),
            'per_axis_mse': per_axis['mse'],
            'per_axis_mae': per_axis['mae'],
            'per_axis_rmse': per_axis['rmse']
        }
        
        return metrics
    
    def print_metrics(self, title="Error Metrics"):
        """Print all error metrics in a formatted way."""
        metrics = self.get_all_metrics()
        
        print("\n" + "="*70)
        print(f"{title}")
        print("="*70)
        print(f"Overall Metrics:")
        print(f"  Mean Squared Error (MSE):     {metrics['mse']:.8f}")
        print(f"  Mean Absolute Error (MAE):    {metrics['mae']:.8f}")
        print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.8f}")
        print(f"  Maximum Error:                {metrics['max_error']:.8f}")
        print(f"  Relative Error:               {metrics['relative_error_percent']:.4f}%")
        
        print("\n" + "-"*70)
        print("Per-Axis Errors:")
        print(f"{'Axis':<10} {'MSE':>15} {'MAE':>15} {'RMSE':>15}")
        print("-"*70)
        for axis in ['X', 'Y', 'Z']:
            print(f"{axis:<10} {metrics['per_axis_mse'][axis]:>15.8f} "
                  f"{metrics['per_axis_mae'][axis]:>15.8f} "
                  f"{metrics['per_axis_rmse'][axis]:>15.8f}")
        print("="*70 + "\n")
        
        return metrics


class Visualizer:
    """Create visualizations for error analysis."""
    
    @staticmethod
    def plot_error_distribution(original, reconstructed, save_path=None, title="Error Distribution"):
        """
        Plot the distribution of errors.
        
        Args:
            original: Original vertices
            reconstructed: Reconstructed vertices
            save_path: Path to save the plot
            title: Plot title
        """
        errors = original - reconstructed
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Per-axis error distribution
        axes_names = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue']
        
        for i in range(3):
            axes[i//2, i%2].hist(errors[:, i], bins=50, alpha=0.7, 
                                 color=colors[i], edgecolor='black')
            axes[i//2, i%2].set_xlabel(f'{axes_names[i]}-axis Error', fontsize=12)
            axes[i//2, i%2].set_ylabel('Frequency', fontsize=12)
            axes[i//2, i%2].set_title(f'{axes_names[i]}-axis Error Distribution')
            axes[i//2, i%2].grid(True, alpha=0.3)
            axes[i//2, i%2].axvline(x=0, color='black', linestyle='--', linewidth=2)
        
        # Overall error magnitude
        error_magnitudes = np.linalg.norm(errors, axis=1)
        axes[1, 1].hist(error_magnitudes, bins=50, alpha=0.7, 
                       color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Error Magnitude', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Overall Error Magnitude Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved error distribution plot to: {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_per_axis_comparison(original, reconstructed, save_path=None, title="Per-Axis Error Comparison"):
        """
        Create bar plots comparing per-axis errors.
        
        Args:
            original: Original vertices
            reconstructed: Reconstructed vertices
            save_path: Path to save the plot
            title: Plot title
        """
        analyzer = ErrorAnalyzer(original, reconstructed)
        per_axis = analyzer.compute_per_axis_error()
        
        axes_names = ['X', 'Y', 'Z']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        metrics = ['MSE', 'MAE', 'RMSE']
        colors_map = {'MSE': 'red', 'MAE': 'green', 'RMSE': 'blue'}
        
        for idx, metric in enumerate(['mse', 'mae', 'rmse']):
            values = [per_axis[metric][axis] for axis in axes_names]
            
            axes[idx].bar(axes_names, values, color=colors_map[metrics[idx]], 
                         alpha=0.7, edgecolor='black', linewidth=1.5)
            axes[idx].set_ylabel(metrics[idx], fontsize=12)
            axes[idx].set_xlabel('Axis', fontsize=12)
            axes[idx].set_title(f'{metrics[idx]} per Axis')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v, f'{v:.6f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved per-axis comparison plot to: {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_comparison_scatter(original, reconstructed, sample_size=1000, 
                               save_path=None, title="Original vs Reconstructed"):
        """
        Create 3D scatter plot comparing original and reconstructed vertices.
        
        Args:
            original: Original vertices
            reconstructed: Reconstructed vertices
            sample_size: Number of points to sample for visualization
            save_path: Path to save the plot
            title: Plot title
        """
        # Sample points if too many
        n_vertices = len(original)
        if n_vertices > sample_size:
            indices = np.random.choice(n_vertices, sample_size, replace=False)
            orig_sample = original[indices]
            recon_sample = reconstructed[indices]
        else:
            orig_sample = original
            recon_sample = reconstructed
        
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Original mesh
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(orig_sample[:, 0], orig_sample[:, 1], orig_sample[:, 2],
                   c='blue', alpha=0.5, s=1)
        ax1.set_title('Original Mesh')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Reconstructed mesh
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(recon_sample[:, 0], recon_sample[:, 1], recon_sample[:, 2],
                   c='red', alpha=0.5, s=1)
        ax2.set_title('Reconstructed Mesh')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Overlay
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(orig_sample[:, 0], orig_sample[:, 1], orig_sample[:, 2],
                   c='blue', alpha=0.3, s=1, label='Original')
        ax3.scatter(recon_sample[:, 0], recon_sample[:, 1], recon_sample[:, 2],
                   c='red', alpha=0.3, s=1, label='Reconstructed')
        ax3.set_title('Overlay')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison scatter plot to: {save_path}")
        
        plt.close()


def reconstruct_mesh(quantized_vertices, normalization_params, quantizer, method='minmax'):
    """
    Complete reconstruction pipeline: dequantize -> denormalize.
    
    Args:
        quantized_vertices: Quantized vertex array
        normalization_params: Parameters from normalization
        quantizer: MeshQuantizer instance
        method: Normalization method used
        
    Returns:
        Reconstructed vertices
    """
    # Dequantize
    dequantized = quantizer.dequantize(quantized_vertices, output_range=(0, 1))
    
    # Handle unit sphere special case (was shifted to [0, 1])
    if method == 'unit_sphere':
        dequantized = dequantized * 2 - 1  # Map [0, 1] back to [-1, 1]
    
    # Denormalize
    from normalization import MeshNormalizer
    normalizer = MeshNormalizer(dequantized)  # Dummy initialization
    reconstructed = normalizer.denormalize(dequantized, normalization_params)
    
    return reconstructed


if __name__ == "__main__":
    # Test error analysis
    original = np.random.randn(1000, 3)
    noise = np.random.randn(1000, 3) * 0.01
    reconstructed = original + noise
    
    analyzer = ErrorAnalyzer(original, reconstructed)
    analyzer.print_metrics("Test Error Analysis")
    
    # Test visualization
    visualizer = Visualizer()
    visualizer.plot_error_distribution(original, reconstructed, 
                                      save_path='test_error_dist.png')
    print("Test complete!")
