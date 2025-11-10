"""
Main Script: Complete Mesh Processing Pipeline
Executes all tasks: Loading, Normalization, Quantization, Reconstruction, and Analysis
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mesh_loader import MeshLoader, load_all_meshes
from normalization import MeshNormalizer, MeshQuantizer, process_mesh_pipeline, save_mesh
from reconstruction import ErrorAnalyzer, Visualizer, reconstruct_mesh


class MeshProcessingPipeline:
    """Complete pipeline for mesh processing and analysis."""
    
    def __init__(self, samples_dir, output_dir, n_bins=1024):
        """
        Initialize the pipeline.
        
        Args:
            samples_dir: Directory containing .obj files
            output_dir: Directory for output files
            n_bins: Number of quantization bins
        """
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path(output_dir)
        self.n_bins = n_bins
        self.meshes = {}
        self.results = {}
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'normalized').mkdir(exist_ok=True)
        (self.output_dir / 'quantized').mkdir(exist_ok=True)
        (self.output_dir / 'reconstructed').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
    
    def task1_load_and_inspect(self):
        """Task 1: Load all meshes and print statistics."""
        print("\n" + "="*80)
        print("TASK 1: LOADING AND INSPECTING MESHES")
        print("="*80)
        
        self.meshes = load_all_meshes(self.samples_dir)
        
        for mesh_name, loader in self.meshes.items():
            loader.print_statistics()
        
        print(f"✓ Task 1 Complete: Loaded {len(self.meshes)} meshes\n")
    
    def task2_normalize_and_quantize(self, methods=['minmax', 'unit_sphere']):
        """
        Task 2: Apply normalization and quantization.
        
        Args:
            methods: List of normalization methods to apply
        """
        print("\n" + "="*80)
        print("TASK 2: NORMALIZATION AND QUANTIZATION")
        print("="*80)
        
        for mesh_name, loader in self.meshes.items():
            print(f"\nProcessing mesh: {mesh_name}")
            print("-" * 70)
            
            vertices = loader.get_vertices()
            faces = loader.get_faces()
            
            self.results[mesh_name] = {}
            
            for method in methods:
                print(f"  Applying {method} normalization...")
                
                result = process_mesh_pipeline(
                    vertices=vertices,
                    faces=faces,
                    method=method,
                    n_bins=self.n_bins,
                    output_dir=self.output_dir,
                    mesh_name=mesh_name
                )
                
                self.results[mesh_name][method] = result
                
                print(f"    ✓ Normalized vertices range: "
                      f"[{result['normalized_vertices'].min():.4f}, "
                      f"{result['normalized_vertices'].max():.4f}]")
                print(f"    ✓ Quantized vertices range: "
                      f"[{result['quantized_vertices'].min()}, "
                      f"{result['quantized_vertices'].max()}]")
        
        print(f"\n✓ Task 2 Complete: Processed {len(self.meshes)} meshes with "
              f"{len(methods)} methods\n")
    
    def task3_reconstruct_and_analyze(self):
        """Task 3: Reconstruct meshes and analyze errors."""
        print("\n" + "="*80)
        print("TASK 3: RECONSTRUCTION AND ERROR ANALYSIS")
        print("="*80)
        
        comparison_data = []
        
        for mesh_name, methods_results in self.results.items():
            print(f"\nAnalyzing mesh: {mesh_name}")
            print("-" * 70)
            
            for method, result in methods_results.items():
                print(f"\n  Method: {method.upper()}")
                
                # Reconstruct
                reconstructed = reconstruct_mesh(
                    quantized_vertices=result['quantized_vertices'],
                    normalization_params=result['normalization_params'],
                    quantizer=result['quantizer'],
                    method=method
                )
                
                result['reconstructed_vertices'] = reconstructed
                
                # Save reconstructed mesh
                faces = self.meshes[mesh_name].get_faces()
                recon_path = (self.output_dir / 'reconstructed' / 
                            f"{mesh_name}_{method}_reconstructed.obj")
                save_mesh(reconstructed, faces, recon_path)
                
                # Analyze errors
                analyzer = ErrorAnalyzer(result['original_vertices'], reconstructed)
                metrics = analyzer.print_metrics(f"{mesh_name} - {method}")
                
                result['metrics'] = metrics
                
                # Store for comparison
                comparison_data.append({
                    'mesh': mesh_name,
                    'method': method,
                    'mse': metrics['mse'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'max_error': metrics['max_error'],
                    'relative_error': metrics['relative_error_percent']
                })
                
                # Create visualizations
                plot_dir = self.output_dir / 'plots' / mesh_name
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                visualizer = Visualizer()
                
                # Error distribution
                visualizer.plot_error_distribution(
                    result['original_vertices'],
                    reconstructed,
                    save_path=plot_dir / f"{method}_error_distribution.png",
                    title=f"{mesh_name} - {method} - Error Distribution"
                )
                
                # Per-axis comparison
                visualizer.plot_per_axis_comparison(
                    result['original_vertices'],
                    reconstructed,
                    save_path=plot_dir / f"{method}_per_axis_errors.png",
                    title=f"{mesh_name} - {method} - Per-Axis Errors"
                )
                
                # 3D scatter comparison (sample for performance)
                visualizer.plot_comparison_scatter(
                    result['original_vertices'],
                    reconstructed,
                    sample_size=1000,
                    save_path=plot_dir / f"{method}_scatter_comparison.png",
                    title=f"{mesh_name} - {method} - Original vs Reconstructed"
                )
        
        # Create comparison plots across all methods and meshes
        self._create_comparison_plots(comparison_data)
        
        print(f"\n✓ Task 3 Complete: Analyzed all meshes\n")
        
        return comparison_data
    
    def _create_comparison_plots(self, comparison_data):
        """Create comparison plots across all methods."""
        print("\nCreating comparison plots...")
        
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        
        # Group by method
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Error Metrics Comparison Across All Meshes', 
                    fontsize=16, fontweight='bold')
        
        metrics = ['mse', 'mae', 'rmse', 'relative_error']
        titles = ['Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)', 
                 'Root Mean Squared Error (RMSE)', 'Relative Error (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            methods = df['method'].unique()
            x = np.arange(len(df['mesh'].unique()))
            width = 0.35
            
            for i, method in enumerate(methods):
                method_data = df[df['method'] == method]
                values = method_data[metric].values
                ax.bar(x + i * width, values, width, label=method, alpha=0.8)
            
            ax.set_xlabel('Mesh', fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title)
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(df['mesh'].unique(), rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / 'plots' / 'overall_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved overall comparison to: {save_path}")
        plt.close()
        
        # Summary table
        summary = df.groupby('method').agg({
            'mse': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'relative_error': ['mean', 'std']
        }).round(8)
        
        print("\n" + "="*80)
        print("SUMMARY: AVERAGE ERRORS ACROSS ALL MESHES")
        print("="*80)
        print(summary)
        print("="*80 + "\n")
        
        # Save summary to file
        summary.to_csv(self.output_dir / 'error_summary.csv')
        print(f"✓ Saved error summary to: {self.output_dir / 'error_summary.csv'}")
    
    def generate_report(self, comparison_data):
        """Generate final analysis report."""
        print("\n" + "="*80)
        print("FINAL ANALYSIS AND CONCLUSIONS")
        print("="*80)
        
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        
        # Find best method
        best_by_mse = df.loc[df.groupby('mesh')['mse'].idxmin()]
        best_by_mae = df.loc[df.groupby('mesh')['mae'].idxmin()]
        
        print("\nBest Normalization Method by MSE (per mesh):")
        print("-" * 70)
        for _, row in best_by_mse.iterrows():
            print(f"  {row['mesh']:<15} -> {row['method']:<15} "
                  f"(MSE: {row['mse']:.8f})")
        
        print("\nBest Normalization Method by MAE (per mesh):")
        print("-" * 70)
        for _, row in best_by_mae.iterrows():
            print(f"  {row['mesh']:<15} -> {row['method']:<15} "
                  f"(MAE: {row['mae']:.8f})")
        
        # Overall best
        avg_errors = df.groupby('method')[['mse', 'mae', 'rmse']].mean()
        best_overall = avg_errors['mse'].idxmin()
        
        print(f"\n{'='*70}")
        print(f"OVERALL BEST METHOD: {best_overall.upper()}")
        print(f"{'='*70}")
        print(f"Average MSE: {avg_errors.loc[best_overall, 'mse']:.8f}")
        print(f"Average MAE: {avg_errors.loc[best_overall, 'mae']:.8f}")
        print(f"Average RMSE: {avg_errors.loc[best_overall, 'rmse']:.8f}")
        
        # Write conclusions
        conclusions = f"""
CONCLUSIONS AND OBSERVATIONS:

1. BEST NORMALIZATION METHOD:
   - The {best_overall} normalization method achieved the lowest average 
     reconstruction error across all meshes with an MSE of {avg_errors.loc[best_overall, 'mse']:.8f}.

2. ERROR PATTERNS OBSERVED:
   - Quantization with {self.n_bins} bins introduces discrete approximation errors
   - Min-Max normalization tends to preserve absolute coordinate relationships well
   - Unit Sphere normalization is rotation-invariant but may introduce scaling artifacts
   - Per-axis errors vary based on the mesh's geometric distribution along each axis

3. INFORMATION LOSS:
   - The reconstruction error is primarily due to quantization (discretization)
   - Higher bin counts would reduce quantization error but increase storage requirements
   - Trade-off between compression and accuracy is evident

4. RECOMMENDATIONS:
   - For geometric accuracy: Use Min-Max normalization with high bin count
   - For rotation invariance: Use Unit Sphere normalization
   - Consider adaptive quantization for meshes with non-uniform vertex distributions

5. PRACTICAL IMPLICATIONS FOR AI/ML:
   - Normalized and quantized meshes are suitable for neural network training
   - The error levels observed ({avg_errors['mse'].min():.8f} MSE) are acceptable for most 
     3D learning tasks
   - SeamGPT-style systems can work effectively with this preprocessing pipeline
"""
        
        print(conclusions)
        
        # Save report
        report_path = self.output_dir / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write(conclusions)
        print(f"✓ Saved analysis report to: {report_path}")
        
        print("="*80 + "\n")
    
    def run_complete_pipeline(self):
        """Execute the complete pipeline: Tasks 1, 2, and 3."""
        print("\n" + "="*80)
        print("=" + " "*78 + "=")
        print("=" + " "*20 + "MESH PROCESSING PIPELINE" + " "*34 + "=")
        print("=" + " "*78 + "=")
        print("="*80 + "\n")
        
        # Task 1
        self.task1_load_and_inspect()
        
        # Task 2
        self.task2_normalize_and_quantize(methods=['minmax', 'unit_sphere'])
        
        # Task 3
        comparison_data = self.task3_reconstruct_and_analyze()
        
        # Generate report
        self.generate_report(comparison_data)
        
        print("\n" + "="*80)
        print("=" + " "*78 + "=")
        print("=" + " "*25 + "PIPELINE COMPLETE!" + " "*33 + "=")
        print("=" + " "*78 + "=")
        print("="*80 + "\n")
        
        print(f"All outputs saved to: {self.output_dir}")
        print(f"  - Normalized meshes: {self.output_dir / 'normalized'}")
        print(f"  - Quantized meshes: {self.output_dir / 'quantized'}")
        print(f"  - Reconstructed meshes: {self.output_dir / 'reconstructed'}")
        print(f"  - Plots and visualizations: {self.output_dir / 'plots'}")
        print(f"  - Error summary: {self.output_dir / 'error_summary.csv'}")
        print(f"  - Analysis report: {self.output_dir / 'analysis_report.txt'}")


def main():
    """Main entry point."""
    # Configuration
    SAMPLES_DIR = Path(__file__).parent / "samples"
    OUTPUT_DIR = Path(__file__).parent / "outputs"
    N_BINS = 1024
    
    # Run pipeline
    pipeline = MeshProcessingPipeline(
        samples_dir=SAMPLES_DIR,
        output_dir=OUTPUT_DIR,
        n_bins=N_BINS
    )
    
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
