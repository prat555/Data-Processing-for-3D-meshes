# Mesh Normalization, Quantization, and Error Analysis

Complete pipeline for 3D mesh preprocessing: normalization, quantization, and error analysis for AI systems like SeamGPT.

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run pipeline
python main.py

# Or interactive notebook
jupyter notebook mesh_analysis.ipynb
```

---

## 📁 Project Structure

```
├── samples/              # 8 input .obj meshes
├── src/                  # Source code
│   ├── mesh_loader.py   # Task 1: Loading & inspection
│   ├── normalization.py # Task 2: Normalization & quantization
│   └── reconstruction.py# Task 3: Reconstruction & analysis
├── outputs/              # Generated results
│   ├── normalized/      # 16 normalized meshes
│   ├── quantized/       # 16 quantized meshes
│   ├── reconstructed/   # 16 reconstructed meshes
│   ├── plots/           # 25+ visualizations
│   ├── error_summary.csv
│   └── analysis_report.txt
├── main.py              # Main script
└── mesh_analysis.ipynb  # Interactive notebook
```

---

## 💻 Usage Examples

### Process a Mesh
```python
from src.mesh_loader import MeshLoader
from src.normalization import MeshNormalizer, MeshQuantizer

# Load
loader = MeshLoader("samples/branch.obj")
vertices = loader.get_vertices()

# Normalize
normalizer = MeshNormalizer(vertices)
normalized = normalizer.minmax_normalize()

# Quantize
quantizer = MeshQuantizer(n_bins=1024)
quantized = quantizer.quantize(normalized)
```

### Analyze Errors
```python
from src.reconstruction import ErrorAnalyzer

analyzer = ErrorAnalyzer(original, reconstructed)
metrics = analyzer.get_all_metrics()
print(f"MSE: {metrics['mse']}")
```

---

## 🏆 Key Results

| Method | Avg MSE | Avg MAE | Relative Error |
|--------|---------|---------|----------------|
| **Min-Max** ✓ | 3.9×10⁻⁷ | 0.000436 | 0.19% |
| Unit Sphere | 1.1×10⁻⁶ | 0.000838 | 0.36% |

**Conclusion:** Min-Max normalization achieves best reconstruction accuracy (<1% error).

---

## 📈 Generated Outputs

- **48 mesh files** (normalized, quantized, reconstructed)
- **25+ visualizations** (error distributions, comparisons, scatter plots)
- **Statistical summary** (`error_summary.csv`)
- **Analysis report** (`analysis_report.txt`)

---

## 🐛 Troubleshooting

**Import errors:**
```bash
pip install -r requirements.txt
```

**Memory issues:**
```python
visualizer.plot_comparison_scatter(original, reconstructed, sample_size=500)
```

**Path issues:**
```python
from pathlib import Path
mesh_path = Path("samples/branch.obj")
```

---

## 🔧 Customization

**Change bin count:**
```python
N_BINS = 2048  # Higher accuracy
```

**Add custom normalization:**
```python
def custom_normalize(self):
    normalized = your_formula(self.original_vertices)
    return normalized
```

---

## 📚 Dependencies

- Python 3.8+ • NumPy • Trimesh • Open3D • Matplotlib • Scipy • Pandas

---

**Assignment for SeamGPT Company • November 2025**

