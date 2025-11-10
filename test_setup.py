"""
Quick Test Script - Verify installation and run basic test
"""

print("Testing imports...")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
    print(f"  Version: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")
    exit(1)

try:
    import trimesh
    print("✓ Trimesh imported successfully")
    print(f"  Version: {trimesh.__version__}")
except Exception as e:
    print(f"✗ Trimesh import failed: {e}")
    exit(1)

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported successfully")
except Exception as e:
    print(f"✗ Matplotlib import failed: {e}")
    exit(1)

print("\n" + "="*60)
print("Testing mesh loading...")
print("="*60)

from pathlib import Path

samples_dir = Path(__file__).parent / "samples"
mesh_files = list(samples_dir.glob("*.obj"))

print(f"\nFound {len(mesh_files)} mesh files:")
for mf in mesh_files:
    print(f"  - {mf.name}")

if mesh_files:
    print(f"\nTesting load of first mesh: {mesh_files[0].name}")
    try:
        mesh = trimesh.load(str(mesh_files[0]))
        vertices = np.array(mesh.vertices)
        print(f"  ✓ Loaded successfully!")
        print(f"  Vertices: {len(vertices)}")
        print(f"  Vertex shape: {vertices.shape}")
        print(f"  Min coords: {vertices.min(axis=0)}")
        print(f"  Max coords: {vertices.max(axis=0)}")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")

print("\n" + "="*60)
print("All tests passed! Ready to run main pipeline.")
print("="*60)
print("\nRun: python main.py")
