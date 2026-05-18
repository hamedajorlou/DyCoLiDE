from core import generate_structure, sem_generator
import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Generate a random DAG (Erdős–Rényi with 5 nodes)
dag = generate_structure(num_nodes=10, degree=2, graph_type='erdos-renyi')

# Step 2: Define the schema of variable types
schema = {
    'X0': 'continuous',
    'X1': 'binary',
    'X2': 'categorical:3',
    'X3': 'continuous',
    'X4': 'count'
}

# Step 3: Generate synthetic data
data = sem_generator(graph=dag, schema=schema, n_samples=1000, seed=42)

# Step 4: Visualize the generated DAG
nx.draw_networkx(dag, with_labels=True)
plt.title("Generated Causal DAG")
plt.show()

# Step 5: View sample data
print(data.head())
