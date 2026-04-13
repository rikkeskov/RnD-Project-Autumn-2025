""" import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("shrink_results.csv")  # Replace with your file path

# Group by base_epsilon
grouped = df.groupby("base_epsilon")

# Create a figure
plt.figure(figsize=(10, 6))

# Plot each group
for name, group in grouped:
    plt.plot(
        group["epsilon_pct"],
        group["compression_ratio"],
        marker="o",
        label=f"Base Epsilon = {name}"
    )

# Add labels and title
plt.xlabel("Epsilon Percent")
plt.ylabel("Compression Ratio")
plt.title("Compression Ratio vs. Epsilon Percent for Different Base Epsilons")
plt.legend()
plt.grid(True)

# Show the plot
plt.show() """

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("data/shrink_results_overviews/HouseholdPowerConsumption1_TEST_dim0_results.csv")  # Replace with your file path

# Create a 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract data
epsilon_pct = df["epsilon_pct"]
base_epsilon = df["base_epsilon"]
compression_ratio = df["compression_ratio"]

# Scatter plot in 3D
scatter = ax.scatter(
    epsilon_pct,
    base_epsilon,
    compression_ratio,
    c=compression_ratio,
    cmap='viridis',
    s=50,
    alpha=0.8
)

# Add labels and title
ax.set_xlabel("Epsilon Percent")
ax.set_ylabel("Base Epsilon")
ax.set_zlabel("Compression Ratio")
ax.set_title("3D Plot: Compression Ratio vs. Epsilon Percent and Base Epsilon")

# Add a color bar
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label("Compression Ratio")

# Show the plot
plt.tight_layout()
plt.show()