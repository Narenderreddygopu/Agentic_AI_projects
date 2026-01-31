import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

# Labels and title
plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

# Plot line
plt.plot(x, y)

# Add grid
plt.grid()

# Save the plot as PNG
plt.savefig("graph.png")

print("Graph saved as graph.png in current folder")
