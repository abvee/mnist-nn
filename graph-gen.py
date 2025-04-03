import matplotlib.pyplot as plt
import numpy as np

values = np.zeros(10)
with open("graph-3.dat") as f:
	for i in range(10):
		values[i] = float(f.readline())

x = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# Create the plot
plt.figure(figsize=(12, 6))  # Set figure size for better visibility
plt.plot(x, values)
plt.xticks(x, [str(pos) for pos in x])

# Add labels and title
plt.xlabel('Alpha')
plt.ylabel('Accuracy Percentage')
plt.title('Accuracy by Tweaking Alpha Rate')

# Optional: Add grid lines for better readability
plt.grid(True, alpha=0.3)

# Save the figure if needed
plt.savefig('graph-3.jpg', dpi=300)
