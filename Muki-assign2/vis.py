import matplotlib.pyplot as plt

# Data from your runs
processes = [4, 8, 16, 32, 64]
times = [0.120, 0.068, 0.041, 0.023, 0.015]

# Create figure and plot
plt.figure(figsize=(10, 6))
plt.plot(processes, times, marker='o', linewidth=2, markersize=8, color='blue')

# Labels and title
plt.xlabel('Number of Processes', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.title('1024x1024 Jacobi2D: Execution Time vs Number of Processes\n(500 iterations)', fontsize=14)
plt.grid(True, alpha=0.3)

# Set x-axis to show all process counts
plt.xticks(processes)

# Save as PDF
plt.savefig('performance_plot.pdf', format='pdf', bbox_inches='tight')
print("Plot saved to performance_plot.pdf")
plt.close()