import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic data
n_samples = 300  # Number of samples
n_features = 2   # Number of features (dimensions)
n_clusters = 5   # Number of clusters
# Create synthetic data using make_blobs
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Define the Particle class
class Particle:
    def __init__(self, n_clusters, n_features):
        # Initialize the position of the particle (centroids)
        self.position = np.random.rand(n_clusters, n_features)
        # Initialize the velocity of the particle
        self.velocity = np.random.randn(n_clusters, n_features) * 0.1
        # Best position found by the particle
        self.best_position = self.position.copy()
        # Best score (fitness) found by the particle
        self.best_score = float('inf')

# Fitness function to evaluate the particle's position
def fitness(particle, X):
    # Calculate distances from each point to each centroid
    distances = np.sqrt(((X[:, np.newaxis, :] - particle.position[np.newaxis, :, :]) ** 2).sum(axis=2))
    # Get the minimum distance for each point to the nearest centroid
    min_distances = distances.min(axis=1)
    # Return the mean of the minimum distances (lower is better)
    return np.mean(min_distances)

# Update the velocity of the particle
def update_velocity(particle, global_best_position, w=0.5, c1=1, c2=1):
    r1, r2 = np.random.rand(2)  # Random coefficients for exploration
    # Cognitive component: attraction to the particle's best position
    cognitive = c1 * r1 * (particle.best_position - particle.position)
    # Social component: attraction to the global best position
    social = c2 * r2 * (global_best_position - particle.position)
    # Update the particle's velocity
    particle.velocity = w * particle.velocity + cognitive + social

# Update the position of the particle
def update_position(particle):
    # Update the position based on the velocity
    particle.position += particle.velocity
    # Ensure the position is within bounds [0, 1]
    particle.position = np.clip(particle.position, 0, 1)

# Main PSO clustering function
def pso_clustering(X, n_clusters, n_particles=20, n_iterations=100):
    # Initialize particles
    particles = [Particle(n_clusters, X.shape[1]) for _ in range(n_particles)]
    # Initialize global best position and score
    global_best_position = particles[0].position.copy()
    global_best_score = float('inf')

    # Main loop for the number of iterations
    for iteration in range(n_iterations):
        for particle in particles:
            # Evaluate the fitness of the particle
            score = fitness(particle, X)
            # Update the particle's best score and position if the current score is better
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            # Update the global best score and position if the current score is better
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()

        # Update the velocity and position of each particle
        for particle in particles:
            update_velocity(particle, global_best_position)
            update_position(particle)

        # Print the best score every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best score = {global_best_score:.4f}")

    return global_best_position  # Return the best centroids found

# Run PSO clustering
best_centroids = pso_clustering(X, n_clusters)

# Assign points to clusters based on the best centroids
distances = np.sqrt(((X[:, np.newaxis, :] - best_centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
cluster_assignments = distances.argmin(axis=1)  # Get the index of the closest centroid for each point

# Visualize the results
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis')  # Scatter plot of data points
plt.scatter(best_centroids[:, 0], best_centroids[:, 1], c='red', marker='x', s=200, linewidths=3)  # Plot centroids
plt.title('PSO Clustering Results')
plt.xlabel('Feature  1')
plt.ylabel('Feature 2')
plt.show()  # Display the plot

# Print completion message and final centroids
print("\nClustering complete!")
print(f"Final centroids:\n{best_centroids}")