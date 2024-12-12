import numpy as np
def fuzzy_union(A, B):
# """Perform fuzzy union (max) operation on two fuzzy sets."""
   return np.maximum(A, B)
def fuzzy_intersection(A, B):
# """Perform fuzzy intersection (min) operation on two fuzzy sets."""
   return np.minimum(A, B)
def fuzzy_complement(A):
# """Perform fuzzy complement operation on a fuzzy set."""
   return 1 - A
def max_min_composition(R, S):
# """Perform max-min composition on two fuzzy relations."""
    result = np.zeros((R.shape[0], S.shape[1]))
    for i in range(R.shape[0]):
        for j in range(S.shape[1]):
            result[i, j] = np.max(np.minimum(R[i, :], S[:, j]))
    return result
# Example usage
if __name__ == "__main__":
# Define fuzzy sets
    A = np.array([0.2, 0.5, 0.8, 1.0])
    B = np.array([0.3, 0.6, 0.7, 0.9])
    print("Fuzzy Set A:", A)
    print("Fuzzy Set B:", B)
# Perform fuzzy operations
    print("Union (A ∪ B):", fuzzy_union(A, B))
    print("Intersection (A ∩ B):", fuzzy_intersection(A, B))
    print("Complement of A:", fuzzy_complement(A))
# Define fuzzy relations
    R = np.array([[0.2, 0.5, 0.1],[0.3, 0.8, 0.4],[0.7, 0.2, 0.6]])
    S = np.array([[0.6, 0.1],[0.2, 0.7],[0.9, 0.5]])
    print("\nFuzzy Relation R:")
    print(R)
    print("\nFuzzy Relation S:")
    print(S)
    # Perform max-min composition
    result = max_min_composition(R, S)
    print("\nMax-Min Composition (R ∘ S):")
    print(result)