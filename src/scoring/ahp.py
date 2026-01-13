import numpy as np

class AHPScorer:
    """
    Implements Analytical Hierarchy Process (AHP) for weighting ethical criteria.
    """
    
    def __init__(self, criteria=None):
        self.criteria = criteria if criteria else ["Fairness", "Transparency", "Privacy", "Accountability"]
        self.num_criteria = len(self.criteria)
        self.comparison_matrix = np.ones((self.num_criteria, self.num_criteria))
        self.weights = None
        self.consistency_ratio = None
        
    def set_comparison_matrix(self, matrix):
        """
        Sets the pairwise comparison matrix.
        Matrix should be symmetric with 1s on diagonal.
        If A is x times more important than B, then B is 1/x times more important than A.
        """
        matrix = np.array(matrix)
        if matrix.shape != (self.num_criteria, self.num_criteria):
            raise ValueError(f"Matrix must be {self.num_criteria}x{self.num_criteria}")
            
        self.comparison_matrix = matrix
        
    def calculate_weights(self):
        """
        Calculates weights using the Eigenvector method.
        Returns:
            dict: {criterion: weight}
        """
        # Calculate eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(self.comparison_matrix)
        
        # Max eigenvalue
        max_eig_prob = np.max(eigvals)
        
        # Corresponding eigenvector (principal eigenvector)
        principal_eigvec = eigvecs[:, np.argmax(eigvals)]
        
        # Normalize to sum to 1 (take real part as it might be complex due to slight inconsistencies)
        principal_eigvec = np.real(principal_eigvec)
        self.weights = principal_eigvec / np.sum(principal_eigvec)
        
        # Calculate Consistency Ratio (CR)
        # CI = (lambda_max - n) / (n - 1)
        ci = (max_eig_prob - self.num_criteria) / (self.num_criteria - 1)
        
        # Random Index (RI) table for n=1 to 10
        ri_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = ri_dict.get(self.num_criteria, 1.49)
        
        self.consistency_ratio = np.real(ci / ri) if ri != 0 else 0.0
        
        return dict(zip(self.criteria, self.weights))

    def get_consistency_ratio(self):
        return self.consistency_ratio
