import numpy as np

def compute_metrics(acc_matrix, b, optimal_result):
    T = acc_matrix.shape[0]  # Number of tasks
    
    # Average Accuracy (ACC)
    ACC = np.mean(acc_matrix[-1, :])
    
    # Backward Transfer (BWT)
    BWT = np.mean([acc_matrix[-1, i] - acc_matrix[i, i] for i in range(T - 1)])
    
    # Forward Transfer (FWT)
    FWT = np.mean([acc_matrix[i - 1, i] - b[i] for i in range(1, T)])
    
    # Average Forgetting (AF)
    max_acc = np.max(acc_matrix[:-1, :], axis=0)
    AF = np.mean(max_acc[:-1] - acc_matrix[-1, :-1])
    
    # Intransigence Measure (I)
    I = optimal_result - acc_matrix[-1, -1]
    
    # Plasticity-Stability-Ratio 
    numerator = sum(acc_matrix[k, k] -acc_matrix[k - 1, k] for k in range(1, T))
    denominator = sum(abs(acc_matrix[ - 1, k] - acc_matrix[k, k]) for k in range(T - 1))

    PS= numerator / denominator if denominator != 0 else 0

    
    metrics = {
        "ACC": ACC,
        "BWT": BWT,
        "FWT": FWT,
        "AF": AF,
        "I": I,
        "PS": PS,
       
    }
    
    return metrics

# Example usage:
# acc_matrix = np.array([...])  # Your accuracy matrix (20x20)
# b = np.array([...])  # Baseline accuracy for FWT
# optimal_result = 0.9  # Example optimal performance
# metrics = compute_metrics(acc_matrix, b, optimal_result)
# print(metrics)

    
