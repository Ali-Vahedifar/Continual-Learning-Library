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
    
    # Stability-Plasticity Ratio (SP)
    numerator = np.sum(np.abs(np.diag(acc_matrix)[1:] - np.diag(acc_matrix, k=-1)[1:]))
    denominator = np.sum(np.abs(np.diag(acc_matrix)[1:] - acc_matrix[1:, 0]))
    SP = numerator / denominator if denominator != 0 else 0
    
    # Task Retention Ratio (TR)
    TR = np.sum(acc_matrix[:, 0]) / np.sum(acc_matrix[0, :]) if np.sum(acc_matrix[0, :]) != 0 else 0
    
    metrics = {
        "ACC": ACC,
        "BWT": BWT,
        "FWT": FWT,
        "AF": AF,
        "I": I,
        "SP": SP,
        "TR": TR
    }
    
    return metrics

# Example usage:
# acc_matrix = np.array([...])  # Your accuracy matrix (20x20)
# b = np.array([...])  # Baseline accuracy for FWT
# optimal_result = 0.9  # Example optimal performance
# metrics = compute_metrics(acc_matrix, b, optimal_result)
# print(metrics)

    
