import torch

# Function to determine task change points and number of tasks
def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)  # Number of unique tasks
    changes = []  # Store indices where task changes occur
    current = result_t[0]  # Track current task
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)  # Record change index
            current = t  # Update current task
    return n_tasks, changes  # Return total tasks and change indices

# Function to compute confusion matrix and associated statistics
def confusion_matrix(result_t, result_a, optimal_result, fname=None):
    nt, changes = task_changes(result_t)  # Get number of tasks and change indices
    
    baseline = result_a[0]  # Baseline performance before learning
    changes = torch.LongTensor(changes + [result_a.size(0)]) - 1  # Append last index to changes
    result = result_a[changes]  # Extract results at change points

    acc = result.diag().mean()  # Compute average accuracy (ACC)
    fin = result[nt - 1]  # Final accuracy across all tasks
    bwt = (result[nt - 1] - result.diag()).mean()  # Compute backward transfer (BWT)
    
    fwt = torch.zeros(nt)  # Initialize forward transfer (FWT)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]  # Compute forward transfer
    fwt = fwt.mean()
    
    max_acc = result.max(dim=0)[0]  # Get max accuracy per task
    af = (max_acc[:-1] - result[-1, :-1]).mean()  # Compute average forgetting (AF)
    
    i_t = (optimal_result - result[-1, -1]).mean()  # Compute intransigence measure (I)
    
    numerator = sum(abs(result[k, k] - result[k, k - 1]) for k in range(1, nt))
    denominator = sum(abs(result[k, k] - result[k, 0]) for k in range(1, nt))
    sp = numerator / denominator if denominator != 0 else 0  # Compute Stability-Plasticity Ratio (SP)
    
    tr = (result[:, 0].sum() / result[0].sum()).item()  # Compute Task Retention Ratio (TR)

    if fname is not None:
        f = open(fname, 'w')  # Open file for writing
        
        print(' '.join(['%.4f' % r for r in baseline]), file=f)  # Write baseline
        print('|', file=f)  # Separator
        for row in range(result.size(0)):
            print(' '.join(['%.4f' % r for r in result[row]]), file=f)  # Write matrix rows
        print('', file=f)
        print('Final Accuracy: %.4f' % fin.mean(), file=f)  # Write final accuracy
        print('Backward: %.4f' % bwt, file=f)  # Write backward transfer
        print('Forward:  %.4f' % fwt, file=f)  # Write forward transfer
        print('Forgetting: %.4f' % af, file=f)  # Write average forgetting
        print('Intransigence: %.4f' % i_t, file=f)  # Write intransigence
        print('Stability-Plasticity Ratio: %.4f' % sp, file=f)  # Write SP ratio
        print('Task Retention Ratio: %.4f' % tr, file=f)  # Write TR ratio
        f.close()  # Close file
    
    stats = [acc, bwt, fwt, af, i_t, sp, tr]  # Store computed statistics
    return stats  # Return computed statistics
