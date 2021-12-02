def get_accuracy(yh,y):
    yh = [0 if i<0.5 else 1 for i in yh]
    return sum(yh==y)/len(yh)

def get_log_reg_stats(x, y, max_iters, learning_rate, batch_size = None):
    model = LogisticRegression(max_iters=max_iters, learning_rate=learning_rate)
    if batch_size is not None:
        yh =  model.fit_mini_batch_stochastic(x,y, batch_size).predict(x)
    else:
        yh =  model.fit(x,y).predict(x)
    g_norm = np.linalg.norm(model.g)
    iterations = model.iterations
    accuracy = get_accuracy(yh, y)
    norms = model.norms
    return [max_iters, learning_rate, accuracy, g_norm, iterations, norms]

def get_best_params(x, y, max_iters, learning_rates, with_norms=False):
    column_names = ["max_iters", "learning_rate", "accuracy","gradient_norm", "iterations"]
    results = pd.DataFrame(columns=column_names)
    norms = []
    i=0
    for max_iter in max_iters:
        for learning_rate in learning_rates:
            row = get_log_reg_stats(x, y, max_iter, learning_rate)
            results.loc[i] = row[:-1]
            norms.append(row[-1])
            i+=1
    if with_norms:
        return results, norms
    else:
        return results
    
def get_best_batch_size(x, y, max_iters, learning_rate, batch_sizes, with_norms=False):
    column_names = ["accuracy","gradient_norm", "iterations","batch_size"]
    results = pd.DataFrame(columns=column_names)
    norms = []
    i=0
    for batch_size in batch_sizes:
        row = get_log_reg_stats(x, y, max_iters, learning_rate, batch_size=batch_size)
        norms.append(row[-1])
        row = row[2:-1]
        row.append(batch_size)
        results.loc[i] = row
        i+=1
    if with_norms:
        return results, norms
    else:
        return results