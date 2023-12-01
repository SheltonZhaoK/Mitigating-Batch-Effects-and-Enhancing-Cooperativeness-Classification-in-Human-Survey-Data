import os

def create_mlp_args_adatively(configs, trainSize, numLayers):
    arg_dict = configs["Classifiers"]["mlp"]
    arg_dict['hidden_layer_sizes'] = []
    numParameters = trainSize // 5

    for i in range(numLayers):
        i+=1
        # eg: numParam = 100, numLayers = 2,  solve equation x + 0.5x = 100
        # eg: numParam = 100, numLayers = 3,  solve equation x + 0.5x + (0.5^2)x= 100
        x = numParameters // sum([(1/2) ** i for i in range(i)]) 
        hidden_layer_sizes = []
        for j in range(i):
            layer_size = round(((1/2)**j) * x )
            hidden_layer_sizes.append(layer_size)
        arg_dict['hidden_layer_sizes'].append(tuple(hidden_layer_sizes))
    print(arg_dict)
    return arg_dict

def format_gridSearch_results(searcher, outputFile):
    cv_results = searcher.cv_results_
    mean_test_scores = cv_results['mean_test_score']
    params = cv_results['params']
    std_test_scores = cv_results['std_test_score']
    fit_times = cv_results['mean_fit_time']
    std_fit_times = cv_results['std_fit_time']

    sorted_results = sorted(zip(mean_test_scores, params, std_test_scores, fit_times, std_fit_times), key=lambda x: x[0], reverse=True)

    if not os.path.exists("../output"):
        os.makedirs("../output")
    with open(outputFile, 'w') as f:
        for score, param, std, fit_time, std_fit_time in sorted_results:
            f.write(f"Score: {score:.3f} +- {std:.3f}, Fit Time: {fit_time:.3f} +- {std_fit_time:.3f}, Parameters: {param}\n")
        print(f"{searcher.best_estimator_} search results saved to {outputFile}")

