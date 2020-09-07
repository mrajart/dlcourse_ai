def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    all1  = ground_truth.shape[0]
    tp = 0
    for i in range(all1):
        if prediction[i] == ground_truth[i]: tp += 1
    return tp / all1
