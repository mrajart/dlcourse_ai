def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    all1  = ground_truth.shape[0]
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(all1):
        if prediction[i] == True and ground_truth[i] == True: tp += 1
        if prediction[i] == False and ground_truth[i] == False: tn += 1
        if prediction[i] == False and ground_truth[i] == True: fn += 1
        if prediction[i] == True and ground_truth[i] == False: fp += 1
            
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    all1  = ground_truth.shape[0]
    tp = 0
    for i in range(all1):
        if prediction[i] == ground_truth[i]: tp += 1
    return tp / all1
