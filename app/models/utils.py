def to_labels(
        data,
        threshold=0.5
):
    ypred = []
    for pred in data:
        if pred >= threshold:
            ypred.append('SUBJ')
        else:
            ypred.append('OBJ')
    return ypred
