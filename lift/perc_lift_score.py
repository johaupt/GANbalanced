import numpy as np

def perc_lift_score(y_true, y_score, percentile):
    """
    Compute the model lift from prediction scores. The lift is defined as the
    ratio of the accuracy of the model to the accuracy of the random model in
    *percentile* of the observations with the highest scores.
    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels or binary label indicators.
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers). y_score is
        supposed to be the score of the class with greater label.
    percentile : float > 0 and <= 1
        The percentile of observations over which the lift is calculated. The
        top decile lift (10%) is calculated by percentile = 0.1.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    lift : float
    """
    # Calculate overall occurance of target class in data (i.e. naive model)
    baseline = np.mean(y_true) # Note: The mean of a 0/1 vector is the ratio of 1s

    # Calculate how many observations are in the top x% based on the cutoff value and the number of observations
    idx_cutoff = int(len(y_true)*percentile)
    if percentile == 0:
        return 0
    # Sort observations by predicted probability
    y_true = y_true[np.argpartition(y_score, kth = idx_cutoff)]

    # How many hits are in the top x%
    model_hitrate = np.mean(y_true[idx_cutoff:])

    # Compare model performance to baseline
    return model_hitrate / baseline
