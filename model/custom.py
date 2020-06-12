def load_model():
    """
    Modify this method to deserialize you model if this environment's standard model
    loader cannot.  For example, if your custom model archive contains multiple pickle
    files, you must explicitly load which ever one corresponds to your serialized model
    here.

    Returns
    -------
    object, the deserialized model
    """
    return None


def predict(data, model_predict):
    """
    Modify this method to add pre and post processing for scoring calls.  For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.

    Parameters
    ----------
    data: pd.DataFrame
    model_predict: Callable[[pd.DataFrame], pd.DataFrame]

    Returns
    -------
    pd.DataFrame
    """
    # Execute any steps you need to do before scoring
    # Remove target columns if  they're in the dataset
    if "MEDV" in data:
        data.pop("MEDV")
    # Apply null value imputation
    data = data.fillna(0)
    # This method makes predictions against the raw, deserialized model
    predictions = model_predict(data)

    # Execute any steps you need to do after scoring
    # Note: To properly send predictions back to DataRobot, the returned DataFrame should contain a
    # column for each output label for classification or a single value column for regression
    return predictions
