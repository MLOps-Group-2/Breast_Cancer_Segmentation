from breast_cancer_segmentation.trainer import train_model


def test_train_model():
    assert train_model.training_step() == "Hello World"