from sklearn.model_selection import train_test_split
from src.model import get_cnn_model
from src.utils import get_logger
from src.utils import prepare_train_data
from src.utils import convert_for_CNN
from src.utils import get_check_point
saved_path = 'saved/doodle_cnn_model_p3.hdf5'


def train(model, X_train, y_train, X_test, y_test, epochs=3):
    checkpoint = get_check_point(saved_path)
    logger = get_logger()

    model.fit(X_train,
              y_train,
              epochs=epochs,
              batch_size=64,
              validation_data=(X_test, y_test),
              callbacks=[checkpoint, logger])
    model.summary()
    return model


if __name__ == '__main__':
    X, y = prepare_train_data()
    X, y = convert_for_CNN(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
    model = get_cnn_model()
    train(model, X_train, y_train, X_test, y_test, 10)
