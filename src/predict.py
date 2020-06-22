import numpy as np
from src import model as dr_model
from src import utils

cnn_weight_path = 'saved/doodle_cnn_model.hdf5'
lstm_weight_path = 'saved/doodle_model.hdf5'
predict_image_path = 'predict/demo.jpg'


def get_list_class_names():
    list_classes = utils.get_list_train_npy()
    list_class_names = [class_path.rsplit('/', 1)[1].replace('.npy', '') for class_path in list_classes]
    return list_class_names


def predict_from_file(image_path):
    img_data = np.array(utils.image_prepare(image_path))
    img_data = img_data.reshape([28, 28, 1])
    img_data = np.array([img_data])

    model = dr_model.get_cnn_model()
    model.load_weights(cnn_weight_path)
    list_names = get_list_class_names()

    prob = model.predict(np.array([img_data]))[0]
    index = np.argmax(prob)

    return {
        'probability': float(prob[index]),
        'object': list_names[index]
    }


def predict_image(img):
    model = dr_model.get_cnn_model()
    model.load_weights(cnn_weight_path)
    list_names = get_list_class_names()
    img_data = utils.pre_process(img)
    prob = model.predict(np.array([img_data]))[0]
    index = np.argmax(prob)

    return {
        'probability': str(prob[index]),
        'object': list_names[index]
    }



