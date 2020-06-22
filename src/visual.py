from src.utils import get_list_train_npy
from src.utils import load_data
from src.utils import prepare_train_data
import matplotlib.pyplot as plt
import pandas as pd


def view_data_count():
    data = dict()
    list_names = get_list_train_npy()
    for index, full_name in enumerate(list_names, 0):
        name = full_name.rsplit('/', 1)[1]
        class_data = load_data(name)
        class_name = name.replace(".npy", "")
        data[class_name] = len(class_data)
        print(class_name, len(class_data))

    plt.bar(data.keys(), data.values())
    plt.show()


def train_history():
    data = pd.read_csv('../saved/training.log')

    print(data[
              'accuracy'
          ])

    plt.plot(data['accuracy'])
    plt.plot(data['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    # X, y = prepare_train_data()
    # print(y.value_counts())
    # view_data_count()
    train_history()

