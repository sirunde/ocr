
DATA_DIR = "data/"
TEST_DATA_FILENAME = DATA_DIR + 't10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + 'train-labels.idx1-ubyte'

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def read_images(file, n_max_images=None):
    images = []
    with open(file, 'rb') as f:
        _ = f.read(4) # magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for images_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

def read_labels(file, n_max_labels=None):
    labels = []
    with open(file, 'rb') as f:
        _ = f.read(4) # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = f.read(1)
            labels.append(label)

    return labels

def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

def extract_features(X):
    return [flatten_list(sample) for sample in X]
    return flatten_list(X)

def knn(X_train, Y_train, X_test, k=3):


def main():
    X_train = read_images(TRAIN_DATA_FILENAME, 100)
    X_test = read_images(TEST_DATA_FILENAME,100)

    print(len(X_train[0]))
    print(len(X_test[0]))


    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    print(len(X_train[0]))
    print(len(X_test[0]))

if __name__ == '__main__':
    main()