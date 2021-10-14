
DATA_DIR = "data/"
TEST_DATA_FILENAME = DATA_DIR + 't10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + 't10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + 'train-labels-idx1-ubyte'

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

def main():
    X_train = read_images(TRAIN_DATA_FILENAME, 100)

    print(len(X_train))

if __name__ == '__main__':
    main()