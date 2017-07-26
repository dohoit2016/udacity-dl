from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.'


def download_progress_hook(count, blockSize, totalSize):
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)
    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(" ")
            sys.stdout.flush()
        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception('Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception('Expected %d folders, one per class. Found %d instead.' % (num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
# prop 1
display(Image(filename='notMNIST_large/A/Q29tZW5pdXNBbnRpcXVhIE1lZGl1bS50dGY=.png'))

image_size = 28
pixel_depth = 255.0

print("save data to pickle file...")


def load_letter(folder, min_num_images):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth

            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read: ' + image_file, ':', e, '- it\'s ok, skipping.')
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images then expected')
    print("Full dataset tensor: ", dataset.shape)
    print("Mean: ", np.mean(dataset))
    print("Standard deviation: ", np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + ".pickle"
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print("%s already present - Skipping picking." % set_filename)
        else:
            print("picking %s." % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    # read as binary
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names


train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

print("Do dai train_datasets: %d" % len(train_datasets))
print("Do dai tap test: %d" % len(test_datasets))
print("Get data done!")


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size * 10, image_size)  # 10k valid
    train_dataset, train_labels = make_arrays(train_size, image_size)  # 200k train
    vsize_per_class = valid_size  # 10k
    tsize_per_class = train_size  # 200k

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, 0
    end_l = vsize_per_class + tsize_per_class  # 210k
    count = 0
    length_train = 0
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                count += 1
                print("Count file: %d" % count)
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                    print("index of valid: %d %d, index of train: %d %d" % (start_v, end_v, start_t, end_t))
                len_train_letter = len(letter_set) - vsize_per_class
                length_train += len_train_letter
                train_letter = letter_set[vsize_per_class:len(letter_set), :, :]
                print("Len train letter of file: %d" % len(train_letter))
                end_t += len_train_letter
                print("Len train dataset of file: %d" % len(train_dataset[start_t:end_t, :, :]))
                print("vsize_per_class: %d, len a file: %d" % (vsize_per_class, len(letter_set)))
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += len_train_letter
                print("----------------------------------------------")
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    print('Len train dataset: %d' % length_train)
    return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 429114

valid_size = 10000

test_size = 18724

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffle_dataset = dataset[permutation, :, :]
    shuffle_labels = labels[permutation]
    return shuffle_dataset, shuffle_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# prop 4
# plt.title(train_labels[0])
# plt.imshow(train_dataset[0], cmap="Greys_r")
# plt.show()

# pickle_file = os.path.join(data_root, 'notMNIST.pickle')

# try:
# 	f = open(pickle_file, 'wb')
# 	save = {
# 		'train_dataset': train_dataset,
# 		'train_labels': train_labels,
# 		'valid_dataset': valid_dataset,
# 		'valid_labels': valid_labels,
# 		'test_dataset': test_dataset,
# 		'test_labels': test_labels
# 	}
# 	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
# 	f.close()
# except Exception as e:
# 	print("File is not saved")
# 	raise

# statinfo = os.stat(pickle_file)
# print("File size: ", statinfo.st_size)

# Prediction

train_num = 5000
X_train = train_dataset[:train_num].reshape(-1, 28*28)
y_train = train_labels[:train_num]

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

valid_num = 1000
X_valid = valid_dataset[:valid_num].reshape(-1, 28*28)
y_valid = valid_labels[:valid_num]

X_test = test_dataset.reshape(-1,28*28)
y_test = test_labels

# valid_acc = sum(logreg.predict(X_valid) == y_valid)/len(y_valid)
valid_acc = sum(logreg.predict(X_valid) == y_valid)/valid_num
print("accuracy rate: %f" % valid_acc)

# test_acc = sum(logreg.predict(X_test) == y_test)/len(y_test)
# print("accuracy rate: %f" % test_acc)

