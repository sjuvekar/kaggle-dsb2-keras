from __future__ import print_function

import os
import numpy as np
import dicom
import random
from scipy.misc import imresize

img_resize = True
img_shape = (64, 64)


def crop_resize(img):
    """
    Crop center and resize.

    :param img: image to be cropped and resized.
    """
    if img.shape[0] < img.shape[1]:
        img = img.T
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    img = crop_img
    img = imresize(img, img_shape)
    return img


def load_images(from_dir, batch_size=30, is_train=True, verbose=True):
    """
    Load images in the form study x slices x width x height.
    Each image contains batch_size time series frames so that it is ready for the convolutional network.

    :param from_dir: directory with images (train or validate)
    :param verbose: if true then print data
    """
    print('-'*50)
    print('Loading all DICOM images from {0}...'.format(from_dir))
    print('-'*50)

    current_study_sub = ''  # saves the current study sub_folder
    current_study = ''  # saves the current study folder
    current_study_images = []  # holds current study images
    ids = []  # keeps the ids of the studies
    study_to_images = dict()  # dictionary for studies to images
    total = 0
    images = []  # saves batch_size-frame-images
    from_dir = from_dir if from_dir.endswith('/') else from_dir + '/'
    for subdir, _, files in os.walk(from_dir):
        subdir = subdir.replace('\\', '/')  # windows path fix
        subdir_split = subdir.split('/')
        study_id = subdir_split[-3]
        if "sax" in subdir or "ch" in subdir:
            for f in files:
                image_path = os.path.join(subdir, f)
                if not image_path.endswith('.dcm'):
                    continue

                image = dicom.read_file(image_path)
                image = image.pixel_array.astype(float)
                image /= np.max(image)  # scale to [0,1]
                if img_resize:
                    image = crop_resize(image)

                if current_study_sub != subdir:
                    x = 0
                    try:
                        while len(images) % batch_size != 0:
                            images.append(images[x])
                            x += 1
                        if len(images) > batch_size and not is_train:
                            images = images[0:batch_size]

                    except IndexError:
                        pass
                    
                    current_study_sub = subdir
                    if is_train and len(images) > batch_size:
                      random.shuffle(images)
                      #print(current_study, ":", current_study_sub, ": found > ", batch_size, "images")
                      for i in range(0, len(images), batch_size):
                        current_study_images.append(images[i:i+batch_size])
                    else:
                      current_study_images.append(images)

                    images = []

                if current_study != study_id:
                    if current_study != "":
                        study_to_images[current_study] = np.array(current_study_images)
                        ids.append(current_study)
                        print ("Current_study = ", current_study, "Size = ", study_to_images[current_study].shape)
                        print ("Total Size of studies = ", sum(map(lambda a: a.shape[0], study_to_images.values())))
                    current_study = study_id
                    current_study_images = []
                
                images.append(image)
                if verbose:
                    if total % 1000 == 0:
                        print('Images processed {0}'.format(total))
                total += 1
    x = 0
    try:
        while len(images) % batch_size != 0:
            images.append(images[x])
            x += 1
        if len(images) > batch_size and not is_train:
            images = images[0:batch_size]
    except IndexError:
        pass

    print('-'*50)
    print('All DICOM in {0} images loaded.'.format(from_dir))
    print('-'*50)

    if is_train and len(images) > batch_size:
      random.shuffle(images)
      #print(current_study, ":", current_study_sub, ": found > ", batch_size, "images")
      for i in range(0, len(images), batch_size):
        current_study_images.append(images[i:i+batch_size])
    else:
      current_study_images.append(images)
    
    study_to_images[current_study] = np.array(current_study_images)
    ids.append(current_study)

    return ids, study_to_images


def map_studies_results():
    """
    Maps studies to their respective targets.
    """
    id_to_results = dict()
    train_csv = open('/data/heart/train.csv')
    lines = train_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, diastole, systole = item.replace('\n', '').split(',')
        id_to_results[id] = [float(diastole), float(systole)]

    return id_to_results


def write_train_npy():
    """
    Loads the training data set including X and y and saves it to .npy file.
    """
    print('-'*50)
    print('Writing training data to .npy file...')
    print('-'*50)

    study_ids, images = load_images('/data/heart/train')  # load images and their ids
    studies_to_results = map_studies_results()  # load the dictionary of studies to targets
    X = []
    y = []

    for study_id in study_ids:
        study = images[study_id]
        outputs = studies_to_results[study_id]
        for i in range(study.shape[0]):
            X.append(study[i, :, :, :])
            y.append(outputs)

    X = np.array(X, dtype=np.uint8)
    y = np.array(y)
    np.save('/data/heart/X_train.npy', X)
    np.save('/data/heart/y_train.npy', y)
    print('Done.')


def write_validation_npy():
    """
    Loads the validation data set including X and study ids and saves it to .npy file.
    """
    print('-'*50)
    print('Writing validation data to .npy file...')
    print('-'*50)

    ids, images = load_images('/data/heart/validate')
    study_ids = []
    X = []

    for study_id in ids:
        study = images[study_id]
        for i in range(study.shape[0]):
            study_ids.append(study_id)
            X.append(study[i, :, :, :])

    X = np.array(X, dtype=np.uint8)
    np.save('/data/heart/X_validate.npy', X)
    np.save('/data/heart/ids_validate.npy', study_ids)
    print('Done.')


write_train_npy()
write_validation_npy()
