from __future__ import print_function

import csv
import numpy as np

from model import get_model
from model_vgg import get_vgg_model
from model_alex import get_alex_model
from utils import real_to_cdf, preprocess


def load_validation_data():
    """
    Load validation data from .npy files.
    """
    X = np.load('/data/heart/final/X_validate.npy')
    ids = np.load('/data/heart/final/ids_validate.npy')

    X = X.astype(np.float32)
    X /= 255

    return X, ids


def accumulate_study_results(ids, prob):
    """
    Accumulate results per study (because one study has many SAX slices),
    so the averaged CDF for all slices is returned.
    """
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    for i in range(size):
        study_id = ids[i]
        idx = int(study_id)
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]), dtype=np.float32)
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result


def submission():
    """
    Generate submission file for the trained models.
    """
    print('Loading and compiling models...')
    model_systole = get_vgg_model()
    model_diastole = get_vgg_model()

    print('Loading models weights...')
    model_systole.load_weights('weights_systole_best.hdf5')
    model_diastole.load_weights('weights_diastole_best.hdf5')

    print('Loading validation data...')
    X, ids = load_validation_data()

    print('Pre-processing images...')
    X = preprocess(X)

    batch_size = 32
    print('Predicting on validation data...')
    pred_systole = model_systole.predict(X, batch_size=batch_size, verbose=1)
    pred_diastole = model_diastole.predict(X, batch_size=batch_size, verbose=1)

    # real predictions to CDF
    cdf_pred_systole = pred_systole.cumsum(axis=-1) 
    cdf_pred_diastole = pred_diastole.cumsum(axis=-1)

    print('Accumulating results...')
    sub_systole = accumulate_study_results(ids, cdf_pred_systole)
    sub_diastole = accumulate_study_results(ids, cdf_pred_diastole)

    # write to submission file
    print('Writing submission to file...')
    fi = csv.reader(open('/data/heart/sample_submission_test.csv'))
    f = open('submission.csv', 'w')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.next())
    for line in fi:
        idx = line[0]
        key, target = idx.split('_')
        key = int(key)
        out = [idx]
        if key in sub_systole:
            if target == 'Diastole':
                out.extend(list(sub_diastole[key][0]))
            else:
                out.extend(list(sub_systole[key][0]))
        else:
            print('Miss {0}'.format(idx))
        fo.writerow(out)
    f.close()

    print('Done.')

submission()
