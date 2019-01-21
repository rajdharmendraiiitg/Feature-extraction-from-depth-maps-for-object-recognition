import os
from os.path import join
from os import listdir
from plot_hog import hog_calculator
# from hog import hog_from_path
import numpy as np
import cv2
# import cv
from collections import OrderedDict
from os.path import isfile

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from lbp import lbp_calculator
# from sklearn.metrics import Confusion_Matrix


def num_to_label(array, folder):
    class_names = [i for i in sorted(listdir(folder))]
    j = 1
    # class_labels = [i for i in range(1, len(class_names) + 1)]
    print(class_names)
    num_labels = [class_names[array[i] - 1] for i in range(0, len(array))]
    return num_labels


def missclassification_rate(pred_names, truth_names):
    count = 0
    # pred_names = num_to_label(num_to_label(pred, train_folder))
    # truth_names = num_to_label(num_to_label(truth, test_folder))
    for i in range(0, len(pred_names)):
        if truth_names[i] != pred_names[i]:
            count = count + 1
        i = i + 1
    return count


def one_hot(a, num_classes):
    vector = np.zeros(shape=(num_classes, 1))
    vector[a][0] = 1
    return vector


def one_hot_to_integer(a):
    for i in range(0, a.shape[0]):
        if a[i][0] == 1:
            break
        i += 1

    return i


def getFilesInDir(foldername='Dataset'):
    file_map = OrderedDict()

    for f in listdir(foldername):
        # print(f)
        for h in listdir(join(foldername, f)):
            # print(h)
            for k in listdir(join(join(foldername, f), h)):
                file_path = join(join(join(foldername, f), h), k)
                if f not in file_map.keys():
                    file_map[f] = [file_path]
                else:
                    file_map[f].append(
                        file_path)

    # print("Getting the names of the files in the directory:\n" +
    # str(all_files))
    # for i in file_map.keys():
        # print(len(file_map[i]))
    return (file_map)


def create_labels(file_map, output_classes):
    labels = []
    idx = 1
    for i in file_map:
        p = file_map[i]
        for j in p:
            categorical_labels = one_hot(idx, output_classes)
            labels.append(categorical_labels)
        idx = idx + 1
    return labels


def test_train_split(file_map, num):
    test_file_map = OrderedDict()
    train_file_map = OrderedDict()
    for i, j in file_map.items():
        train_file_map[i] = j[:-num]
        test_file_map[i] = j[-num:]

    return (train_file_map, test_file_map)


# def images_load(foldername):
#     a = []
#     b = []
#     for f in listdir(foldername):
#         if f.endswith('_crop.png'):
#             a.append((join(foldername, f)))
#         if f.endswith('_depthcrop.png'):
#             b.append(join(foldername, f))
#     return (a, b)


def images_load(file_map):
    index = 0
    rgb_images = []
    depth_images = []
    rgb_labels = []
    depth_labels = []

    for i in file_map:
        for k in file_map[i]:
            if k.endswith('_crop.png'):
                rgb_images.append(k)
                rgb_labels.append(index)
            if k.endswith('_depthcrop.png'):
                depth_images.append(k)
                depth_labels.append(index)
        index += 1
    return (rgb_images, rgb_labels, depth_images, depth_labels)


def image_loader(file_list):
    for i in file_list:
        im = Image.read()


def rgb_image_to_hog_features(image_list):
    feature_list = []
    j = 0
    for i in image_list:
        print(j)
        p = (hog_calculator(i))
        print("HOG Feature Calculated RGB Based for>> " + str(i))
        p = p.reshape((1, 228))
        feature_list.append(p)
        j += 1
    return feature_list


def depth_image_to_hog(image_list):
    feature_list = []
    for i in image_list:
        print(i)
        f = (hog_calculator(i))
        # print(f)
        print("HOG Feature Calculated Depth Based for>> " + str(i))
        f = f.reshape((1, 228))
        feature_list.append(f)
    return feature_list


def rgb_image_to_lbp(image_list):
    feature_list = []
    j = 0
    for i in image_list:
        print(j)
        p = (lbp_calculator(i))
        print("LBP Feature Calculated RGB Based for>> " + str(i))
        p = p.reshape((1, 228))
        feature_list.append(p)
        j += 1
    return feature_list


def depth_image_to_lbp(image_list):
    feature_list = []
    j = 0
    for i in image_list:
        print(j)
        p = (lbp_calculator(i))
        print("LBP Feature Calculated For Depth Image >> " + str(i))
        p = p.reshape((1, 228))
        feature_list.append(p)
        j += 1
    return feature_list


def hybrid_feature_extractor(lbp_feature, hog_feature):
    hybrid_feature = []
    for i in range(0, len(lbp_feature)):
        hybrid_feature.append(
            float(lbp_feature[i] + hog_feature[i]) / float(2))
    return hybrid_feature

# def data_preprocessor(features, labels):
#     create_feature_map(folder)


# def train_classifier(X, Y):
#     svm = cv2.ml.SVM_create()
#     svm.setType(cv2.ml.SVM_C_SVC)
#     svm.setKernel(cv2.ml.SVM_LINEAR)
#     # s = SVM()
#     # params = dict(kernel_type=SVM_LINEAR,
#     #               svm_type=SVM_C_SVC, C=1)
#     svm.train(X, Y)
#     return svm

def baseline_train_svm_classifier(X, Y):
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf.fit(X, Y)
    return clf


def baseline_predict_svm_classifier(X_test, model):
    return model.predict(X_test)
    # return np.float32([model.predict(x) for x in X_test])


def baseline_train_randomforest_classifier(X, Y):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X, Y)
    return clf


def baseline_train_randomforest_predictor(X, model):
    return model.predict(X)


if __name__ == '__main__':
    # (a, b) = images_load('apple_1')
    # features_list = rgb_image_to_hog_features(a)
    # print(len(features_list))
    # print(features_list[0].shape)
    # print(len(a))
    # print(len(b))

    # class_1_rgb = images_load()

    file_map = getFilesInDir('dataset')

    train_file_map, test_file_map = test_train_split(file_map, 50)

    (a, b, c, d) = images_load(train_file_map)
    (a_test, b_test, c_test, d_test) = images_load(test_file_map)

    print("Total Images RGB Loaded:>>" + str(len(a)))
    print("Total Labels RGB Loaded:>>" + str(len(b)))
    print("Total Depth Images  Loaded:>>" + str(len(c)))
    print("Total Depth Labels  Loaded:>>" + str(len(d)))

    # print(len(b))
    # print(len(c))
    # # sdgdg
    print("HOG Features Analysis.......\n")

    rgb_hog_features = rgb_image_to_hog_features(a)
    rgb_hog_features_test = rgb_image_to_hog_features(a_test)
    depth_hog_features = depth_image_to_hog(c)
    depth_hog_features_test = depth_image_to_hog(c_test)

    print(len(rgb_hog_features))
    print(len(depth_hog_features))
    print(rgb_hog_features[0].shape)
    print(len(depth_hog_features))

    rgb_hog_features = np.concatenate(rgb_hog_features, axis=0)
    rgb_hog_features_test = np.concatenate(rgb_hog_features_test, axis=0)
    b = np.array(b).reshape(len(b), -1)
    print(rgb_hog_features.shape)
    # .reshape(
    # len(rgb_hog_features), -1)

    depth_hog_features = np.concatenate(depth_hog_features, axis=0)
    depth_hog_features_test = np.concatenate(depth_hog_features_test, axis=0)
    d = np.array(d).reshape(len(d), -1)
    print(rgb_hog_features.shape)
    # .reshape(
    # len(depth_hog_features), -1)

    print(rgb_hog_features.shape)
    print(b.shape)
    print(depth_hog_features.shape)
    print(d.shape)
    model1 = baseline_train_svm_classifier(rgb_hog_features, b)
    acc1 = baseline_predict_svm_classifier(rgb_hog_features_test, model1)
    # print(acc1)
    print(missclassification_rate(acc1, b_test))

    # print(d)
    model2 = baseline_train_svm_classifier(depth_hog_features, d)
    acc2 = baseline_predict_svm_classifier(depth_hog_features_test, model2)
    print(acc2)
    print(missclassification_rate(acc2, d_test))

    print("LBPH Analysis ..........\n")

    (a, b, c, d) = images_load(train_file_map)
    (a_test, b_test, c_test, d_test) = images_load(test_file_map)

    rgb_lbp_features = rgb_image_to_lbp(a)
    rgb_lbp_features_test = rgb_image_to_lbp(a_test)
    depth_lbp_features = depth_image_to_lbp(c)
    depth_lbp_features_test = depth_image_to_lbp(c_test)

    print(len(rgb_lbp_features))
    print(len(depth_lbp_features))
    print(rgb_lbp_features[0].shape)
    print(len(depth_lbp_features))

    rgb_lbp_features = np.concatenate(rgb_lbp_features, axis=0)
    rgb_lbp_features_test = np.concatenate(rgb_lbp_features_test, axis=0)
    b = np.array(b).reshape(len(b), -1)
    print(rgb_lbp_features.shape)
    # .reshape(
    # len(rgb_lbp_features), -1)

    depth_lbp_features = np.concatenate(depth_lbp_features, axis=0)
    depth_lbp_features_test = np.concatenate(depth_lbp_features_test, axis=0)
    d = np.array(d).reshape(len(d), -1)
    print(rgb_lbp_features.shape)
    # .reshape(
    # len(depth_lbp_features), -1)

    print(rgb_lbp_features.shape)
    print(b.shape)
    print(depth_lbp_features.shape)
    print(d.shape)
    model1 = baseline_train_svm_classifier(rgb_lbp_features, b)
    acc1 = baseline_predict_svm_classifier(rgb_lbp_features_test, model1)
    # print(acc1)
    print(missclassification_rate(acc1, b_test))

    # print(d)
    model2 = baseline_train_svm_classifier(depth_lbp_features, d)
    acc2 = baseline_predict_svm_classifier(depth_lbp_features_test, model2)
    print(acc2)
    print(missclassification_rate(acc2, d_test))
    # print(d)
    # print(filemap['apple'])

    print("Hybrid Analysis............\n")

    hybrid_features_rgb = hybrid_feature_extractor(
        rgb_hog_features, rgb_lbp_features)
    hybrid_features_depth = hybrid_feature_extractor(
        depth_hog_features, depth_lbp_features)

    hybrid_features_rgb_test = np.concatenate(hybrid_features_rgb, axis=0)
    d = np.array(d).reshape(len(d), -1)
    print(hybrid_features_rgb_test.shape)
    # .reshape(
    # len(depth_lbp_features), -1)

    print(hybrid_features_rgb.shape)
    print(b.shape)
    print(hybrid_features_rgb_test.shape)
    print(d.shape)
    model1 = baseline_train_svm_classifier(hybrid_features_rgb, b)
    acc1 = baseline_predict_svm_classifier(
        hybrid_features_rgb_test, model1)
    # print(acc1)
    print(missclassification_rate(acc1, b_test))
