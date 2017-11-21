import csv
import glob
from segmentation import segmentation
from features import get_features
import cv2
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

def save_model(cls, filename):
    pickle.dump(cls, open(filename, 'wb'))


def load_model(filename):
    return pickle.load(open(filename, 'rb'))

if __name__ == "__main__":

    #UNCOMMENT TO BUILD DATASET
    # images = glob.glob("./img/*.JPG")
    #
    # imgId=0
    # with open('dataset/data.csv','w') as f:
    #     writer=csv.writer(f)
    #     writer.writerow(['id','width', 'height', 'area_ratio', 'box_shape', 'total_corners', 'circularity','corner/area','classified'])
    #     for img in images:
    #         segm,input, out = segmentation(img)
    #
    #         for segment in segm:
    #             x, y, w, h = segment
    #             seg_img = out[y:y+h, x:x+w]
    #             features = get_features(seg_img)
    #             name='./dataset/img/'+str(imgId)+'.png'
    #             cv2.imwrite(name, input[y:y+h, x:x+w])  # write segmented image
    #             writer.writerow([imgId,seg_img.shape[1],seg_img.shape[0],features[0],features[1],features[2],features[3],features[4],'NONE'])
    #             imgId+=1
    trained_data = pd.read_csv('./dataset/final_data.csv')
    #print(trained_data.head())
    trained_data['classified'] = trained_data['classified'].str.replace('NUT', '1')
    trained_data['classified'] = trained_data['classified'].str.replace('BOLT', '2')
    trained_data['classified'] = trained_data['classified'].str.replace('WASHER', '3')

    #dropped the column no longer needed, DO NOT WRITE
    for col in ['id','width','height']:
         trained_data = trained_data.drop(col, axis=1)
    trained_data['classified'] = trained_data['classified'].apply(pd.to_numeric)
    #print(trained_data)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(trained_data, trained_data['classified']):
        strat_train_set = trained_data.loc[train_index]
        strat_test_set = trained_data.loc[test_index]
    train_set_y = strat_train_set["classified"]
    train_set_x = strat_train_set.drop('classified', axis=1)
    # print("-----------SET-------------")
    # print(train_set_y.head())
    # print(train_set_x.head())

    #train model
    forest_clf = RandomForestClassifier(random_state=42)
    score = cross_val_score(forest_clf, train_set_x, train_set_y, cv=3, scoring="accuracy")
    print(score)
    params = [
        {'n_estimators': [5,10,30], 'max_features': [2,4,5], 'min_samples_split':[0.1,0.5,1.0]}
    ]
    #
    forest_clf = RandomForestClassifier(random_state=42)
    #
    grid_search = GridSearchCV(forest_clf, params, cv=2, scoring='accuracy')
    #
    grid_search.fit(train_set_x, train_set_y)
    #
    grid_search.best_params_

    features = grid_search.best_estimator_.feature_importances_
    attibs = list(trained_data)
    print("-----------FEATURES UTILIZED-------------")
    print(sorted(zip(features, attibs), reverse=True))


    final_model = grid_search.best_estimator_

    test_set_y = strat_test_set["classified"]
    test_set_X = strat_test_set.drop('classified', axis=1)

    final_predictions = final_model.predict(test_set_X)

    final_mse = mean_squared_error(test_set_y, final_predictions)
    final_rmse = np.sqrt(final_mse)

    print("FINAL RNSE:", final_rmse)

    save_model(final_model, './models/nut_bolt_test.sav')
