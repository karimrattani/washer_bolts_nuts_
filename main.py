#https://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/
#http://www.mathworks.com/matlabcentral/fileexchange/25157-image-segmentation-tutorial
#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
#https://stackoverflow.com/questions/35472712/how-to-split-data-on-balanced-training-set-and-test-set-on-sklearn
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
#https://docs.opencv.org/3.3.1/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a


from segmentation import segmentation
from features import get_features
from build_model import load_model





#segments the image
img='./Parts2.jpg'
segm,image,out=segmentation(img)

#load model
model = load_model('./models/nut_bolt_test.sav')
nut=0
bolt=0
washer=0
#extract the feature
for segment in segm:
    x,y,w,h=segment
    seg_img = out[y:y + h, x:x + w]
    features = get_features(seg_img)
    print(features)
    pred = model.predict([features])
    if pred == 1:
        nut+=1
    elif pred==2:
        bolt+=1
    else:
        washer+=1
print("Nuts:"+str(nut)+" Bolts:"+str(bolt)+" Washer:"+str(washer))


#
# #-----------------------Begin Classification-----------------------#
# imgId=0
# #write features to file
# with open('dataset/data.csv','w') as f:
#     writer=csv.writer(f)
#     writer.writerow(['id', 'width', 'height', 'area_ratio', 'box_shape', 'total_corners', 'circularity','file','corner/area' 'classified'])
#     for segmentation in segm:
#         x, y, w, h = segmentation
#         seg_img = out[y:y+h, x:x+w]
#         features = get_features(seg_img)
#         filename = './dataset/img/'+str(imgId)+'.png'
#         cv2.imwrite(filename,img[y:y+h, x:x+w])#write original image with segmented image x and y
#         writer.writerow([imgId,seg_img.shape[1],seg_img.shape[0],features[0],features[1],features[2],features[3],features[4],filename,'NONE'])
#         imgId+=1
#
# trained_data = pd.read_csv('./dataset/trained_data.csv')
# #print(trained_data.describe())
#
# #transform data from word to numeric
# trained_data['classified'] = trained_data['classified'].str.replace('NUTS', '1')
# trained_data['classified'] = trained_data['classified'].str.replace('BOLTS', '2')
# trained_data['classified'] = trained_data['classified'].str.replace('WASHER', '3')
#
# #dropped the column no longer needed, DO NOT WRITE
# for col in ['id','file','width','height']:
#      trained_data = trained_data.drop(col, axis=1)
# trained_data['classified'] = trained_data['classified'].apply(pd.to_numeric)
# print(trained_data)
#
# #training and validation sets
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
# for train_index, test_index in split.split(trained_data, trained_data['classified']):
#     strat_train_set = trained_data.loc[train_index]
#     strat_test_set = trained_data.loc[test_index]
# train_set_y = strat_train_set["classified"]
# train_set_x = strat_train_set.drop('classified', axis=1)
# # print("-----------SET-------------")
# # print(train_set_y.head())
# # print(train_set_x.head())
#
# #train model
# forest_clf = RandomForestClassifier(random_state=42)
# score = cross_val_score(forest_clf, train_set_x, train_set_y, cv=2, scoring="accuracy")
# #print(score)
#
# params = [
#     {'n_estimators': [5,10,30], 'max_features': [2,4,5], 'min_samples_split':[0.1,0.5,1.0]}
# ]
# #
# forest_clf = RandomForestClassifier(random_state=42)
# #
# grid_search = GridSearchCV(forest_clf, params, cv=2, scoring='accuracy')
# #
# grid_search.fit(train_set_x, train_set_y)
# #
# grid_search.best_params_
#
# features = grid_search.best_estimator_.feature_importances_
# attibs = list(trained_data)
# print("-----------FEATURES UTILIZED-------------")
# print(sorted(zip(features, attibs), reverse=True))
#
#
# final_model = grid_search.best_estimator_
#
# test_set_y = strat_test_set["classified"]
# test_set_X = strat_test_set.drop('classified', axis=1)
#
# final_predictions = final_model.predict(test_set_X)
#
# final_mse = mean_squared_error(test_set_y, final_predictions)
# final_rmse = np.sqrt(final_mse)
#
# print("FINAL RNSE:", final_rmse)
# save_model(final_model, './models/nut_bolt_test.sav')
#
#
# #-------------END UTILIZED--------------
# #-----------------PREDICT RESULTS--------------
#
# model = load_model('./models/nut_bolt_test.sav')
# nut=0
# bolt=0
# washer=0
# for segmentation in segm:
#     x, y, w, h = segmentation
#     seg_img = out[y:y + h, x:x + w]
#     features = get_features(seg_img)
#    # print(features)
#     cls = model.predict([features])
#     #print(cls)
#     if cls == 1:
#         nut+=1
#     elif cls==2:
#         bolt+=1
#     else:
#         washer+=1
# print("Nuts:"+str(nut)+" Bolts:"+str(bolt)+" Washer:"+str(washer))