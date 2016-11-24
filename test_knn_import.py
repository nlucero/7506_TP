from test_knn import *
train, test = main()
trainKNN = trainingKNN(train, dimTHT, dimMH, shingleSize, hashesGroups, hashesPerGroup, hashSTR, hashINT, hashVEC)
