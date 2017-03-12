import numpy as np
import pandas as pd
from sklearn import linear_model

logistic = linear_model.LogisticRegression(tol=0.1, max_iter=100, verbose=1)

trainX = np.load('tinyX.npy')  # this should have shape (26344, 3, 64, 64)
trainY = np.load('tinyY.npy')
testX = np.load('tinyX_test.npy')

print("Flattening...")
trainX_flattened = np.zeros((26344, 12288))
i = 0
for i in range(len(trainX)):
    a = trainX[i].flatten()
    trainX_flattened[i] = a
testX_flattened = np.zeros((len(testX), 12288))
j = 0
for i in range(len(testX)):
    b = testX[j].flatten()
    testX_flattened[j] = b

x_train, x_val, y_train, y_val = train_test_split(trainX_flattened, trainY, test_size=0.1)

print("Fitting data...")
logistic.fit(x_train, y_train)

print("Prediction training...")
pred_train = logistic.predict(x_train)
pred_train_df = pd.DataFrame({
    'prediction': pred_train,
    'category': y_train
})
j = 0
y = 0
result = []
for j in range(len(pred_train_df)):
    if pred_train_df['prediction'][j] == pred_train_df['category'][j]:
        y = 1
    else:
        y = 0
    result.append(y)

b = pd.Series(result)
pred_train_df['result'] = b.values
summation_train = pred_train_df['result'].sum(axis=0)
accuracy_train = float(summation_train) / float(len(pred_train_df))
print("Training accuracy: " + str(accuracy_train))

print("Prediction validation...")
pred_val = logistic.predict(x_val)
pred_val_df = pd.DataFrame({
    'prediction': pred_val,
    'category': y_val
})
i = 0
x = 0
result = []
for i in range(len(pred_val_df)):
    if pred_val_df['prediction'][i] == pred_val_df['category'][i]:
        x = 1
    else:
        x = 0
    result.append(x)

a = pd.Series(result)
pred_val_df['result'] = a.values
summation_val = pred_val_df['result'].sum(axis=0)
accuracy_val = float(summation_val) / float(len(pred_val_df))
print("Validation accuracy: " + str(accuracy_val))

print("Prediction testset...")
pred = logistic.predict(testX_flattened)
pred_df = pd.DataFrame(pred)

pred_train_df.to_csv("predictions_train.csv", index_label="id", header=["prediction", "category"])
pred_val_df.to_csv("predictions_val.csv", index_label="id", header=["prediction", "category"])
pred_df.to_csv("predictions_test.csv", index_label="id")

# to visualize only
# plt.imshow(trainX[18015].transpose(2, 1, 0))
# plt.show()
