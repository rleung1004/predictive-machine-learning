import pandas                as     pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from tensorflow.keras        import Sequential
from tensorflow.keras.layers import Dense


PATH       = "../datasets/"
df         = pd.read_csv(PATH + 'iris_old.csv')
df.columns = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W', 'Iris Type']
print(df)

# Convert text to numeric category.
# 0 is setosa, 1 is versacolor and 2 is virginica
df['y'] = LabelEncoder().fit_transform(df['Iris Type'])

# Prepare the data.
dfX = df.iloc[:, 0:4] # Get X features only from columns 0 to 3
dfY = df.iloc[:, 5:6] # Get X features only from column 5

x_array = dfX.values
x_arrayReshaped = x_array.reshape(x_array.shape[0], x_array.shape[1])

y_array = dfY.values
y_arrayReshaped = y_array.reshape(y_array.shape[0], y_array.shape[1])

# Split into train and test data sets.
X_train, X_test, y_train, y_test = train_test_split(
    x_arrayReshaped, y_arrayReshaped, test_size=0.33)

model = Sequential()

# Sigmoid first hidden layer with 12 neurons
model.add(Dense(12, activation='sigmoid', input_shape=(x_arrayReshaped.shape[1],)))
# Softmax output layer with 3 neurons
# Softmax activation is used to allow multi-class output
model.add(Dense(3, activation='softmax'))
# Compile the model
# Sparse categorical crossentropy function is used to calculate the non-binary classification predictions
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
# An epoch is one iteration for all samples through the network.
# verbose can be set to 1 to show detailed output during training.
model.fit(x_arrayReshaped, y_arrayReshaped, epochs=1000, verbose=1)
# Evaluate the model
loss, acc = model.evaluate(x_arrayReshaped, y_arrayReshaped, verbose=0)
print('Test Accuracy: %.3f' % acc)


# Make a prediction
row = [5.1, 3.5, 1.4, 0.2]
yhat = model.predict([row])
print(yhat)
print('Predicted: %.3f' % yhat)


