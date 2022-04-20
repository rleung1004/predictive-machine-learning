def example_1():
    from random import randint
    from numpy import array
    from numpy import argmax
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense

    NUM_FEATURES        = 10
    NUM_SAMPLES         = 5
    TARGET_INDEX        = 2
    NUM_WEIGHT_UPDATES  = 1000


    # generate array of 5 numbers like [5, 8, 3, 0, 9].
    # each number is >=0 and <10
    def generate_sequence():
        return [randint(0, NUM_FEATURES-1) for _ in range(NUM_SAMPLES)]


    # one hot encode sequence
    def oneHotEncode(sequence):
        encoding = list()
        # Convert [5, 8, 3, 0, 9]
        # to
        # [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        #  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        #  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        #  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        for value in sequence:
            # Create vector of zeros.
            vector = [0 for _ in range(NUM_FEATURES)]
            vector[value] = 1 # Add 1 to vector.
            encoding.append(vector)
        return array(encoding)


    # decode a one hot encoded string
    def oneHotDecode(encoded_seq):
        # gets index of element with the maximum value.
        return [argmax(vector) for vector in encoded_seq]


    # generate one example for an lstm
    def generateSample(targetIndex):
        # generate sequence such as [5, 8, 3, 0, 9]
        sequence = generate_sequence()
        # one hot encode sequence so [5, 8, 3, 0, 9] becomes
        # [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        #  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        #  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        #  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        encoded = oneHotEncode(sequence)
        # reshape sequence to be 3D
        X = encoded.reshape((1, NUM_SAMPLES, NUM_FEATURES))
        # y becomes second element.
        # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        y = encoded[targetIndex].reshape(1, NUM_FEATURES)
        return X, y

    # define model
    model = Sequential()
    model.add(LSTM(25, input_shape=(NUM_SAMPLES, NUM_FEATURES)))
    model.add(Dense(NUM_FEATURES, activation='softmax'))

    # Our output is a one-hot encoded vector so use categorical
    # crossentropy.
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    for i in range(NUM_WEIGHT_UPDATES):
        trainX, trainy = generateSample(TARGET_INDEX)
        # Update model - weights are updated and are not reset.
        model.fit(trainX, trainy, epochs=1, verbose=2)

    # evaluate model
    correct = 0
    NUM_EVALUATIONS = 100
    for i in range(NUM_EVALUATIONS):
        X, y = generateSample(TARGET_INDEX)

        yhat = model.predict(X)
        if oneHotDecode(yhat) == oneHotDecode(y):
            correct += 1
    print('Accuracy: %f' % ((correct / NUM_EVALUATIONS)))


def example_2():
    from math import sin
    from math import pi
    from math import exp
    from random import randint
    from random import uniform
    from numpy import array
    from matplotlib import pyplot
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense

    # generate damped sine wave in [0,1]
    def generate_sequence(length, period, decay):
        return [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) \
                for i in range(length)]

    # generate input and output pairs of damped sine waves
    def generate_examples(input_len, n_patterns, output_len):
        X, y = list(), list()
        for _ in range(n_patterns):
            p = randint(10, 20)
            d = uniform(0.01, 0.1)
            sequence = generate_sequence(input_len + output_len, p, d)

            X.append(sequence[:-output_len])
            y.append(sequence[-output_len:])  # Assigns next 5 values in sequence.
        X = array(X).reshape(n_patterns, input_len, 1)
        y = array(y).reshape(n_patterns, output_len)
        return X, y

    # configure problem
    INPUT_LEN = 50
    OUTPUT_LEN = 5
    # define model
    model = Sequential()
    model.add(LSTM(20, input_shape=(INPUT_LEN, 1)))
    model.add(Dense(OUTPUT_LEN))
    model.compile(loss='mae', optimizer='adam')
    model.summary()

    # fit model
    X, y = generate_examples(INPUT_LEN, 10000, OUTPUT_LEN)
    history = model.fit(X, y, batch_size=10, epochs=1)

    # evaluate model
    X, y = generate_examples(INPUT_LEN, 1000, OUTPUT_LEN)
    loss = model.evaluate(X, y, verbose=0)
    print('Mean squared error: %f' % loss)

    print("\n*** Make predictions")
    for i in range(0, 5):
        # prediction on new data
        X, y = generate_examples(INPUT_LEN, 1, OUTPUT_LEN)
        yhat = model.predict(X, verbose=0)

        pyplot.title("Y and Yhat")
        pyplot.plot(y[0], label='y')
        pyplot.plot(yhat[0], label='yhat')
        pyplot.legend()
        pyplot.show()


example_2()
