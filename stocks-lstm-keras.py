# -*- coding: utf-8 -*-
import numpy
import pandas
import matplotlib.pyplot as plt
 
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
 
class Predict:
  def __init__(self):
    self.length_of_sequences = 5
    self.in_out_neurons = 1
    self.hidden_neurons = 300
    self.batch_size = 32
    self.epochs = 100
    self.percentage = 0.8
 
  # prepare data
  def load_data(self, data, n_prev):
    x, y = [], []
    for i in range(len(data) - n_prev):
      x.append(data.iloc[i:(i+n_prev)].values)
      y.append(data.iloc[i+n_prev].values)
    X = numpy.array(x)
    Y = numpy.array(y)
    return X, Y
 
  # make model
  def create_model(self) :
    Model = Sequential()
    Model.add(LSTM(self.hidden_neurons, batch_input_shape=(None, self.length_of_sequences, self.in_out_neurons), return_sequences=False))
    Model.add(Dense(self.in_out_neurons))
    Model.add(Activation("linear"))
    Model.compile(loss="mape", optimizer="adam")
    return Model
 
  # do learning
  def train(self, x_train, y_train) :
    Model = self.create_model()
    Model.fit(x_train, y_train, self.batch_size, self.epochs)
    return Model
 
if __name__ == "__main__":
 
  predict = Predict()
  nstocks = 2;
 
  # do learning, prediction, show to each stock
  for istock in range(1, nstocks + 1):
    
    # prepare data 
    data = None
    data = pandas.read_csv('/content/drive/My Drive/LSTM/csv/' + str(istock) + '_stock_price.csv')
    data.columns = ['date', 'open', 'high', 'low', 'close']
    data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
 
    # standalize the close data
    data['close'] = preprocessing.scale(data['close'])
    data = data.sort_values(by='date')
    data = data.reset_index(drop=True)
    data = data.loc[:, ['date', 'close']]
 
    # split data for learning and test by partition
    split_pos = int(len(data) * predict.percentage)
    x_train, y_train = predict.load_data(data[['close']].iloc[0:split_pos], predict.length_of_sequences)
    x_test, y_test = predict.load_data(data[['close']].iloc[split_pos:], predict.length_of_sequences)
 
    # do learning
    model = predict.train(x_train, y_train)
 
    # do test
    predicted = model.predict(x_test)
    result = pandas.DataFrame(predicted)
    result.columns = [str(istock) + '_predict']
    result[str(istock) + '_actual'] = y_test
 
    # show
    result.plot()
    plt.show()
 
    # compare prices
    current = result.iloc[-1][str(istock) + '_actual']
    predictable = result.iloc[-1][str(istock) + '_predict']
    if (predictable - current) > 0:
      print(f'{istock} stock price of the next day INcreases: {predictable-current:.2f}, predictable:{predictable:.2f}, current:{current:.2f}')
    else:
      print(f'{istock} stock price of the next day DEcreases: {current-predictable:.2f}, predictable:{predictable:.2f}, current:{current:.2f}')
