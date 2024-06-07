import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class MultiStepLstmLayer(tf.keras.layers.Layer):
    def __init__(self, units, step, dropout):
        super(MultiStepLstmLayer, self).__init__()
        self.units = units
        self.step = step
        self.lstm = tf.keras.layers.LSTM(units, dropout=dropout)
        self.dense = tf.keras.layers.Dense(step, activation='linear')
  
    def call(self, inputs, training=None):
        outputs = self.lstm(inputs)
        outputs = self.dense(outputs)
        outputs = tf.expand_dims(outputs, -1)
        return outputs


def get_model(units, input_width, label_width, dropout, learning_rate=None):
    tf.keras.backend.clear_session()
    
    inputs = tf.keras.Input(shape=(input_width, 1))
    multi_lstm = MultiStepLstmLayer(units, label_width, dropout)
    outputs = multi_lstm(inputs)
    model = tf.keras.Model(inputs, outputs)

    if learning_rate == None:
        model.compile(loss='mse', optimizer="adam")
        #model.compile(loss='mse', optimizer=AngularGrad())
    else:
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        #model.compile(loss='mse', optimizer=AngularGrad(learning_rate=learning_rate))
    #end if
    
    return model
       
        