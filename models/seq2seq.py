import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class Encoder(tf.keras.layers.Layer):
    """ Seq2Seq 모델의 Encoder layer
    
    Translation에서는 source language를 입력으로 받아 특징을 추출
    
    시계열 예측에서는 t 이전 시점의 데이터를 입력으로 받고 특징을 추출  
    
    Attributes:
        units: (int) LSTM의 차원
    """
    def __init__(self, units, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.units = units
        self.lstm = tf.keras.layers.LSTM(units, 
                                         return_sequences=True, 
                                         return_state=True,)
    
    def call(self, inputs, training=False):
        outputs, *state = self.lstm(inputs)
        return outputs, state

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"class_name": "Encoder", "config": {"units": self.units}})
        return config


class Decoder(tf.keras.layers.Layer):
    """ Seq2Seq 모델의 Decoder layer
    
    Translation에서는 시작 토큰(e.g., <SOS>)과 인코더의 벡터를 받아 단어를 예측하고, 
    예측한 단어를 다시 입력으로 사용해 연속적으로 예측함. 
    
    시계열 예측에서는 t시점의 데이터를 시작 토큰으로 주고 이후 n-step을 예측함
    
    Attention을 사용할 경우 디코더의 state를 계산할 때 이전 state와 input만을 참고하는 것이 아니라 
    인코더의 모든 step에서의 출력을 추가로 참고하여 state를 계산함.
    
    다양한 종류의 Attention이 있으며 여기서는 dot-product attention을 사용
    
    Attributes:
        units: (int) LSTM의 차원
        step: (int) output sequences의 길이
        attention: (bool) Attenion layer를 사용할 지 여부
    """
    def __init__(self, units, dropout, attention=True, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        self.units = units
        self.attention = attention

        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, dropout=dropout)
        if attention:
            self.attn = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, hidden, enc_output, training=False):
        # inputs => (batch, label_width, 1)
        # hidden => (batch, enc_dim)
        # enc_output => (batch, input_width, enc_dim)
        
        outputs, *state = self.lstm(inputs, initial_state=hidden)
        context_vec = self.attn([outputs, enc_output])
        outputs = tf.concat([outputs, context_vec], axis=-1)
        outputs = self.dense(outputs)
        return outputs

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"class_name": "Decoder", "config": {"units": self.units}})
        return config
    

def get_model(units, input_width, label_width, feature_num,
              dropout, learning_rate=None, attention=True):
    """ Encoder-Decoder 구조의 Seq2Seq 모델
    
    Attributes:
        units: (int) 인코더와 디코더에 사용되는 GRU의 차원의 크기
        input_width: (int) 인코더에 입력될 인풋의 길이
        label_width: (int) 디코더에서 출력할 아웃풋의 길이
        attention: (bool) 어텐션 레이어를 사용할 지 여부
        
    Returns:
        model: (class) 인코더-디코더 구조의 keras 모델
        
    Examples:
        >>> model = get_model(64, 10, 5, 0.3, True)
        >>> prediction = model((tf.random.uniform((32, 10, 1)), tf.random.uniform((32, 5, 1))))
        >>> prediction.shape
        (32, 5, 1)
    """
    tf.keras.backend.clear_session()
    enc_input = tf.keras.Input(shape=(input_width, 1))
    encoder = Encoder(units)
    dec_input = tf.keras.Input(shape=(None, feature_num))
    decoder = Decoder(units, dropout, attention=attention)
    
    enc_outputs, enc_state = encoder(enc_input)
    dec_outputs = decoder(dec_input, enc_state, enc_outputs)
    model = tf.keras.Model((enc_input, dec_input), dec_outputs)
    
    if learning_rate == None:
        model.compile(loss='mse', optimizer="adam")
        #model.compile(loss='mse', optimizer=AngularGrad())
    else:
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        #model.compile(loss='mse', optimizer=AngularGrad(learning_rate=learning_rate))
    #end if
    
    return model
