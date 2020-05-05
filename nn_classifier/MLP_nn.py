from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
#import config from config_dic


class ClassifierMLP():

    def __init__(config):
        #Constructor get values from config dictionary
        self.input_dim=config['input_dim']
        self.learning_rate=config['learning_rate']
        self.epochs=config['epochs']
        self.batch_size=config['batch_size']
        self.dropout=config['dropout']
        #define here things direclty
        self.loss=tf.losses.CategoricalCrossentropy(from_logits=True)
        self.metric=['accuracy']


    def create_model(self):
        #Create NN Architecture, 3 fully connected layers with dropout between layers
        self.model=Sequential()
        self.model.add(Dense(units=512,activation='sigmoid',input_dim=self.input_dim))
        self.model.add(Dropout(rate=self.dropout))
        self.model.add(Dense(units=32,activation='relu'))
        self.model.add(Dropout(rate=self.dropout))
        self.model.add(Dense(units=1,activation='sigmoid'))

        #compile the model
        opt=keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        self.model.compile(optimizer=opt,loss=self.loss, metrics=self.metric)
        self.model.summary()
        return self.model

    def train_nn_model(self, x_train, y_train,epochs=self.epochs,batch_size=self.batch_size):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        return self.model

    def evaluate_nn_model(self,x_test, y_test,batch_size=self.batch_size):
        self.score=self.model.evaluate(x_test,y_test, batch_size=batch_size)
        return self.score

    def predict_nn_model(self, x_sample):
        self.prediction=self.model.predict(x_sample,batch_size=self.batch_size)
        return self.prediction
