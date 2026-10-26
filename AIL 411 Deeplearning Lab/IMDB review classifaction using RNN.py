from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_size=10000
max_length=200
(xtrain,ytrain),(xtest,ytest)=imdb.load_data(num_words=vocab_size)
xtrain=pad_sequences(xtrain,maxlen=max_length)
xtest=pad_sequences(xtest,maxlen=max_length)
model=Sequential([Embedding(vocab_size,32,input_length=vocab_size),
                 SimpleRNN(32),
                 Dense(1,activation='sigmoid')])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=5,batch_size=64)
loss,accuracy=model.evaluate(xtrain,ytrain,batch_size=32)
print("loss ",loss ,"\n Accuracy ",accuracy)
pred=model.predict(xtest[:1])
print("postive" if pred>=0.5 else "negative",pred)
