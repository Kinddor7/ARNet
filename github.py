import pickle
import datetime

inputs = Input(shape=(224, 224, 3))


x = Conv2D(96, (11, 11), strides=(2, 2), padding='valid')(inputs)
x = BatchNormalization()(x) 
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2),padding='valid')(x)

x = Conv2D(256, (5, 5), padding='same')(x)
x = BatchNormalization()(x) 
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2),padding='valid')(x)

residual2 = x


x = Conv2D(384, (3, 3), padding='same')(x)
x = BatchNormalization()(x) 
x = Activation('relu')(x)
residual = x
x = Conv2D(384, (3, 3), padding='same')(x)
x = BatchNormalization()(x)  
x = Activation('relu')(x)
x = add([x,residual])
x = Conv2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)  
x = Activation('relu')(x)
x = add([x,residual2])
x = MaxPooling2D((3, 3), strides=(2, 2),padding='valid')(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x) 
x = Activation('relu')(x)


outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer= tf.keras.optimizers.Adam(0.0001), metrics=['acc',tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

from keras.callbacks import EarlyStopping,ReduceLROnPlateau


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)

import time
curr_time = round(time.time()*1000)
print("Milliseconds since epoch start:",curr_time)



# start training
history = model.fit(train_generator,
                    epochs=32,
                    steps_per_epoch=len(train_generator),
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    callbacks=[reduce_lr],
                    verbose=1)
curr_time = round(time.time()*1000)
print("Milliseconds since epoch final:",curr_time)
