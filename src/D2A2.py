import utils, copy
from keras.models import clone_model
from keras.optimizers import SGD

(_, _), (X_test, Y_test) = utils.load_dataset()

NUM_COMMUNICATION_ROUND = 10

file_o = open('D2A2.txt','w')
file_o.write('epoch,device,acc_local,acc_global\n')
model = utils.model_init()
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
res = model.evaluate(X_test, Y_test)
file_o.write('0,initial,%f,%f\n'%(0,res[1]))

for r in range(NUM_COMMUNICATION_ROUND):
    # root 
    modelR = clone_model(model)
    modelA1 = clone_model(modelR)
    modelA2 = clone_model(modelR)

    A1_models = []
    A1_models.append(modelA1.get_weights())
    for i in range(3):
        temp_model = clone_model(modelA1)
        LOSS = 'categorical_crossentropy' 
        lr = 0.000025
        OPTIMIZER = SGD(lr=lr, momentum=0.9, decay=lr, nesterov=False) # lr = 0.015, 67 ~ 69%
        temp_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
        x_train, y_train = utils.sampling_data(1000)
        temp_model.fit(x_train, y_train,  epochs=5, batch_size=10, verbose=1, validation_split=0.2)
        local_res = temp_model.evaluate(x_train, y_train)
        global_res = temp_model.evaluate(X_test, Y_test)
        temp_dev = 'device%003d'%i
        file_o.write('%d,%s,%f,%f\n'%(r,temp_dev,local_res[1],global_res[1]))
        A1_models.append(temp_model.get_weights())
    aggregated_A1_weights = utils.fedAVG(A1_models)


    A2_models = []
    A2_models.append(modelA2.get_weights())
    for i in range(3):
        temp_model = clone_model(modelA2)
        LOSS = 'categorical_crossentropy' 
        lr = 0.000025
        OPTIMIZER = SGD(lr=lr, momentum=0.9, decay=lr, nesterov=False) # lr = 0.015, 67 ~ 69%
        temp_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
        x_train, y_train = utils.sampling_data(1000)
        temp_model.fit(x_train, y_train,  epochs=5, batch_size=10, verbose=1, validation_split=0.2)
        local_res = temp_model.evaluate(x_train, y_train)
        global_res = temp_model.evaluate(X_test, Y_test)
        temp_dev = 'device%003d'%i
        file_o.write('%d,%s,%f,%f\n'%(r,temp_dev,local_res[1],global_res[1]))
        A2_models.append(temp_model.get_weights())
    aggregated_A2_weights = utils.fedAVG(A2_models)

    modelA1.set_weights(aggregated_A1_weights)
    global_res = modelA1.evaluate(X_test, Y_test)
    file_o.write('%d,%s,%f,%f\n'%(r,'aggregator01',0,global_res[1]))
    modelA2.set_weights(aggregated_A2_weights)
    global_res = modelA2.evaluate(X_test, Y_test)
    file_o.write('%d,%s,%f,%f\n'%(r,'aggregator02',0,global_res[1]))
    
    aggregated_A1A2_weights = utils.fedAVG([modelA1.get_weights(),modelA2.get_weights()])
    model.set_weights(aggregated_A1_weights)
    global_res = model.evaluate(X_test, Y_test)
    file_o.write('%d,%s,%f,%f\n'%(r,'root',0,global_res[1]))