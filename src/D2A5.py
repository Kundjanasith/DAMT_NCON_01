import utils, copy
from keras.models import clone_model
from keras.optimizers import SGD
import sys
counter = sys.argv[1]

(_, _), (X_test, Y_test) = utils.load_dataset()

NUM_COMMUNICATION_ROUND = 100

file_o = open('%s_D2A5.txt'%counter,'w')
file_o.write('epoch,device,acc_local,acc_global\n')
model = utils.model_init()
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
res = model.evaluate(X_test, Y_test)
file_o.write('0,initial,%f,%f\n'%(0,res[1]))

for r in range(NUM_COMMUNICATION_ROUND):
    # root 
    # modelR = clone_model(model)
    # modelA1 = clone_model(modelR)
    # modelA2 = clone_model(modelR)
    modelR = model 
    modelA1 = modelR
    modelA2 = modelR
    modelA3 = modelR
    modelA4 = modelR
    modelA5 = modelR

    A1_models = []
    # A1_models.append(modelA1.get_weights())
    for i in range(12):
        # temp_model = clone_model(modelA1)
        temp_model = modelA1
        LOSS = 'categorical_crossentropy' 
        lr = 0.000001
        OPTIMIZER = SGD(lr=lr, momentum=0.9, nesterov=False) # lr = 0.015, 67 ~ 69%
        temp_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
        x_train, y_train = utils.sampling_data(1000)
        temp_model.fit(x_train, y_train,  epochs=5, batch_size=10, verbose=0, validation_split=0.2)
        local_res = temp_model.evaluate(x_train, y_train)
        global_res = temp_model.evaluate(X_test, Y_test)
        temp_dev = 'device%003d'%i
        file_o.write('%d,%s,%f,%f\n'%(r,temp_dev,local_res[1],global_res[1]))
        temp_weights = copy.deepcopy(temp_model.get_weights())
        A1_models.append(temp_weights)
    modelA1 = utils.fedAVG(A1_models)


    A2_models = []
    # A2_models.append(modelA2.get_weights())
    for i in range(12):
        # temp_model = clone_model(modelA2)
        temp_model = modelA2
        LOSS = 'categorical_crossentropy' 
        lr = 0.000001
        OPTIMIZER = SGD(lr=lr, momentum=0.9, nesterov=False) # lr = 0.015, 67 ~ 69%
        temp_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
        x_train, y_train = utils.sampling_data(1000)
        temp_model.fit(x_train, y_train,  epochs=5, batch_size=10, verbose=0, validation_split=0.2)
        local_res = temp_model.evaluate(x_train, y_train)
        global_res = temp_model.evaluate(X_test, Y_test)
        temp_dev = 'device%003d'%i
        file_o.write('%d,%s,%f,%f\n'%(r,temp_dev,local_res[1],global_res[1]))
        temp_weights = copy.deepcopy(temp_model.get_weights())
        A2_models.append(temp_weights)
    modelA2 = utils.fedAVG(A2_models)

    A3_models = []
    # A3_models.append(modelA3.get_weights())
    for i in range(12):
        # temp_model = clone_model(modelA3)
        temp_model = modelA3
        LOSS = 'categorical_crossentropy' 
        lr = 0.000001
        OPTIMIZER = SGD(lr=lr, momentum=0.9, nesterov=False) # lr = 0.015, 67 ~ 69%
        temp_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
        x_train, y_train = utils.sampling_data(1000)
        temp_model.fit(x_train, y_train,  epochs=5, batch_size=10, verbose=0, validation_split=0.2)
        local_res = temp_model.evaluate(x_train, y_train)
        global_res = temp_model.evaluate(X_test, Y_test)
        temp_dev = 'device%003d'%i
        file_o.write('%d,%s,%f,%f\n'%(r,temp_dev,local_res[1],global_res[1]))
        temp_weights = copy.deepcopy(temp_model.get_weights())
        A3_models.append(temp_weights)
    modelA3 = utils.fedAVG(A3_models)
    

    A4_models = []
    # A4_models.append(modelA4.get_weights())
    for i in range(12):
        # temp_model = clone_model(modelA4)
        temp_model = modelA4
        LOSS = 'categorical_crossentropy' 
        lr = 0.000001
        OPTIMIZER = SGD(lr=lr, momentum=0.9, nesterov=False) # lr = 0.015, 67 ~ 69%
        temp_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
        x_train, y_train = utils.sampling_data(1000)
        temp_model.fit(x_train, y_train,  epochs=5, batch_size=10, verbose=0, validation_split=0.2)
        local_res = temp_model.evaluate(x_train, y_train)
        global_res = temp_model.evaluate(X_test, Y_test)
        temp_dev = 'device%003d'%i
        file_o.write('%d,%s,%f,%f\n'%(r,temp_dev,local_res[1],global_res[1]))
        temp_weights = copy.deepcopy(temp_model.get_weights())
        A4_models.append(temp_weights)
    modelA4 = utils.fedAVG(A4_models)

    A5_models = []
    # A5_models.append(modelA5.get_weights())
    for i in range(12):
        # temp_model = clone_model(modelA5)
        temp_model = modelA5
        LOSS = 'categorical_crossentropy' 
        lr = 0.000001
        OPTIMIZER = SGD(lr=lr, momentum=0.9, nesterov=False) # lr = 0.015, 67 ~ 69%
        temp_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
        x_train, y_train = utils.sampling_data(1000)
        temp_model.fit(x_train, y_train,  epochs=5, batch_size=10, verbose=0, validation_split=0.2)
        local_res = temp_model.evaluate(x_train, y_train)
        global_res = temp_model.evaluate(X_test, Y_test)
        temp_dev = 'device%003d'%i
        file_o.write('%d,%s,%f,%f\n'%(r,temp_dev,local_res[1],global_res[1]))
        temp_weights = copy.deepcopy(temp_model.get_weights())
        A5_models.append(temp_weights)
    modelA5 = utils.fedAVG(A5_models)

    print(r,'AGGREGATOR01')
    modelA1.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    global_res = modelA1.evaluate(X_test, Y_test)
    file_o.write('%d,%s,%f,%f\n'%(r,'aggregator01',0,global_res[1]))
    
    print(r,'AGGREGATOR02')
    modelA2.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    global_res = modelA2.evaluate(X_test, Y_test)
    file_o.write('%d,%s,%f,%f\n'%(r,'aggregator02',0,global_res[1]))

    print(r,'AGGREGATOR03')
    modelA3.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    global_res = modelA3.evaluate(X_test, Y_test)
    file_o.write('%d,%s,%f,%f\n'%(r,'aggregator03',0,global_res[1]))

    print(r,'AGGREGATOR04')
    modelA4.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    global_res = modelA4.evaluate(X_test, Y_test)
    file_o.write('%d,%s,%f,%f\n'%(r,'aggregator04',0,global_res[1]))

    print(r,'AGGREGATOR05')
    modelA5.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    global_res = modelA5.evaluate(X_test, Y_test)
    file_o.write('%d,%s,%f,%f\n'%(r,'aggregator05',0,global_res[1]))
    
    model = utils.fedAVG([modelA1.get_weights(),modelA2.get_weights(),modelA3.get_weights(),modelA4.get_weights(),modelA5.get_weights()])

    print(r,'ROOT')
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    global_res = model.evaluate(X_test, Y_test)
    file_o.write('%d,%s,%f,%f\n'%(r,'root',0,global_res[1]))
