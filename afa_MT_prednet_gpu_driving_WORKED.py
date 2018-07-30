# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch

import numpy
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from DiscriminativeCell import DiscriminativeCell
from GenerativeCell import GenerativeCell

from attention import Attention
import time
 

from read_img import read_pgm

# Define some constants
OUT_LAYER_SIZE = (1,) + tuple(2 ** p for p in range(4, 10))
ERR_LAYER_SIZE = tuple(size * 2 for size in OUT_LAYER_SIZE)
IN_LAYER_SIZE = (1,) + ERR_LAYER_SIZE

DIRECTION  = 2 

LAYERS = 4

SAVE_IMG = True
VERBOSE = 0

tau = [1.0,1.0,1.2, 1.5]


class PrednetModel(nn.Module):
    """
    Build the Prednet model
    """

    def __init__(self, error_size_list, num_of_layers):
        super(PrednetModel,self).__init__()
        # print "error_size_list", error_size_list
        self.number_of_layers = num_of_layers
        # print "self.number_of_layers", self.number_of_layers
        for layer in range(0, self.number_of_layers):
            
            setattr(self, 'discriminator_' + str(layer + 1), DiscriminativeCell(
                        input_size={'input': IN_LAYER_SIZE[layer], 'state': OUT_LAYER_SIZE[layer]},
                        hidden_size=OUT_LAYER_SIZE[layer],
                        first=(not layer)
                        ))
            
            for d in range(0, DIRECTION):
                setattr(self, 'generator_' + str(layer + 1) + "_" + str(d), GenerativeCell(
                        input_size={'error': ERR_LAYER_SIZE[layer], 'up_state':
                        OUT_LAYER_SIZE[layer + 1] if layer != self.number_of_layers - 1 else 0},
                        hidden_size=OUT_LAYER_SIZE[layer],
                        error_init_size=error_size_list[layer]
                        ))
                    
        
    def forward(self, bottom_up_input, error, state, action_in):

        # generative branch
        up_state = [None] * self.number_of_layers
        
        #self.action = [Attention(10) for count in range(self.number_of_layers-1)]
        self.action = Attention(10) 
        
        for layer in reversed(range(0, self.number_of_layers)):
            
             
            if not layer < self.number_of_layers - 1 :
                
                
                
                for d in range(0, DIRECTION):
                        
                    if not state[d][layer] == None:
                   
                        current_state = getattr(self, 'generator_' + str(layer + 1) + "_" + str(d))(
                                error[layer], None, state[d][layer]
                                )
                    
                        current_state_list = list(current_state)
                    
                        state[d][layer] = (torch.mul(state[d][layer][0],(1-1/tau[layer]))  +  torch.mul(current_state_list[0], 1/tau[layer]), torch.mul(state[d][layer][1],(1-1/tau[layer]))  +  torch.mul(current_state_list[1], 1/tau[layer]))
                
                    else:
                        state[d][layer] = getattr(self, 'generator_' + str(layer + 1) + "_" + str(d))(
                                error[layer], None, state[d][layer]
                                )
                    
                    
                    
                
            else:
                
                
                  
                for d in range(0, DIRECTION):
                        
                    if not state[d][layer] == None:
                        current_state = getattr(self, 'generator_' + str(layer + 1) + "_" + str(d))(
                                error[layer], up_state[layer+1], state[d][layer]
                                )
                    
                        current_state_list = list(current_state)
                    
                        state[d][layer] = (torch.mul(state[d][layer][0],(1-1/tau[layer]))  +  torch.mul(current_state_list[0], 1/tau[layer]), torch.mul(state[d][layer][1],(1-1/tau[layer]))  +  torch.mul(current_state_list[1], 1/tau[layer]))
                        
                    else:
                    
                        state[d][layer] = getattr(self, 'generator_' + str(layer + 1) + "_" + str(d))(
                                error[layer], up_state[layer+1], state[d][layer]
                                )
            
            #up_state[layer] = self.action[layer-1]([i[layer][0] for i in state], action_in) 
            up_state[layer] = self.action([i[layer][0] for i in state], action_in) 

        # discriminative branch
        for layer in range(0, self.number_of_layers):
            if layer == 0:
                error[layer] = getattr(self, 'discriminator_' + str(layer + 1))(
                bottom_up_input,
                up_state[layer]
                #state[layer][0]
            )
            else:
                error[layer] = getattr(self, 'discriminator_' + str(layer + 1))(
                error[layer - 1],
                up_state[layer]
            )
        #print up_state[0].size()
        return error, state, up_state[0]





def _test_training(inData, outData, action, Length, TotalLength, count, load_model_name=None):
    number_of_layers = LAYERS
     
    # T =  inData.size()[0]
    T =  TotalLength
    # print inData
    #print outData
    # T = 6  # sequence length
    max_epoch = 600 # number of epochs
    max_iter = 50
    lr = 1e-3     # learning rate
    # lr_schedule = lambda epoch: 0.0003 if epoch < 30000 else 0.0001 
    momentum = 0.8
    # set manual seed
    torch.manual_seed(0)

    L = number_of_layers - 1
    print('\n---------- Train a', str(L + 1), 'layer network ----------')
    print('Create the input image and target sequences')
    # inData = inData.view(T, 1, 8, 8)
    
    
    
    # input_sequence = Variable(torch.rand(T, 1, 3, 4 * 2 ** L, 6 * 2 ** L))
    
   
    
    # error_init_size_list = input_sequence.data.size()
    error_init_size_list = tuple(
     #   (1, ERR_LAYER_SIZE[l], 15 * 2 ** (L - l),  20 * 2 ** (L - l)) for l in range(0, L + 1)
    (1, ERR_LAYER_SIZE[l], int(16 * 2 ** (L - l - 1)),  int(20 * 2 ** (L - l - 1))) for l in range(0, L + 1)
    )
    
    print('The error initialisation sizes are', error_init_size_list)
    
    #target_sequence = Variable(outData)
    
    
    #TODO: more elegent way
    if load_model_name == None:

        print('Define a', str(L + 1), 'layer Prednet')
        model = PrednetModel(error_init_size_list, number_of_layers)
        
    else:
        model = load_model(load_model_name)
        print("model has been loaded!")

    
    
    
    
    if torch.cuda.is_available():
        if VERBOSE > 1:
            print("Using GPU")
        model.cuda()
        
    optimizer =  torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = LambdaLR(optimizer, lr_lambda = lr_schedule)
    
   # optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    #model = model.cuda()

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()
    
    
    #print numpy.shape(sequence_pred)

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        
        # scheduler.step()
        # print L
        
        t1 = time.time()
        for seq in range(count):
            
            
            Seq_length = Length[seq]
            
            
            
            for iteration in xrange(max_iter):
                
                
                
                if torch.cuda.is_available():
                
                    input_sequence = Variable(inData[seq]).cuda()
                    action_in = Variable(action[seq]).cuda()
                    target_sequence =  Variable(torch.zeros(Seq_length, error_init_size_list[0][0], error_init_size_list[0][1], error_init_size_list[0][2], error_init_size_list[0][3])).cuda()
                else:
    
                    input_sequence = Variable(inData[seq])
                    action_in = Variable(action[seq])
                    target_sequence =  Variable(torch.zeros(Seq_length, error_init_size_list[0][0], error_init_size_list[0][1], error_init_size_list[0][2], error_init_size_list[0][3]))
            
                if VERBOSE  >= 2: # we need to save something
            
                    print('Input has size', list(input_sequence.data.size()))
                    print("we are learning seq", seq)
                if SAVE_IMG == True:
                    sequence_pred = torch.zeros((Seq_length, 1, 1, 16 * 2 ** (L-1), 20 * 2 ** (L-1)))
                
                
            
                state = [[None] * (L+1)] * DIRECTION
                error = [None] * (L + 1)
                loss = 0
                for t in range(0, Seq_length):
                    error, state, prediction = model(input_sequence[t], error, state, action_in[t,:,:])
                    loss += loss_fn(error[0], target_sequence[t])
                    sequence_pred[t,:,:,:,:] = (prediction.data)
                
                    if  epoch == max_epoch - 1 or epoch % 10 == 0 and SAVE_IMG == True:
                        torch.save(prediction, "savetxt/mid_prediction"+ str(epoch) + "_" + str(seq) + "_"+ str(t) + ".txt")
                        torch.save(state, "savetxt/mid_state"+ str(epoch) + "_" + str(seq) + "_"+ str(t) + ".txt")

                print(' > Epoch {:2d}, Seq: {}, Iteration {:2d}, loss: {}'.format((epoch + 1), (seq + 1), (iteration + 1), loss.data[0]))
        
            
                    

                # zero grad parameters
                optimizer.zero_grad()
    
                # compute new grad parameters through time!
                loss.backward()
                optimizer.step()
                # learning_rate step against the gradient
                #for p in model.parameters():
                
                 #   p.data.sub_(p.grad.data * lr)
                
        t2 = time.time()
        print("Epoch time: ~%f milliseconds" % ((t2 - t1) * 1000.))
                
                
        if epoch % 100 == 0 or epoch == max_epoch - 1  and SAVE_IMG == True:
            
                    for i in range(0, Seq_length):
               
                        savematrix(sequence_pred[i,0,0,:,:], "savetxt/prediction_"  + str(seq) + "_" + str(epoch) + "_" + str(i) +".txt")
            
            #for l in (model.mid_layer):
            #    for p in l.parameters():
            #        p.data.sub_(p.grad.data * (lr*5.0))
                    
            #for l in (model.low_layer):
            #    for p in l.parameters():
            #        p.data.sub_(p.grad.data * (lr*5.0))
            
            
                
                
        
        if epoch % 50 == 0:
            torch.save(model, str(epoch)+'model.pt')
            
def loadImage(size1, size2):
    
    import glob 
    import numpy as np
    
    length = len(glob.glob("images/*.txt"))
    
    InputData = torch.zeros((length, 1, 1, size1, size2)) 
    #InputData_gpu = InputData.cuda()
    
    i = 0
    
    for infile in sorted(glob.glob("images/*.txt")):
        print("Current File Being Processed is: " + infile)
        
        InputData[i, 0, 0, :, :] = torch.Tensor(normalisation(np.loadtxt(infile)))
        #InputData[i, 0, 1, :, :] =  (InputData[i, 0, 0, :, :])
        #InputData[i, 0, 2, :, :] =  (InputData[i, 0, 0, :, :]) 
        i += 1
    
    for i in range(0, length):
            
        if SAVE_IMG == True:    
            
            savematrix(InputData[i,0,0,:,:], "input_"+str(i)+".txt")    
    
        
    print("load image files completed!")
        
    action = torch.zeros((length, 1,  2))
    
    action_np = np.loadtxt("2-sorted.csv", delimiter=",")
    
    step = 4.77
    
    for i in range(length):
        
        
        if round(step*i) <= np.shape(action_np)[0]:
             
            action[i, 0, :] = torch.Tensor(action_np[int(round(step*i)),:])
            
        else:
            
            break
        
     
    #action_gpu = action.cuda()
    OutputData = np.copy(InputData)
        
    return InputData, OutputData, action  
 
def normalisation_action_list(action):
    
    # wheel = numpy.array([]).reshape(0,5)
    gmax_wheel = -1000
    gmin_wheel = 1000
    
    gmax_steer = -3.14
    gmin_steer = 3.14
    
    
    for i in range(len(action)):
        part_action = action[i].numpy()
         
        wheel = part_action[:,:,0:4]
        steer = part_action[:,:,4]
        
        cmax_wheel, cmin_wheel = numpy.max(wheel), numpy.min(wheel)
        
        if cmax_wheel > gmax_wheel:
            gmax_wheel=cmax_wheel
        if cmin_wheel < gmin_wheel:
            gmin_wheel=cmin_wheel
            
        
        cmax_steer, cmin_steer = numpy.max(steer), numpy.min(steer)
        
        if cmax_steer > gmax_steer:
            gmax_steer = cmax_steer
        if cmin_steer  < gmin_steer :
            gmin_steer =cmin_steer 
            
           
    
    for i in range(len(action)):
        action[i][:,:,0:4] = (action[i][:,:,0:4] - float(gmin_wheel))/(float(gmax_wheel) - float(gmin_wheel))
        
        action[i][:,:,4] = (action[i][:,:,4] - float(gmin_steer))/(float(gmax_steer) - float(gmin_steer))
        
    return action    
    
 # ADD THIS LINE
def normalisation(matrix):
    
    # print matrix.size()
    cmax, cmin = numpy.max(matrix), numpy.min(matrix)
    
    
    norm = cmax - cmin
    #norm=numpy.linalg.norm(matrix, ord=1)
    if norm==0:
        norm=numpy.finfo(matrix.dtype).eps
        
        
    return (matrix - cmin) / norm


def gensmallData(size1, size2, nDirection):
        
        #print size1, size2
        
       
        
        length = size1 * size2 * nDirection
        
        
        OutputData = torch.ones((length, 1, 1, size1, size2))
        InputData = torch.ones((length, 1, 1, size1, size2)) 
        
        #InputData = InputData.cuda()
        
        action = torch.zeros((length, 1,  2))
        
        step = 0
        
        for direction in range(0, nDirection):
        
            if direction == 0:
        
            
                for l in range(0, size1*size2):
                    h = l / size2 
                    w = l % size2
                 
                        
                    #InputData[l,0, :, h,w]=1/1.1
                     
                    InputData[l,0, 0, h,w] = 0.1
                    #else:
                        #InputData[l,0, :, h,0] = 1
                        
                    
                        
                        
                    action[step,0, 0] = 1
                    step += 1
                        
            elif direction == 1:
            
             
                for l in range(0, size1*size2):
                        w = l / size1 
                        h = l % size1
                
                
                        
                        
                        #InputData[l,0, :, h,w]=1/1.1
                        #if w + 1 < size2:
                        #    InputData[l,0, :, h,w+1] = 1
                        #else:
                        #    InputData[l,0, :, h,0] = 1
                        InputData[l+size1*size2,0, 0, h,w] = 0.1 
            
                        action[step,0, 1] = 1
                        step += 1
                                 
        #for i in range(0, length):
            
            
            
         #   savematrix(InputData[i,0,0,:,:], "input_"+str(i)+".txt")
            
       # OutputData = OutputData.cuda()      
        #action = action.cuda()
        
        return InputData, OutputData, action

def load_driving_files(folder_name, size1, size2):
    
        
        import os
        import glob

        list = os.listdir(folder_name) # dir is your directory path
        length = len(list)
     #print size1, size2
        
        action_tensor = torch.zeros((length, 1,  5))
        InputData = torch.zeros((length, 1, 1, size1, size2)) 
        
        i = 0
    
        for infile in sorted(glob.glob(folder_name + "/*.pgm")):
            
            if VERBOSE > 1:
                print("Current File Being Processed is: " + infile)
            
            action, image = read_pgm(infile, byteorder='<')
            action_tensor[i,:,:] = torch.Tensor(action)
            img1_corr = (image / 65535.)
            #img1_corr = (image) # / 65535.)
            InputData[i, 0, 0, 0:60, :] = torch.Tensor((img1_corr))
            #InputData[i, 0, 1, :, :] =  (InputData[i, 0, 0, :, :])
            #InputData[i, 0, 2, :, :] =  (InputData[i, 0, 0, :, :]) 
            i += 1
        
    
        if VERBOSE > 1:
            print("load image files completed!")
        
    
        
     
    #action_gpu = action.cuda()
        OutputData =  InputData 
        
        return InputData, OutputData, action_tensor, i
    
def load_model(file):
    
    model = torch.load(file)
    
    return model

def load_driving_files_new(folder_name, size1, size2):
    
        
        import os
        import glob
        import subprocess
        from scipy.misc import imread, imresize

        list = os.listdir(folder_name) # dir is your directory path
        length = len(list)
     #print size1, size2
        
        action_tensor = torch.zeros((length, 1,  5))
        InputData = torch.zeros((length, 1, 1, size1, size2)) 
        
        i = 0
    
        for infile in sorted(glob.glob(folder_name + "/*.pgm")):
            
            if VERBOSE > 1:
                print("Current File Being Processed is: " + infile)
            
            action, _ = read_pgm(infile, byteorder='<')
            action_tensor[i,:,:] = torch.Tensor(action)
            
            outfile = infile.replace('.pgm', '.jpg')
            
            
            
            subprocess.call(["convert",infile,outfile])
            #img1_corr = (image / 255.)
            #img1_corr = (image) # / 65535.)
            im = imresize(imread(outfile),(60,80)) / 255.
            
            
            
            
            subprocess.call(["rm",outfile])
            
            if VERBOSE > 1:
                print "...Converting file ", infile
            
            
            
            
            
            InputData[i, 0, 0, 0:60, :] = torch.Tensor(im.tolist())
            #InputData[i, 0, 1, :, :] =  (InputData[i, 0, 0, :, :])
            #InputData[i, 0, 2, :, :] =  (InputData[i, 0, 0, :, :]) 
            i += 1
        
    
        if VERBOSE > 1:
            print("load image files completed!")
        
    
        
     
    #action_gpu = action.cuda()
        OutputData =  InputData 
        
        return InputData, OutputData, action_tensor, i

def savematrix(array, filename):
      
    pic = numpy.zeros((array.size()), dtype=numpy.uint8)
    pic  = normalisation(array.numpy())
     
    numpy.savetxt(filename, pic)
 


    

if __name__ == '__main__':
    #_test_one_layer_model()
    #_test_two_layer_model()
    #_test_L_layer_model()
    
    In_data = []
    Out_data = []
    Action_data = []
    Length = []
    TotalLength = 0
    count = 0
    #in_data1, out_data1, action1 = gensmallData(4*2**(LAYERS-1), 6*2**(LAYERS-1),2)
    # folder_names = ["driving/ConstructionSite-left", "driving/CrazyTurn-left", "driving/DancingLight-left" , "driving/InternOnBike-left", "driving/SafeTurn-left", "driving/Squirrel-left"]
    folder_names = ["driving/ConstructionSite-left", "driving/CrazyTurn-left", "driving/DancingLight-left"  , "driving/InternOnBike-left", "driving/SafeTurn-left", "driving/Squirrel-left"]
    
    
    for name in folder_names:
        # folder_in_data1, folder_out_data1, folder_action1, i = load_driving_files(name, 15*2**(LAYERS-1), 20*2**(LAYERS-1))
        folder_in_data1, folder_out_data1, folder_action1, i = load_driving_files_new(name, 16*2**(LAYERS-2), 20*2**(LAYERS-2))
        # normalisation(folder_action1)
        In_data.append(folder_in_data1)
        Out_data.append(folder_out_data1)
        Action_data.append(folder_action1)
        Length.append(i)
        TotalLength += i
        count += 1 
    #in_data2, out_data2, action2 = gensmallData(4*2**4, 6*2**4,2)
    #Length = [4*2**4 * 6*2**4 * 2]
    #TotalLength = 4*2**4 * 6*2**4 * 2
    #count = 1
    Action_data = normalisation_action_list(Action_data)
    #_test_training(in_data2, out_data2, action2, Length, TotalLength, count)
    _test_training(In_data, Out_data, Action_data, Length, TotalLength, count) #, load_model_name = "/home/joni33/Dropbox/pytorch/320model.pt")



