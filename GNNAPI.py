#network analysis API

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from visdom import Visdom
#V 04.03.23

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = 1e9

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss- self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            # print(self.min_validation_loss- self.min_delta,validation_loss)
        else:
            # print(self.min_validation_loss- self.min_delta,validation_loss)
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
# class EarlyStopper:
#     def __init__(self, patience=1, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.min_validation_loss = np.inf

#     def early_stop(self, validation_loss):
#         if validation_loss < self.min_validation_loss:
#             self.min_validation_loss = validation_loss
#             self.counter = 0
#         elif validation_loss >= (self.min_validation_loss + self.min_delta):            
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False

def buildLaplacian(A, normalize = True):
    w = np.sum(A, axis=1)
    L = np.diag(w) - A
    if normalize:
        L /= (w + (w==0)).reshape((-1,1))
    return L

def checksum(x, previous):
    return np.mean((x.detach().cpu().numpy() - previous) ** 2).item()

#Vanilla neural network implementation

class VNNLayer(nn.Module): #layer of the VNN
    batchnorm = False #!!!!
    def __init__(self, in_features, out_features, dropout=0.0, batchnorm = batchnorm):
        super(VNNLayer, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(in_features, out_features,device='cuda')) # 0.5 * torch.eye(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(1, out_features,device='cuda')) # -0.5 * torch.ones(1, out_features))
        self.dropout = dropout
        #xavier initialization
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.bias)
        self.in_features = in_features
        self.out_features = out_features
        self.batchnorm = batchnorm

    def forward(self, input):
        if self.batchnorm:
            v0 = nn.BatchNorm1d(self.in_features)(input)
        else:
            v0 = input
        v0 = v0.to(torch.device("cuda"))
        v1 = torch.mm(v0, self.weight1)
        output = v1 + self.bias
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output

    def train(self,train_data,label,batch_size):
        num_batches = len(train_data) // batch_size

        for i in range(num_batches):
            # Generate mini-batch indices
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Extract mini-batch data and labels
            inputs = train_data[start_idx:end_idx]

            # Forward pass
            outputs = forward(inputs)


    def param_checksum(self, previous = {}, cumulative = True): #checksum for controlling values or change of the coefficients #cumulative with previous - for aggregated parameter difference tracking
        checksum_ = {'weight1' : checksum(self.weight1, previous.get('weight1', 0)), 'bias': checksum(self.bias, previous.get('bias', 0))}
        if cumulative == True:
            checksum_ = (checksum_['weight1'] * self.in_features + checksum_['bias']) / (self.in_features + 1)
        return checksum_

    def out_params(self): #output parameters for saving pre-trained configuration
        return {'weight1' : self.weight1.data.cpu().numpy().copy(), 'bias': self.bias.data.cpu().numpy().copy()}

    def init_params(self, params):
        if 'bias' in params.keys():
            self.bias = nn.Parameter(torch.FloatTensor(params['bias']))
        if 'weight1' in params.keys():
            self.weight1 = nn.Parameter(torch.FloatTensor(params['weight1']))

VNNDefaultConfig = { #default VNN configuration
    'in_features' : 1,
    'out_features' : 1,
    'layer_dims' : [],
    'dropout' : 0,
    'batchnorm' : False,
    'actfunc' : nn.ReLU(),
    'actfuncFinal' : nn.ReLU(),
    'initSeed' : 1
}

def buildVNNConfig(updates): #build VNN configuration by updating the default config
    Config = VNNDefaultConfig.copy()
    Config.update(updates)
    return Config

class NNmodel(nn.Module): #NN model class template
    iteratelearning : 0
    reloadBestAfterEpochs: 0
    brokenGradientReload: False
    #def __init__(self):
    def forward(self, X): #abstract method forward
        pass


    def params(self): #abstract method for the set of trainable parameters
        return None

    def out_params(self): #abstract method for getting the set of parameters for saving/passing the model setpu
        return None

    def init_params(self, params): #initialize parameters manually or through pre-saved model configuration
        pass

    def batch(self, X, Y, batchsize = 0): #default batching method; could be overriden
        # mh edited for situation batchsize = 0
        print('in batch function')
        batchind = range(X.shape[0])
        if batchsize > 0:
            batchind = np.random.choice(batchind, batchsize) #batching
            return X[batchind, :], Y[batchind, :]
        else:
            print('return X,Y')
            return X,Y

    #handling fit events
    # def fitInitHandle(self):
    #     print("Optimization starting")

    def fitEpochHandle(self):
            print('Epoch: {:04d} of {:04d}'.format(self.fitstate['epoch'] + 1, self.fitstate['n_epochs']), 'batch loss: {:.8f}'.format(self.fitstate['loss']), 
                'full loss: {:.8f}'.format(self.fitstate['full_loss']),
                  'best loss: {:.8f}'.format(self.fitstate['best_loss']),
                  'time elapsed: {:.4f}s'.format(time.time() - self.fitstate['start_time']))

    def fitFinalHandle(self):
        if not(self.fitstate.get('best_params') is None): #initialize with best parameters found so far
            self.init_params(self.fitstate.get('best_params'))
        print("Optimization Finished with loss", self.fitstate['final_loss'])
        print("Total time elapsed: {:.4f}s".format(time.time() - self.fitstate['start_time']))

    def bestUpdateHandle(self):
        self.fitstate['best_params'] = self.out_params()

    def epochsNoBestHandle(self):
        self.fitstate['iterations_nobest'] = self.fitstate.get('iterations_nobest', 0) + 1
        if (self.reloadBestAfterEpochs > 0) & (self.fitstate['iterations_nobest'] >= self.reloadBestAfterEpochs) & self.brokenGradientReload: #restore previous best setup after a series of unsuccesful epochs
            if not(self.fitstate.get('best_params') is None):
                self.init_params(self.fitstate.get('best_params'))
                self.fitstate['iterations_nobest'] = 0
                self.brokenGradientReload = False


    # def initoptimizer(self, lr): #initialize optimizer with model's trainable parameters and selected learning rate
    #     params = [self.query_projection,self.key_projection,self.g,self.expo,self.expoattn, self.params]
    #     if isinstance(lr, list): #set learning rates
    #         for p in range(len(lr)):
    #             if p < len(params):
    #                 params[p]['lr'] = lr[p]
    #         lr = lr[-1]
    #     elif isinstance(lr, dict): #problematic
    #         for p in range(len(params)):
    #             k = list(params[p].keys())[0]
    #             if k in lr.keys():
    #                 params[p]['lr'] = lr[k]
    #         lr = lr['default']

    #     return optim.Adam(params, lr=lr)

    #model fit - generic approach, to be adjusted with overloaded batching and event handlers
    def fit(self,city,A,X,OD,between_fts,Avalid=None,Xvalid=None,ODvalid=None,between_fts_valid=None, 
            n_epochs = 1000, lr = 0.005, interim_output_freq = 200,
            full_loss_freq = 20, SEED = 1,early_stop=True): #torch fit
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        start_time = time.time()
        self.fitstate = {'epoch' : -1}
        early_stopper = EarlyStopper(patience=3, min_delta=0)
        early_stopper_train = EarlyStopper(patience=1, min_delta=1e-3)
        # params = [*self.parameters()]
        # print()
                            # [self.GNNLayers[i].parameters() for i in range(self.GNNLayerNum)]+[self.OutVNN.parameters()]
        optimizer = optim.Adam(self.parameters(), lr=lr[0])
        best_loss = np.Inf
        # self.fitInitHandle()

        t_1run = time.time()

        self.fitstate = {'epoch': 0, 'n_epochs': n_epochs, 'loss': None, 'full_loss': None, 'start_time': start_time, 'Y': None,'loss_list':[]}

        # viz = Visdom()
        
        

        # viz.line([0.], [0.], win=city,name='train')
        if early_stop:
        	pass
            # viz.line([[0.0]], [0.], win=city,name='valid', opts=dict(title=city, legend=['valid']))
        for epoch in range(n_epochs):
            self.fitstate['epoch'] = epoch
            

            # if self.iteratelearning:
            # optimizer = self.initoptimizer(lr)
            
            optimizer.zero_grad()
            # mh for trip distribution without parital OD
            # Y = self.forward(OD=OD, X = X_b)
            Y = self.forward(A,X,OD,between_fts)
            self.fitstate['Y'] = Y
            # if self.model == 'p':
            #     loss = self.loss(Y,p)

            # else:
            loss = self.loss(Y, OD)
            # print(loss)
            loss.backward()
            optimizer.step()
            self.fitstate['loss'] = loss.item()
            self.fitstate['loss_list'] += [loss.item()]
            # viz.line([loss.item()], [epoch], win=city,name='train', update='append')
            # print(loss)
            # print('--------------------------------')
            if Avalid is None:
                if early_stopper_train.early_stop(loss.item()):
                    print('early stop based on training data at '+str(epoch)+' epochs')             
                    break

            else:
                if early_stop:
                    validation_loss = self.evaluate(Avalid,Xvalid,ODvalid,between_fts_valid)
                    # print('validation_loss',validation_loss)
                    # viz.line([validation_loss], [epoch], win=city,name='valid', update='append')

                    if early_stopper.early_stop(validation_loss):
                        print('early stop at '+str(epoch)+' epochs')             
                        break

        # self.fitstate['final_loss'] = best_loss.item()
        # self.fitFinalHandle()
        # return self.fitstate['final_loss']
    def evaluate(self, A,X,OD,between_fts): #evaluate model performance on the new data
        OD = torch.FloatTensor(OD)
        OD = OD.to(torch.device("cuda"))
        win = OD.sum(axis = 0,keepdims=True)
        win = win.to(torch.device("cuda"))
        p = OD / win
        with torch.no_grad():
            # print(X)
            Y = self.forward(A,X,OD,between_fts)

            loss = self.loss(Y, OD)
            return loss.item()



#VNN MLP model
class VNN_MLP(NNmodel):
    #actfunc = nn.SELU()
    #actfuncFinal = nn.ReLU() #nn.Softmax(dim=1)

    def __init__(self, VNNConfig):
        #super(VNN_MLP, self).__init__()
        np.random.seed(VNNConfig['initSeed'])
        torch.manual_seed(VNNConfig['initSeed'])
        super().__init__()

        self.n_layers = len(VNNConfig['layer_dims']) + 1
        self.reloadBestAfterEpochs = 0
        self.iteratelearning = 0

        if self.n_layers > 1:
            layers = [VNNLayer(VNNConfig['in_features'], VNNConfig['layer_dims'][0], VNNConfig['dropout'], VNNConfig['batchnorm'])]
            for i in range(len(VNNConfig['layer_dims']) - 1):
                layers.append(VNNLayer(VNNConfig['layer_dims'][i], VNNConfig['layer_dims'][i + 1], VNNConfig['dropout'], VNNConfig['batchnorm']))
            layers.append(VNNLayer(VNNConfig['layer_dims'][-1], VNNConfig['out_features'], dropout = 0, batchnorm = False)) #outputlayer without batchnorm
        else:
            layers = [VNNLayer(VNNConfig['in_features'], VNNConfig['out_features'], dropout = 0, batchnorm = False)] #outputlayer without batchnorm

        self.layers = nn.ModuleList(layers)
        self.actfunc = VNNConfig['actfunc']
        self.actfuncFinal = VNNConfig['actfuncFinal']

    def forward(self, X):
        # Y = [X]
        for i in range(self.n_layers - 1): #is it ok to overload the tensor?
            X= self.actfunc(self.layers[i](X))

        return self.actfuncFinal(self.layers[-1](X))
    # def trian(self,X,Y):

    def param_checksum(self, previous = {}, cumulative = True):
        checksum = {'MLPlayer{}'.format(l) : self.layers[l].param_checksum(previous = previous.get('MLPlayer{}'.format(l),{}), cumulative = cumulative) for l in range(self.n_layers)}
        if cumulative:
            checksum = np.mean([checksum[k] for k in checksum.keys()])
        return checksum

    def parameters(self):
        return self.layers.parameters()

    def params(self):
        #return [{'params' : self.layers[l].paramters()} for l in range(self.n_layers)]
        return [{'params' : self.layers.parameters()}]

    def out_params(self):
            return {'MLPlayer{}'.format(l) : self.layers[l].out_params() for l in range(self.n_layers)}

    def init_params(self, params):
        for l in range(self.n_layers):
            self.layers[l].init_params(params.get('MLPlayer{}'.format(l), {}))

class VNN_MLP_BCE(VNN_MLP): #VNN MLP with Binary Cross-Entropy objective function
    def loss(self, Y = None, Ytrue = None):
        return nn.BCELoss()(Y, Ytrue)

dmerge = lambda d1, d2: d1.update(d2) or d1
class GNN_VNN_Layer(nn.Module): #GNN Layer
    def __init__(self, in_features, VNNConfig):
        super().__init__()
        #VNN_Config specifies the activation function VNN, including output dimensionality, in_features overrides the VNNConfig
        # mh commented out, AL will no longer be an argument to initiate GNN_VNN_layers
        # self.AL = AL #a list of adjacency-Laplacian matrixes for network layers
        # self.AN = len(AL)
        # mh: 1 is for X, 13 is for facility_t from POI
        # self.VNN = VNN_MLP(dmerge(VNNConfig, {'in_features' : in_features * (self.AN + 1)}))
        self.VNN = VNN_MLP(dmerge(VNNConfig, {'in_features' : in_features * (1 + 1)}))
        # print(self.VNN)
    def forward(self, AL,X): #X - matrix of node features, axis0 - nodes, axis1 - features
        # print('(X.shape',X.shape)
        # print('(A.shape',AL[0].shape)
        # print('----------------------torch.matmul(A, X)-----------------------------')
        # print(torch.matmul(A, X))
        X_ = torch.cat([X] + [torch.matmul(A, X) for A in AL], dim = 1)
        # print(X_.shape)
        # print('------------------------X_----------------------------')
        # print(X_)
        return self.VNN.forward(X_)

    def param_checksum(self):
        return {'VNN' : self.VNN.param_checksum()}

    def param_checksum(self, previous = {}, cumulative = True):
        checksum_ = {'VNN' : self.VNN.param_checksum(previous.get('VNN', {}), cumulative = cumulative)}
        if cumulative:
            checksum_ = checksum_['VNN']
        return checksum_

    def params(self):
        return self.VNN.params()

    def out_params(self):
        return {'VNN': self.VNN.out_params()}

    def init_params(self, params):
        self.VNN.init_params(params.get('VNN', {}))

