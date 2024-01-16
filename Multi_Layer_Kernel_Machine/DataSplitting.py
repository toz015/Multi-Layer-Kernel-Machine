import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

import torch
from torch.utils.data import Dataset,DataLoader
from torch import optim

class DataSplitting:
    """Data Splitting
    Parameters
    ----------
    split_fit : int
        The number of splits = the number of hidden layer
    modelset_fit :  list of shape (split_fit)
        Network list
    device_fit : char
        Device to use, "cpu" or "cuda"
    train_loaderset_fit : list of shape (split_fit x split_fit)
        Training dataloader set
    test_loader_fit : dataloader
        Testing data
    epochs_fit : int
        Maximum epoch number
    criterion_fit : 
        Chosen criterion
    optimizerset_fit : list of shape (split_fit x split_fit)
        Chosen optimizer set
    terminate_fit : int
        Terminate parameter, terminate if the loss has no significant improvement over T consecutive epochs
    print_fit : int
        Print parameter, print the results after every T epochs
    printchoice : bool
        Print results or not
    """
    
    def __init__(self, split_fit,modelset_fit,device_fit,train_loaderset_fit, test_loader_fit, epochs_fit, 
                 criterion_fit, optimizerset_fit, terminate_fit=10, print_fit=10,printchoice=True):
        self.split_fit = split_fit
        self.modelset_fit = modelset_fit #set
        self.device_fit = device_fit
        self.train_loaderset_fit = train_loaderset_fit #set
        self.test_loader_fit = test_loader_fit
        self.epochs_fit = epochs_fit
        self.criterion_fit = criterion_fit
        self.optimizerset_fit = optimizerset_fit #set
        self.terminate_fit = terminate_fit
        self.print_fit = print_fit
        self.printchoice = printchoice
        
    def fitting(self,train_x,train_y,test_x,test_y):
        """Fitting procedure
        
        Parameters
        -------
        train_x : DataFrame
        train_y : Series
        test_x : DataFrame
        test_y : Series
            Global data
      
        Returns
        -------
        trainloss : list
            Trainning loss (on average)
        testloss : list
            Testing loss (on average)
        prediction : list
            Model prediction for testing data (on average)
        """
        
        train_loss=[]
        test_loss=[]
        for i in range(self.split_fit):
            train_loss.append(0)
            test_loss.append(0)
        alltrainloss=[[] for _ in range(self.split_fit)]
        alltestloss=[[] for _ in range(self.split_fit)]
        spltrainloss=[]
        spltestloss=[]
        t0=time.time()
        netset_fit = self.modelset_fit
        for epoch in range(self.epochs_fit): 
            for i in range(self.split_fit):
                for l in range(self.split_fit):
                    for x, y in self.train_loaderset_fit[i][l]: 
                        # Data Splitting
                        # every step, update a layer with smaller dataset 
                        x, y = x.to(self.device_fit), y.to(self.device_fit)
                        # Compute prediction error
                        y_pred = netset_fit[i](x)
                        y_pred = torch.squeeze(y_pred)
                        train_loss[i] = self.criterion_fit(y_pred, y)
                        
                        # Backpropagation
                        self.optimizerset_fit[i][l].zero_grad() 
                        train_loss[i].backward()
                        self.optimizerset_fit[i][l].step()
            
            x0=torch.from_numpy(train_x[:].to_numpy()).float()
            for i in range(self.split_fit):
                with torch.no_grad():
                    x0 = x0.to(self.device_fit)
                    pred = netset_fit[i](x0)
                    train_loss[i] = mean_squared_error(train_y,pred)
                    alltrainloss[i].append(float(train_loss[i]))
                    
            
            x1=torch.from_numpy(test_x[:].to_numpy()).float()
            for i in range(self.split_fit):
                with torch.no_grad():
                    x1 = x1.to(self.device_fit)
                    pred = netset_fit[i](x1)
                    test_loss[i] = mean_squared_error(test_y,pred)
                    alltestloss[i].append(float(test_loss[i]))
            
            if epoch>self.terminate_fit and float(sum(test_loss)/len(test_loss))>max(spltestloss[-self.terminate_fit:-1]):
                break
            
            if epoch % self.print_fit == 0 and self.printchoice==True:   
                print(f'''epoch {epoch}
                    Train set - loss: {sum(train_loss)/len(train_loss)}
                    Test  set - loss: {sum(test_loss)/len(test_loss)}
                    ''')
            spltrainloss.append(float(sum(train_loss)/len(train_loss)))
            spltestloss.append(float(sum(test_loss)/len(test_loss)))      
                
            
        if self.printchoice==True:              
            fit = time.time() - t0
            print("Model fitted in %.3f s" % fit)
        
        #####
        x0=torch.from_numpy(test_x.to_numpy()).float()
        
        for i in range(self.split_fit):
            if i==0:
                with torch.no_grad():
                    x0 = x0.to(self.device_fit)
                    pred = netset_fit[i](x0)
                    pred = torch.Tensor.cpu(pred)
                    prediction=np.array(pred).reshape(-1)
            else:
                with torch.no_grad():
                    x0 = x0.to(self.device_fit)
                    pred = netset_fit[i](x0)
                    pred = torch.Tensor.cpu(pred)
                    prediction+=np.array(pred).reshape(-1)
        
        return(spltrainloss,spltestloss,prediction/self.split_fit)
    
    