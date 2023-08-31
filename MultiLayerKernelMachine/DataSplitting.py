import pandas as pd
import numpy as np
import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset,DataLoader
from torch import optim

class DataSplitting:
    """Data Splitting
    Parameters
    ----------
    split_fit : int
        the number of splits 
        = the number of hidden layer
    modelset_fit :  list (shape: split_fit)
        network list
    device_fit : "cpu" or "cuda"
    train_loaderset_fit : list (shape: split_fit x split_fit)
        training dataloader set
    test_loader_fit : dataloader
        test data
    epochs_fit : int
        maximum epoch number
    criterion_fit : chosen criterion
    optimizerset_fit : list (shape: split_fit x split_fit)
        chosen optimizer set
    terminate_fit : int
        terminate parameter
    print_fit : 
        print parameter
    printchoice : bool
        print results or not
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
        
    def fitting(self,train_x,train_y,test_x,test_y, batch):
        """Fitting procedure
        Parameters
        -------
        train_x : DataFrame
        train_y : Series
        test_x : DataFrame
        test_y : Series
            global data
        batch : int
            batch size
            
        Returns
        -------
        trainloss : list
            trainning loss (on average)
        testloss : list
            test loss (on average)
        prediction : list
            model prediction for test data (on average)
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
                        alltrainloss[i].append(float(train_loss[i]))
                        
                        # Backpropagation
                        self.optimizerset_fit[i][l].zero_grad() 
                        train_loss[i].backward()
                        self.optimizerset_fit[i][l].step()
            
            
            for x, y in self.test_loader_fit:
                x, y = x.to(self.device_fit), y.to(self.device_fit)
                for i in range(self.split_fit):
                    y_test_pred = netset_fit[i](x)
                    y_test_pred = torch.squeeze(y_test_pred)
                    
                    test_loss[i] = self.criterion_fit(y_test_pred,y)
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
    
    
    def GradientBand(self,loss,train_x,train_y,test_x,test_y, batch):
        """Gradient-based confidence interval
        Parameters
        ----------
        loss : list
            training loss (used for RSS)
        
        train_x : DataFrame
        train_y : Series
        test_x : DataFrame
        test_y : Series
            global data
        batch : int
            batch size
            
        Returns
        -------
        length : float
            confidence interval length
        coverage : %
            confidence interval 95% coverage
        """
        temp=[]
        parset=[]
        netset_fit = self.modelset_fit
        for spl in range(self.split_fit):
            par1=self.optimizerset_fit[spl][0].param_groups[0]['params']
            for jj in range(1,self.split_fit):
                par1=par1+self.optimizerset_fit[spl][jj].param_groups[0]['params']
            parset.append(par1)
                
        for spl in range(self.split_fit):
            par=parset[spl]
            for i in range(len(train_x)):
                x0=torch.from_numpy(train_x[i:1+i].to_numpy()).float()
                x0 = x0.to(self.device_fit)
                pred = netset_fit[spl](x0)
                fi=torch.tensor([]).to(self.device_fit)
                for j in range(len(par)):
                    par[j].grad.data.zero_()
                pred.backward()   
                for j in range(len(par)): 
                    fi=torch.cat([fi,par[j].grad.reshape(-1)])
                fi=fi.reshape(1,-1)
                if i==0:
                    Fi=fi
                else:
                    Fi=torch.cat([Fi,fi])   
            print(Fi.shape)
            temp.append(torch.linalg.inv(Fi.T @ Fi))
            
        length=[]
        coverage=0
        mark=0
        for i in range(len(test_x)):
            fFFf=0
            for spl in range(self.split_fit):
                par=parset[spl]
                x0=torch.from_numpy(test_x[i:i+1].to_numpy()).float()
                x0 = x0.to(self.device_fit)
                pred = netset_fit[spl](x0)
                par=parset[spl]
                f0=torch.tensor([]).to(self.device_fit)
                for j in range(len(par)):
                    par[j].grad.data.zero_()
                pred.backward()
                for j in range(len(par)):
                    f0=torch.cat([f0,par[j].grad.reshape(-1)])
                f0=f0.reshape(-1,1)
                fFFf=fFFf+f0.T @ temp[spl] @ f0
            
            if fFFf < 0:
                continue
            mark=mark+1
            dd=1.96*np.sqrt(float(torch.Tensor.cpu(fFFf)/self.split_fit**2+1))*np.sqrt(loss[-1])
            length.append(2*dd)
            
            #coverage
            if torch.Tensor.cpu(pred.detach()).numpy()[0][0]-dd<test_y[i] and torch.Tensor.cpu(pred.detach()).numpy()[0][0]+dd>test_y[i]:
                coverage=coverage+1
        coverage=coverage/mark

        print("n-p:",len(train_x)-f0.shape[0]," mark:",mark) 
        print("length",np.mean(length))
        print("95 coverage",coverage)
        return(np.mean(length),coverage)
    