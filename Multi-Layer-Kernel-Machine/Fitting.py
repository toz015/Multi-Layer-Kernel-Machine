import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.utils import resample

import torch
from torch.utils.data import Dataset,DataLoader
from torch import optim

from .Mydataset import mydataset

class MultilayerFitting:
    """Multi-layer Fitting
    Parameters
    ----------
    model_fit : chosen network structure
    device_fit : "cpu" or "cuda"
    train_loader_fit : dataloader
        training data
    test_loader_fit : dataloader
        test data
    epochs_fit : int
        maximum epoch number
    criterion_fit : chosen criterion
    optimizer_fit : chosen optimizer
    terminate_fit : int
        terminate parameter
    print_fit : 
        print parameter
    printchoice : bool
        print results or not
    """
    
    def __init__(self, model_fit,device_fit,train_loader_fit, test_loader_fit,
                 epochs_fit, criterion_fit, optimizer_fit, 
                 terminate_fit=10, print_fit=10,printchoice=True):
        self.model_fit = model_fit
        self.device_fit = device_fit
        self.train_loader_fit = train_loader_fit
        self.test_loader_fit = test_loader_fit
        self.epochs_fit = epochs_fit
        self.criterion_fit = criterion_fit
        self.optimizer_fit = optimizer_fit
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
            trainning loss
        testloss : list
            test loss
        prediction : list
            model prediction for test data
        """
        
        trainloss=[]
        testloss=[]
        t0=time.time()
        net_fit = self.model_fit
        net_fit = net_fit.to(self.device_fit)
        for epoch in range(self.epochs_fit): 
            for x, y in self.train_loader_fit: #for batch, (x, y) in enumerate(train_loader): 
                x, y = x.to(self.device_fit), y.to(self.device_fit)
                # Compute prediction error
                y_pred = net_fit(x)
                y_pred = torch.squeeze(y_pred)
                train_loss = self.criterion_fit(y_pred, y)
                # Backpropagation
                self.optimizer_fit.zero_grad() 
                train_loss.backward()
                self.optimizer_fit.step()

            for x, y in self.test_loader_fit:
                x, y = x.to(self.device_fit), y.to(self.device_fit)
                y_test_pred = net_fit(x)
                y_test_pred = torch.squeeze(y_test_pred)
                test_loss = self.criterion_fit(y_test_pred,y)
            
            if epoch>self.terminate_fit and float(test_loss)>max(testloss[-self.terminate_fit:-1]):
                break
            
            if epoch % self.print_fit == 0 and self.printchoice==True:        
                print(f'''epoch {epoch}
                    Train set - loss: {train_loss}
                    Test  set - loss: {test_loss}
                    ''')
            
            trainloss.append(float(train_loss))
            testloss.append(float(test_loss))
        if self.printchoice==True:              
            fit = time.time() - t0
            print("Model fitted in %.3f s" % fit)
        
        
        x0=torch.from_numpy(test_x.to_numpy()).float()
        with torch.no_grad():
            x0 = x0.to(self.device_fit)
            pred = net_fit(x0)
            pred = torch.Tensor.cpu(pred)
            prediction=np.array(pred).reshape(-1)
        
        return(trainloss,testloss,prediction)
    
    
    def Bootstrap(self,time_boot,bootbase,train_x,train_y,test_x,test_y, batch,init_weights):
        """Bootstrap confidence interval
        Parameters
        ----------
        time_boot : int
            Boostrap times
        bootbase : list
            basic prediction
            
        train_x : DataFrame
        train_y : Series
        test_x : DataFrame
        test_y : Series
            global data
        batch : int
            batch size
        init_weights : function
            initialization 
             
        Returns
        -------
        length : float
            confidence interval length
        coverage : %
            confidence interval 95% coverage
        """
        
    
        if time_boot <40:
            raise Exception('Too Small Bootstrap Times')
        bootlist=[]
        for boot in tqdm(range(time_boot)):
            alltrain=pd.concat([train_y,train_x],axis=1) 
            datachoose=resample(alltrain,replace=True)
            bootstrap_y=datachoose.iloc[:,0]
            bootstrap_x=datachoose.iloc[:,1:]
            bootstrap_x.reset_index(drop=True, inplace=True) 
            bootstrap_y.reset_index(drop=True, inplace=True) 
            nnbootstrap_x = torch.from_numpy(bootstrap_x.to_numpy()).float()
            nnbootstrap_y = torch.squeeze(torch.from_numpy(bootstrap_y.to_numpy()).float()) 
            
            boottrain_loader_boot = DataLoader(mydataset(nnbootstrap_x, nnbootstrap_y),batch_size=batch, shuffle=True)
            
            net_fit = self.model_fit
            net_fit = net_fit.to(self.device_fit)
            torch.manual_seed(0)
            net_fit.apply(init_weights)
            
            trainloss=[]
            testloss=[]
            for epoch in range(self.epochs_fit): 
                for x, y in boottrain_loader_boot:
                    x, y = x.to(self.device_fit), y.to(self.device_fit)
                    # Compute prediction error
                    y_pred = net_fit(x)
                    y_pred = torch.squeeze(y_pred)
                    train_loss = self.criterion_fit(y_pred, y)
                    # Backpropagation
                    self.optimizer_fit.zero_grad() 
                    train_loss.backward()
                    self.optimizer_fit.step()

                for x, y in self.test_loader_fit:
                    x, y = x.to(self.device_fit), y.to(self.device_fit)
                    y_test_pred = net_fit(x)
                    y_test_pred = torch.squeeze(y_test_pred)
                    test_loss = self.criterion_fit(y_test_pred,y)
                    testloss.append(float(test_loss))
                
                if epoch>self.terminate_fit and float(test_loss)>max(testloss[-self.terminate_fit:-1]):
                    break

            ######
            x0=torch.from_numpy(test_x.to_numpy()).float()
            with torch.no_grad():
                x0 = x0.to(self.device_fit)
                pred = net_fit(x0)
                pred = torch.Tensor.cpu(pred)
                bootlist.append(np.array(pred).reshape(-1))

        sorted_bootlist = [sorted(x)[:] for x in np.array(bootlist).T]
        sorted_bootlist=np.array(sorted_bootlist)
        
        lowq=int(time_boot*0.025)
        upq=time_boot-1-int(time_boot*0.025)
        lower=bootbase-(sorted_bootlist[:,upq]-bootbase)
        upper=bootbase-(sorted_bootlist[:,lowq]-bootbase)
        print("confidence interval length",sorted_bootlist[:,upq]-sorted_bootlist[:,lowq])
        dnn_length=(sorted_bootlist[:,upq]-sorted_bootlist[:,lowq]).mean()
        print("average confidence interval length",dnn_length)

        cover=0
        for i in range(len(test_y)):
            if lower[i]<=test_y[i] and upper[i]>=test_y[i]:
                cover=cover+1
        coverage=cover/len(test_y)
        print("95 coverage",coverage)
        return dnn_length,coverage
    
    
    def GradientBand(self,loss,train_x,train_y,test_x,test_y, batch,pen=0):
        """Gradient-based confidence interval
        Parameters
        ----------
        loss : list
            training loss (used for RSS)
        pen : float
            penalty parameter
            
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

        
        if pen <0:
            raise Exception('Invalid Penalty')
            
        net_fit = self.model_fit
        net_fit = net_fit.to(self.device_fit)
        par=self.optimizer_fit.param_groups[0]['params']
        for i in tqdm(range(len(train_x))):
            x0=torch.from_numpy(train_x[i:1+i].to_numpy()).float()
            x0 = x0.to(self.device_fit)
            pred = net_fit(x0)
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
        
        
        if pen>=0:
            #original
            temp=torch.linalg.inv(Fi.T @ Fi)
            length=[]
            coverage=0
            mark=0
            for i in tqdm(range(len(test_x))):
                x0=torch.from_numpy(test_x[i:i+1].to_numpy()).float()
                x0 = x0.to(self.device_fit)
                pred = net_fit(x0)
                par=self.optimizer_fit.param_groups[0]['params']
                f0=torch.tensor([]).to(self.device_fit)
                for j in range(len(par)):
                    par[j].grad.data.zero_()
                pred.backward()
                for j in range(len(par)):
                    f0=torch.cat([f0,par[j].grad.reshape(-1)])
                f0=f0.reshape(-1,1)

                fFFf=f0.T @ temp @ f0
                
                if fFFf < 0:
                    continue
                mark=mark+1
                dd=1.96*np.sqrt(float(fFFf+1))*np.sqrt(loss[-1])
                length.append(2*dd)
                
                #coverage
                if torch.Tensor.cpu(pred.detach()).numpy()[0][0]-dd<test_y[i] and torch.Tensor.cpu(pred.detach()).numpy()[0][0]+dd>test_y[i]:
                    coverage=coverage+1
            coverage=coverage/mark
            
            if pen==0:
                print("n-p:",len(train_x)-f0.shape[0]," mark:",mark) 
                print("length",np.mean(length))
                print("95 coverage",coverage)
                 
            else:
                #############penalty
                temp2=torch.linalg.inv(Fi.T @ Fi+pen *np.eye(Fi.T.shape[0]))
                temp2=temp2.float()
                temp=temp2@Fi.T @ Fi@temp2
                p=Fi @temp2 @Fi.T
                print(len(train_x)-np.trace(2*p-p@p))
                corr=(len(train_x)-f0.shape[0])/(len(train_x)-np.trace(2*p-p@p))

                length=[]
                coverage=0
                mark=0
                for i in tqdm(range(len(test_x))):
                    x0=torch.from_numpy(test_x[i:i+1].to_numpy()).float()
                    x0 = x0.to(self.device_fit)
                    pred = net_fit(x0)
                    par=self.optimizer_fit.param_groups[0]['params']
                    f0=torch.tensor([]).to(self.device_fit)
                    for j in range(len(par)):
                        par[j].grad.data.zero_()
                    pred.backward()
                    for j in range(len(par)):
                        f0=torch.cat([f0,par[j].grad.reshape(-1)])
                    f0=f0.reshape(-1,1)
                    fFFf=f0.T @ temp @ f0
                    
                    if fFFf < 0:
                        continue
                    mark=mark+1
                    dd=1.96*np.sqrt(float(fFFf+1))*np.sqrt(loss[-1])*corr
                    length.append(2*dd)
                    
                    #coverage
                    if torch.Tensor.cpu(pred.detach()).numpy()[0][0]-dd<test_y[i] and torch.Tensor.cpu(pred.detach()).numpy()[0][0]+dd>test_y[i]:
                        coverage=coverage+1
                coverage=coverage/mark

                print("n-p:",len(train_x)-f0.shape[0]," mark:",mark) 
                print("length",np.mean(length))
                print("95 coverage",coverage)
        return length,coverage
    
    def HomoConformalBand(self,train_x,train_y,test_x,test_y, batch):
        """Conformal confidence interval
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
        length : float
            confidence interval length
        coverage : %
            confidence interval 95% coverage
        
        References
        ----------
        [1] Lei, Jing, et al. "Distribution-free predictive inference for regression."
        Journal of the American Statistical Association 113.523 (2018): 1094-1111.
        """
        
        ##conformal prediction 
        net_fit = self.model_fit
        net_fit = net_fit.to(self.device_fit)
        x0=torch.from_numpy(train_x[:].to_numpy()).float()
        with torch.no_grad():
            x0 = x0.to(self.device_fit)
            pred = net_fit(x0)
            score=torch.abs(pred.reshape(-1)-torch.Tensor(train_y[:]).to(self.device_fit))
        sorted_score, sorted_indices=torch.sort(score)
        q=(len(train_x)+1)*0.95
        print(np.ceil(q))
        a=sorted_score[int(np.ceil(q))]

        coverage=0
        for i in range(len(test_x)):
            if torch.Tensor.cpu(pred.detach()).numpy()[0][0]-a<test_y[i] and torch.Tensor.cpu(pred.detach()).numpy()[0][0]+a>test_y[i]:
                coverage=coverage+1
        coverage=coverage/len(test_x)

        print("length",2*a)
        print("95 coverage",coverage)
        return 2*a,coverage
    
    def HeteConformalBand(self,loss,train_x,train_y,test_x,test_y, batch):
        """PLUS Conformal confidence interval
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
        
        ##conformal prediction
        net_fit = self.model_fit
        net_fit = net_fit.to(self.device_fit)
        par=self.optimizer_fit.param_groups[0]['params']
        for i in tqdm(range(len(train_x))):
            x0=torch.from_numpy(train_x[i:1+i].to_numpy()).float()
            x0 = x0.to(self.device_fit)
            pred = net_fit(x0)
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
        temp=torch.linalg.inv(Fi.T @ Fi)
         

        mark=0
        score=torch.tensor([])
        for i in tqdm(range(len(train_x))):
            x0=torch.from_numpy(train_x[i:i+1].to_numpy()).float()
            x0 = x0.to(self.device_fit)
            pred = net_fit(x0)
            par=self.optimizer_fit.param_groups[0]['params']
            f0=torch.tensor([]).to(self.device_fit)
            for j in range(len(par)):
                par[j].grad.data.zero_()
            pred.backward()
            for j in range(len(par)):
                f0=torch.cat([f0,par[j].grad.reshape(-1)])
            f0=f0.reshape(-1,1)
            fFFf=f0.T @ temp @ f0
            
            #
            #
            tranaa=np.abs(torch.Tensor.cpu(pred.detach()).numpy()[0][0]-train_y[i])/np.sqrt(loss[-1])/np.sqrt(torch.Tensor.cpu(fFFf)+1)
        
            score=torch.cat([score,tranaa])
            if fFFf < 0:
                continue
            mark=mark+1
        score=score.reshape(-1)
        sorted_score, sorted_indices=torch.sort(score)
        q=(len(train_x)+1)*0.95
        print(np.ceil(q))
        a=sorted_score[int(np.ceil(q))]
        
        coverage=0
        mark=0
        length=[]
        for i in tqdm(range(len(test_x))):
            x0=torch.from_numpy(test_x[i:i+1].to_numpy()).float()
            x0 = x0.to(self.device_fit)
            pred = net_fit(x0)
            par=self.optimizer_fit.param_groups[0]['params']
            f0=torch.tensor([]).to(self.device_fit)
            for j in range(len(par)):
                par[j].grad.data.zero_()
            pred.backward()
            for j in range(len(par)):
                f0=torch.cat([f0,par[j].grad.reshape(-1)])
            f0=f0.reshape(-1,1)

            fFFf=f0.T @ temp @ f0
            
            if fFFf < 0:
                continue
            mark=mark+1
            dd=(np.sqrt(loss[-1])*np.sqrt(torch.Tensor.cpu(fFFf)+1)*a).detach().numpy()[0][0]
            length.append(2*dd)
            
            #coverage
            if torch.Tensor.cpu(pred.detach()).numpy()[0][0]-dd<test_y[i] and torch.Tensor.cpu(pred.detach()).numpy()[0][0]+dd>test_y[i]:
                coverage=coverage+1
        coverage=coverage/mark


        print("length",np.mean(length))
        print("95 coverage",coverage)
        return np.mean(length),coverage