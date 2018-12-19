import os
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataBase import DataBase, DataBase_pkl
import networks
from model import FullNet, ConvNet,ConvNet_Sheng2, FullNet_2,FullNet_skip,ConvNet_skip
from get_freqs import findFreq
import matplotlib.pyplot as plt
from multiscale_resnet import MSResNet,MSResNet2,MSResNet3,ResNet,MSResNet4
import losses
import sys
import argparse
from log import log

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
def accuracyLoss(f1, f2):
    min_dist=np.zeros(f1.shape[0])
    for i in range(f2.shape[1]):
        dist_i = np.min(np.abs(f1-np.expand_dims(f2[:, i],1).repeat(f2.shape[1], axis=1)), axis=1)
        valid_freq=f2[:,i]!=-1
        min_dist=np.maximum(min_dist, dist_i*valid_freq)
    for i in range(f2.shape[1]):
        dist_i = np.min(np.abs(f2-np.expand_dims(f1[:, i],1).repeat(f1.shape[1], axis=1)), axis=1)
        valid_freq=f1[:,i]!=-1
        min_dist=np.maximum(min_dist, dist_i*valid_freq)
    return np.sum(min_dist)

def relative_error(GT,esti):
    #return np.minimum(abs(np.min(esti,1)-np.min(GT,1))/np.min(GT,1) , abs(np.max(esti,1)-np.max(GT,1))/np.max(GT,1))
    return np.sum(abs(np.min(esti,1)-np.min(GT,1))/abs(np.min(GT,1)) + abs(np.max(esti,1)-np.max(GT,1))/abs(np.max(GT,1)))
'''
def relative_error(f,f_hat):
    n = np.size(f,1)
    #Only for two locations
    if  n > 1:
        a = np.sum(np.abs(f-f_hat)/abs(f),axis = 1)/n
        b = np.sum(np.abs(f-f_hat[:,[1,0]])/abs(f),axis = 1)/n
        c = np.sum(np.abs(f-f_hat[:,[0,0]])/abs(f),axis = 1)/n
        d = np.sum(np.abs(f-f_hat[:,[1,1]])/abs(f),axis = 1)/n
        ab = np.minimum(a,b)
        cd = np.minimum(c,d)
        return np.sum(np.minimum(ab,cd))
    else:
        return np.sum(np.abs(f-f_hat)/abs(f))
'''
def eval(dataloader,model,device,epoch, number_of_locations, SNR, used_model,xs,xs2,targets_dimension):
    outputs_all = []
    targets_all = []
    for inputs,spect,targets, targets_ps,_ ,_ in dataloader:
        inputs = torch.cat((inputs,spect), dim=1)
        inputs = inputs.float()
        inputs = inputs.to(device)
        if targets_dimension <=2:
            targets = targets[:,:,0]
        elif targets_dimension == 4:
            targets = torch.cat((targets[:,:,0],targets[:,:,1]),1)
        #targets,_ = targets.sort()
        targets = targets.float()
        targets = targets
        outputs = model(inputs)
        outputs = outputs.data.cpu()
        
        if targets_dimension <=2:
            GT_T1 = targets.cpu() * (max(xs) - min(xs)) + min(xs)
            esti_T1 = outputs.data.cpu()* (max(xs) - min(xs)) + min(xs)
        elif targets_dimension == 4:
            GT_T1 = targets.cpu()[:,0:2] *(max(xs) - min(xs)) + min(xs)
            esti_T1 = outputs.data.cpu()[:,0:2] *(max(xs) - min(xs)) + min(xs) 

            GT_T2 = targets.cpu()[:,2:] *(max(xs2) - min(xs2)) + min(xs2)
            esti_T2 = outputs.data.cpu()[:,2:] *(max(xs2) - min(xs2)) + min(xs2) 

        #outputs,_ = outputs.sort()

        outputs_all.append(esti_T1)
        targets_all.append(GT_T1)
    outputs_all = torch.cat(outputs_all)
    targets_all = torch.cat(targets_all)
    outputs_all_flatted = outputs_all.numpy()#.flatten()
    targets_all_flatted = targets_all.numpy()#.flatten()
    plt.figure()
    for i in range(outputs_all_flatted.shape[1]):
        plt.scatter(targets_all_flatted[:,i], outputs_all_flatted[:,i])
    plt.xlabel('True T1')
    plt.ylabel('Esti T1')
    plt.xlim((min(xs),max(xs)))
    plt.ylim((min(xs),max(xs)))
    #pp = [round(10**(-0.4),1),round(10**(-0.2),1),round(10**0.0,1),round(10**0.2,1),round(10**0.4,1),round(10**0.6,1),round(10**0.8,1)]
    #plt.xticks([-0.4,-0.2,0,0.2,0.4,0.6,0.8],pp)
    #plt.yticks([-0.4,-0.2,0,0.2,0.4,0.6,0.8],pp)
    plt.savefig('./evaluation_figures/model_'+str(used_model)+'eval_'+'nlocations_'+str(number_of_locations)+'_'+str(epoch)+'_'+str(SNR)+'.png')
    return outputs_all_flatted,targets_all_flatted

def plot_ps(outputs,targets_ps, epoch, number_of_locations, SNR):
    plt.figure()
    plt.plot(targets_ps)
    plt.savefig('./evaluation_figures/ps_'+'nlocations_'+str(number_of_locations)+'_'+str(epoch)+'_'+str(SNR)+'.png')
    plt.close()
    plt.figure()
    plt.plot(outputs)
    plt.savefig('./evaluation_figures/ps_esti'+'nlocations_'+str(number_of_locations)+'_'+str(epoch)+'_'+str(SNR)+'.png')
    plt.close()

        

        # keep intermediate states iff backpropagation will be performed. If false, 
        # then all intermediate states will be thrown away during evaluation, to use
        # the least amount of memory possible.
            

def train_model(N, targets_dimension, directory, layer_sizes=[5000,500,500,500,500,500,500,500,500], SNR = 100, slope = 5 ,nlocations = 2, num_epochs=5000, save=True, log_transform = True, evaluation=False, path="model_data.pth.tar", used_model=1, used_loss='l1'):
    """Initalizes and the model for a fixed number of epochs, using dataloaders from the specified directory, 
    selected optimizer, scheduler, criterion, defualt otherwise. Features saving and restoration capabilities as well. 
    Adapted from the PyTorch tutorial found here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

        Args:
            num_classes (int): Number of classes in the data
            directory (str): Directory where the data is to be loaded from
            layer_sizes (list, optional): Number of blocks in each layer. Defaults to [2, 2, 2, 2], equivalent to ResNet18.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 45. 
            save (bool, optional): If true, the model will be saved to path. Defaults to True. 
            path (str, optional): The directory to load a model checkpoint from, and if save == True, save to. Defaults to "model_data.pth.tar".
    """
    #Prepare Data
    #training = DataBase(directotrairy,'psnet.mat','train',256*2)
    
    training = DataBase_pkl('./','train', SNR = SNR, slope = slope, number_of_locations = nlocations, log_transform = log_transform)
    train_dataloader = DataLoader(training,batch_size=128, shuffle=True)#, num_workers=4)
    validating = DataBase_pkl('./','val', SNR = SNR, slope = slope, number_of_locations = nlocations , log_transform = log_transform)
    val_dataloader = DataLoader(validating, batch_size=256)#, num_workers=4)
    testing = DataBase_pkl('./','test', SNR = SNR, number_of_locations = nlocations, log_transform = log_transform)
    test_dataloader = DataLoader(testing, batch_size=256)#, num_workers=4)
    log.l.info('num_train {} num_val {} num_test {}'.format(training.__len__(),validating.__len__(),testing.__len__()))
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    log.l.info('Dataset loaded!')


    # initalize the FullNet of this model
    
    number_of_locations = nlocations
    xs = training.xs
    xs2 = training.xs2
    T1 = training.T1
    T2 = training.T2
    
    if used_model == 1:
        model = MSResNet(input_channel=1,  num_classes=targets_dimension)
    elif used_model == 2:
        model = MSResNet2(input_channel=1,  num_classes=targets_dimension)
    elif used_model ==3:
        model = MSResNet3(input_channel=1, num_classes=targets_dimension)
    elif used_model ==4:
        model = ResNet(input_channel=1, num_classes=targets_dimension)
    elif used_model ==5:
        model = MSResNet4(input_channel=1, num_classes=targets_dimension)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    if used_loss == 'sumKLoss':
        criterion = networks.sumKLoss([], [0.1,0.1])
    elif used_loss == 'l1':
        criterion = torch.nn.L1Loss(reduction='sum')
    elif used_loss == 'mse':
        criterion = torch.nn.MSELoss(reduction='sum')
    elif used_loss == 'char':
        criterion = networks.L1_Charbonnier_loss(1e-6)
    criterion2 = networks.sumKLoss([], [0.1,0.1])
    
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.8)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=8, cooldown=10)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,1000], gamma=0.1)

    

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0
    epoch_loss = 0
    # check if there was a previously saved checkpoint
    '''
    if os.path.exists(path) and save:
        # loads the checkpoint
        checkpoint = torch.load(path)
        log.l.info("Reloading from previously saved checkpoint")

        # restores the model and optimizer state_dicts
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        
        # obtains the epoch the training is to resume from
        epoch_resume = checkpoint["epoch"]
    '''
    lowest_val_err = np.inf
    lowest_train_err = np.inf
    lowest_val_loss = np.inf
    for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", initial=epoch_resume, total=num_epochs):
        # each epoch has a training and validation step, in that order
        for phase in ['train', 'val']:

            # reset the running loss and corrects
            running_loss = 0.0
            running_acc = 0.0
            fn = 0.0
            err = 0.0
            err_T2 = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                #scheduler.step()# is to be called once every epoch during training
                scheduler.step(epoch_loss)
                model.train()
            else:
                model.eval()


            for inputs,spect,targets, targets_ps, signal_proxy ,_ in dataloaders[phase]:

                # move inputs and labels to the device the training is taking place on
                if N == 306 or N == 512:
                    inputs = torch.cat((inputs,spect), dim=1)
                elif N == 37355:
                    inputs = torch.cat((inputs,signal_proxy), dim=1)
                inputs = inputs.float()
                inputs = inputs.to(device)
                if targets_dimension == len(T1) or targets_dimension == len(T2):
                    targets = targets[:,:,0]
                elif targets_dimension == 4:
                    
                    #targets = targets.view(-1, 2*number_of_locations)
                    targets = torch.cat((targets[:,:,0],targets[:,:,1]),1)
                    
                elif targets_dimension <= 2:
                    targets = targets[:,:,0]
                    targets,_ = targets.sort()
    
                targets = targets.float()
                targets = targets.to(device)

                optimizer.zero_grad()

                # keep intermediate states iff backpropagation will be performed. If false, 
                # then all intermediate states will be thrown away during evaluation, to use
                # the least amount of memory possible.
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)

                    if targets_dimension == len(T1) or targets_dimension == len(T2):
                        targets_ps = targets_ps.float().to(device)
                        targets_hat = findFreq(outputs.cpu().detach().numpy(), targets, xs)
                        if epoch % 10 == 0:
                            plot_ps(outputs.cpu().detach().numpy()[0,:],targets_ps.data.cpu().numpy()[0,:], epoch, number_of_locations, SNR)

                        loss = criterion(outputs, targets_ps)
                    elif targets_dimension == 2:
                        #loss = torch.min(criterion(outputs, targets),criterion(outputs, targets[:,[1,0]]))
                        loss = criterion(outputs, targets) + 2.0*criterion2(outputs, targets)
                    else:

                        loss = criterion(outputs, targets) + 0.5*criterion2(outputs, targets)
                    # we're interested in the indices on the max values, not the values themselves
                    #_, preds = torch.max(outputs, 1)  
                    


                    # Backpropagate and optimize iff in training mode, else there's no intermediate
                    # values to backpropagate with and will throw an error.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   
                #if epoch % 10 == 0:
                #            print(targets_hat[0])
                #            print(targets[0])
                if epoch == 1000:
                    criterion2 = networks.sumKLoss([], [0.05,0.05])
                if epoch == 2000:
                    criterion2 = networks.sumKLoss([], [0.03,0.03])
                running_loss += loss.item() * inputs.size(0)
                if targets_dimension == len(T1) or targets_dimension == len(T2):
                    fn += networks.thresholdLoss(targets_hat, targets.cpu().numpy(), 0.1)
                    if log_transform:
                        err += relative_error(10**(targets.cpu().numpy()),10**(targets_hat))
                    else:
                        err += relative_error(targets.cpu().numpy(),targets_hat)
                elif targets_dimension == 4:
                    if log_transform:
                        #err += relative_error(10**(targets.cpu().numpy()[:,0::2]),10**(outputs.data.cpu().numpy()[:,0::2]))
                        #err_T2 += relative_error(10**(targets.cpu().numpy()[:,1::2]),10**(outputs.data.cpu().numpy()[:,1::2]))
                        GT_T1 = targets.cpu().numpy()[:,0:2] *(max(xs) - min(xs)) + min(xs)
                        esti_T1 = outputs.data.cpu().numpy()[:,0:2] *(max(xs) - min(xs)) + min(xs) 

                        GT_T2 = targets.cpu().numpy()[:,2:] *(max(xs2) - min(xs2)) + min(xs2)
                        esti_T2 = outputs.data.cpu().numpy()[:,2:] *(max(xs2) - min(xs2)) + min(xs2) 
                        err += relative_error(10**(GT_T1),10**(esti_T1))
                        err_T2 += relative_error(10**(GT_T2),10**(esti_T2))
                    else:
                        #err += relative_error(targets.cpu().numpy()[:,0::2],outputs.data.cpu().numpy()[:,0::2])
                        #err_T2 += relative_error(targets.cpu().numpy()[:,1::2],outputs.data.cpu().numpy()[:,1::2])
                        GT_T1 = targets.cpu().numpy()[:,0:2] *(max(xs) - min(xs)) + min(xs)
                        esti_T1 = outputs.data.cpu().numpy()[:,0:2] *(max(xs) - min(xs)) + min(xs) 

                        GT_T2 = targets.cpu().numpy()[:,2:] *(max(xs2) - min(xs2)) + min(xs2)
                        esti_T2 = outputs.data.cpu().numpy()[:,2:] *(max(xs2) - min(xs2)) + min(xs2) 
                        err += relative_error(GT_T1, esti_T1)
                        err_T2 += relative_error(GT_T2, esti_T2)
                elif targets_dimension <= 2:
                    if log_transform:
                        GT = targets.cpu().numpy() * (max(xs) - min(xs)) + min(xs)
                        esti = outputs.data.cpu().numpy() * (max(xs) - min(xs)) + min(xs)
                        err += relative_error(10**(GT), 10**(esti))
                    else:
                        GT = targets.cpu().numpy() * (max(xs) - min(xs)) + min(xs)
                        esti = outputs.data.cpu().numpy() * (max(xs) - min(xs)) + min(xs)
                        err += relative_error(GT, esti)
   
                
                #running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            
            
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))
            if targets_dimension == len(T1) or targets_dimension == len(T2):
                fn = fn / dataset_sizes[phase]
                err = err /dataset_sizes[phase]
                log.l.info("\n {} Loss: {} FN: {} Err: {}".format(phase, epoch_loss, fn, err))
            elif targets_dimension == 4:
                err = err /dataset_sizes[phase]
                err_T2 = err_T2 /dataset_sizes[phase]
                log.l.info("  {} Loss: {} T1_Err: {} T2_Err: {}".format(phase, epoch_loss, err, err_T2))
            elif targets_dimension <= 2:
                err = err /dataset_sizes[phase]
                log.l.info("\n {} Loss: {} Err: {}".format(phase, epoch_loss, err))
            if lowest_val_err > err and phase == 'val':
                lowest_val_err = err
            if lowest_val_loss > epoch_loss and phase == 'val':
                lowest_val_loss = epoch_loss
                if save:
                    torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'Err': err,
                    'Loss': epoch_loss,
                    'opt_dict': optimizer.state_dict(),
                    }, './saved_model/'+str(N)+'_'+str(targets_dimension)+'_model_'+str(used_model)+'_'+str(SNR)+'_'+path)
            if lowest_train_err > err and phase == 'train':
                lowest_train_err = err
        log.l.info("\n evaluation: {}".format(evaluation))
        if (epoch % 50 == 0) and evaluation and (phase == 'val'):
            test_output, test_GT = eval(test_dataloader, model, device, epoch, number_of_locations, SNR, used_model,xs,xs2,targets_dimension)
            np.save('./evaluation_figures/test_output_'+str(number_of_locations)+'_'+str(epoch)+'_'+str(SNR), test_output)
            np.save('./evaluation_figures/test_GT_'+str(number_of_locations)+'_'+str(epoch)+'_'+str(SNR),test_GT)


    # save the model if save=True
    

    # print the total time needed, HH:MM:SS format
    time_elapsed = time.time() - start    
    log.l.info("Training complete in {}h {}m {}s, lowest train/validation errors are {}/{}".format(time_elapsed//3600, (time_elapsed%3600)//60, time_elapsed %60,lowest_train_err,lowest_val_err))
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim",type=int,default=306)
    parser.add_argument("--output_dim",type=int,default=2)
    parser.add_argument("--num_locations",type=int,default=2)
    parser.add_argument("--SNR",type= float,default=100)
    parser.add_argument("--save",type=bool,default=False )
    parser.add_argument("--evaluation",type=bool,default=False )
    parser.add_argument("--log_transform",type=bool,default=True)
    parser.add_argument("--slope",type=float,default=50)
    parser.add_argument("--model",type=int,default=4)
    parser.add_argument("--loss",type=str,default='l1')
    parser.add_argument("--inner_dim", type=int, nargs='+', default=[5000,500,500,500,500,500,500,500,500])
    args = parser.parse_args()    
    train_model(args.input_dim, args.output_dim, '/Users/shengliu/Dropbox/phd/PSnet/mrf/', layer_sizes=args.inner_dim,
        nlocations = args.num_locations, slope = args.slope, SNR=args.SNR, save = args.save, log_transform = args.log_transform, 
        evaluation=args.evaluation, used_model = args.model, used_loss = args.loss)


