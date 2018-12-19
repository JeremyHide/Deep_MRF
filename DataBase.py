import numpy as np
import scipy.io as spio
import os
import random
import pickle
import time
import scipy.signal as ss
from torch.utils.data import DataLoader, Dataset
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
class DataBase(Dataset):
    def __init__(self, directory, filename, mode, number_of_example, number_of_locations = 2, SNR = 100, magnitude = [0.5, 0.5] ,npoints=502, train_indices_list = [], save_name = 'data_uniform'):
        directory = os.path.join(directory,filename)
        struct  = spio.loadmat(directory)
        self.dictionary = struct['dictionary']
        self.N = np.shape(self.dictionary)[0]
        self.D = np.shape(self.dictionary)[1]

        self.lookuptable = struct['lookuptable']
        self.T1 = np.unique(self.lookuptable[:,0])
        self.T2 = np.unique(self.lookuptable[:,1])
        self.lookuptable_class = self.lookuptable.copy()
        for i, value in enumerate(self.T1):
            self.lookuptable_class[self.lookuptable_class==value] = i
        for  i, value in enumerate(self.T2):
            self.lookuptable_class[self.lookuptable_class==value] = i

        self.number_of_locations = number_of_locations
        self.number_of_example = number_of_example
        self.SNR = SNR
        training_data = {}
        testing_data = {}
        if mode == 'train':
            self.signals, self.spectrum_list, self.targets, self.indices_list, self.targets_class = self.get_signal_samples(number_of_example, npoints)
            training_data['signals'] = self.signals
            training_data['targets'] = self.targets
            training_data['spectrum_list'] = self.spectrum_list
            training_data['indices_list'] = self.indices_list
            training_data['targets_class'] = self.targets_class
            if self.number_of_locations == 1:
                save_obj(training_data, 'training_'+save_name+'_single')
            elif self.number_of_locations == 2:
                save_obj(training_data, 'training_'+save_name)
            #np.save('./training_data', self.signals, self.spectrum_list, self.targets, self.indices_list)
            
        else:
            self.signals, self.spectrum_list, self.targets, self.indices_list, self.targets_class = self.get_signal_samples_test(number_of_example, npoints, train_indices_list)
            testing_data['signals'] = self.signals
            testing_data['targets'] = self.targets
            testing_data['spectrum_list'] = self.spectrum_list
            testing_data['indices_list'] = self.indices_list
            testing_data['targets_class'] = self.targets_class
            if self.number_of_locations == 1:
                save_obj(testing_data, 'testing_'+save_name+'_single')
            elif self.number_of_locations == 2:
                save_obj(testing_data, 'testing_'+save_name)
            #np.save('./testing_data', self.signals, self.spectrum_list, self.targets, self.indices_list)
        
    def __getitem__(self,index):
        if self.SNR:
            y = self.signals[index,:]
            rnd = np.random.randn(len(y));
            noise = np.linalg.norm(y)/self.SNR*rnd/np.linalg.norm(rnd)
            y = y + noise
            return np.array(y), np.array(self.spectrum_list[index]), np.array(self.targets[index])
        else:
            return np.array(self.signals[index,:]), np.array(self.spectrum_list[index]), np.array(self.targets[index])

    def __len__(self):
        return self.number_of_example



    def get_signal(self, indices,magnitude):
        y = np.sum(self.dictionary[:,indices]*magnitude,1)
        return y
    def get_one_signal_sample(self,i,D,samples,targets,targets_class,spectrum_list,npoints):
        if i % 1000 == 0:
            print("generated {} examples".format(i))
        #while True:
        #    indices = np.random.choice(range(D), self.number_of_locations, replace = False)
        #    if tuple(indices) not in indices_list:
        #        break
        indices = np.random.choice(range(D), self.number_of_locations, replace = True)      ### Number of compartment....              
        indices = np.sort(np.unique(indices))
        if len(indices) == 2:
            c = np.random.uniform(0.0,1.0)
            magnitude = [c, 1-c]

        elif len(indices) == 1:
            magnitude = 1
        else:
            magnitude = np.random.uniform(0,1,len(indices))
            magnitude = magnitude/sum(magnitude)

        
        samples[i,:] = self.get_signal(indices,magnitude)
        targets.append(self.get_2D(indices))
        targets_class.append(self.get_target_class(indices))
        spectrum_list.append(self.get_spectrum(samples[i,:],npoints))
        return samples,targets,targets_class,spectrum_list

    def get_signal_samples(self, number_of_example,  npoints):
        #Parallel 
        
        N = self.N
        D = self.D
        samples = np.zeros((number_of_example, N))
        #samples = []
        targets = []
        targets_class = []
        spectrum_list = []
        indices_list = []
        for i in range(number_of_example):
            samples,targets,targets_class,spectrum_list = self.get_one_signal_sample(i,D,samples,targets,targets_class,spectrum_list,npoints)
        return samples, spectrum_list, targets, indices_list,targets_class

    def get_signal_samples_test(self, number_of_example, npoints, train_indices_list):
        N = self.N
        D = self.D
        samples = np.zeros((number_of_example, N))
        
        targets = []
        targets_class = []
        spectrum_list = []
        indices_list = []
        for i in range(number_of_example):
            #while True:
            #    indices = np.random.choice(range(D), self.number_of_locations, replace = False)
            #    if (tuple(indices) not in indices_list) & (tuple(indices) not in train_indices_list):
            #        break
            indices = np.random.choice(range(D), self.number_of_locations, replace = False)        
            indices = np.sort(indices)
            indices_list.append(tuple(indices))
            if len(indices) == 2:
                c = np.random.uniform(0.0,1.0)
                magnitude = [c, 1-c]
            elif len(indices) == 1:
                magnitude = 1
            else:
                magnitude = np.random.uniform(0,1,len(indices))
            samples[i,:] = self.get_signal(indices,magnitude)
            targets.append(self.get_2D(indices))
            targets_class.append(self.get_target_class(indices))
            spectrum_list.append(self.get_spectrum(samples[i,:],npoints))
        return samples, spectrum_list, targets, indices_list,targets_class


    def get_spectrum(self,signal,npoints):
        fft = np.fft.fft(signal, norm='ortho', n=(npoints))
        spectrum = fft.real**2+fft.imag**2
        return spectrum


    def get_2D(self,indices):

        list_Ts = self.lookuptable[indices,:]
        return [tuple(item) for item in list_Ts]
    
    def get_indices_list(self):

        return self.indices_list
    def get_target_class(self,indices):
        list_Ts = self.lookuptable_class[indices,:]
        return [tuple(item) for item in list_Ts]



class DataBase_pkl(Dataset):
    def __init__(self, directory, mode, d = 1, SNR = 100, slope = 5, number_of_locations = 2, log_transform = True):
        struct  = spio.loadmat('./psnet3.mat')
        self.lookuptable = struct['lookuptable']
        self.dictionary = struct['dictionary']
        self.SNR = SNR
        self.T1 = np.unique(self.lookuptable[:,0])
        self.T2 = np.unique(self.lookuptable[:,1])
        if d == 1:
            min_Td = np.min(self.T1)
            max_Td = np.max(self.T1)
        else:
            min_Td = np.min(self.T2)
            max_Td = np.max(self.T2)
        #self.xs = np.logspace(np.log10(min_Td),np.log10(max_Td),npoints)
        if log_transform: 
            self.xs = np.log10(self.T1)#np.linspace(min_Td, max_Td,npoints)
            self.xs2 = np.log10(self.T2)
        else:
            self.xs = self.T1
            self.xs2 = self.T2
        if mode == 'train':
            if number_of_locations == 1:
                training_dictionary = load_obj(os.path.join(directory,'training_data_uniform_single'))
            elif number_of_locations == 2:
                training_dictionary = load_obj(os.path.join(directory,'training_data_uniform'))
            else:
                print('Other number of locations is not supported yet')
            self.signals, self.spectrum_list, self.targets = training_dictionary['signals'], training_dictionary['spectrum_list'], training_dictionary['targets']
            if log_transform: 
                self.targets = np.log10(self.targets)
            self.targets_ps = self.creat_target_ps_1d(self.xs, np.array(self.targets), d, slope)
            self.signal_proxy = self.creat_signal_proxy(self.signals, self.dictionary)
            self.targets_class = training_dictionary['targets_class']
        elif mode == 'val':
            
            if number_of_locations == 1:
                testing_dictionary = load_obj(os.path.join(directory,'testing_data_uniform_single'))
            elif number_of_locations == 2:
                testing_dictionary = load_obj(os.path.join(directory,'testing_data_uniform'))
            else:
                print('Other number of locations is not supported yet')
            n_example_val = int(0.7*(len(testing_dictionary['signals'])))
            self.signals, self.spectrum_list, self.targets = testing_dictionary['signals'][0:n_example_val], testing_dictionary['spectrum_list'][0:n_example_val], testing_dictionary['targets'][0:n_example_val]
            if log_transform: 
                self.targets = np.log10(self.targets)

            self.targets_ps = self.creat_target_ps_1d(self.xs, np.array(self.targets), d, slope)
            self.signal_proxy = self.creat_signal_proxy(self.signals, self.dictionary)
            self.targets_class = testing_dictionary['targets_class'][0:n_example_val]
        elif mode == 'test':
            if number_of_locations == 1:
                testing_dictionary = load_obj(os.path.join(directory,'testing_data_uniform_single'))
            elif number_of_locations == 2:
                testing_dictionary = load_obj(os.path.join(directory,'testing_data_uniform'))
            else:
                print('Other number of locations is not supported yet')
            n_example_val = int(0.7*(len(testing_dictionary['signals'])))
            self.signals, self.spectrum_list, self.targets = testing_dictionary['signals'][n_example_val:], testing_dictionary['spectrum_list'][n_example_val:], testing_dictionary['targets'][n_example_val:]
            if log_transform: 
                self.targets = np.log10(self.targets)
            self.targets_ps = self.creat_target_ps_1d(self.xs, np.array(self.targets), d, slope)
            self.signal_proxy = self.creat_signal_proxy(self.signals, self.dictionary)
            self.targets_class = testing_dictionary['targets_class'][n_example_val:]

    def __getitem__(self,index):
        if self.SNR:
            y = self.signals[index,:]
            sigma = np.random.uniform(0,1)
            rnd = np.random.randn(len(y))*sigma
            noise = np.linalg.norm(y)/self.SNR*rnd/np.linalg.norm(rnd)
            y = y + noise
            return np.array(y), np.array(self.spectrum_list[index]), (np.array(self.targets[index]) - np.array([min(self.xs),min(self.xs2)]))/np.array(np.array([max(self.xs) - min(self.xs),max(self.xs2)-min(self.xs2)])), self.targets_ps[index], self.signal_proxy[index],np.array(self.targets_class[index])#self.creat_signal_proxy_single(y, self.dictionary)
        else:
            return np.array(self.signals[index,:]), np.array(self.spectrum_list[index]), (np.array(self.targets[index]) - np.array([min(self.xs),min(self.xs2)]))/np.array(np.array([max(self.xs) - min(self.xs),max(self.xs2)-min(self.xs2)])), self.targets_ps[index], self.signal_proxy[index],np.array(self.targets_class[index])#self.creat_signal_proxy_single(self.signals[index], self.dictionary)

    def __len__(self):
        return len(self.signals)

    def triangle_mrf_1d(self, f, xs, slope):
        n = f.shape[0]
        npoints = len(xs)
        ps = np.zeros((n, npoints))
        for i in range(f.shape[1]):
            ps += np.clip(1 - slope * np.abs(np.repeat(xs.reshape(1, npoints), n, 0) - f[:, i].reshape(n, 1)), 0, None)
        return ps

    def creat_target_ps_1d(self,xs, targets, d, slope):
        n_points = len(xs)
        
        targets_ps = np.zeros((targets.shape[0],n_points))
        for i in range(targets.shape[0]):
            Td = []
            for j in range(targets.shape[1]):
                Td.append(targets[i][j][d-1])
            targets_ps[i,:] = self.triangle_mrf_1d(np.array([Td]), xs, slope)
        return targets_ps

    def creat_signal_proxy(self, signals, dictionary):
        number_of_example = len(signals)
        proxy_len = dictionary.shape[1]
        signal_proxy = np.zeros((number_of_example, proxy_len))
        for i in range(number_of_example):
            signal_proxy[i,:] = np.dot(dictionary.T,signals[i,:])
        return signal_proxy

    def creat_signal_proxy_single(self, signal, dictionary):
        proxy_len = dictionary.shape[1]
        #signal_proxy = np.zeros((1, proxy_len))
        signal_proxy = np.dot(dictionary.T,signal)
        return signal_proxy



class DataBase_pkl_2D(Dataset):
    def __init__(self, directory, mode, d = 1, SNR = 100, slope = 5, number_of_locations = 2, log_transform = True, load_name = 'data_uniform' ):
        struct  = spio.loadmat('./psnet3.mat')
        self.lookuptable = struct['lookuptable']
        self.dictionary = struct['dictionary']
        self.SNR = SNR
        self.T1 = np.unique(self.lookuptable[:,0])
        self.T2 = np.unique(self.lookuptable[:,1])
        #self.kernel =  np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])#*0.0625
        #self.kernel =  np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])/256.0
        self.kernel =  np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]])*0.0625
        if d == 1:
            min_Td = np.min(self.T1)
            max_Td = np.max(self.T1)
        else:
            min_Td = np.min(self.T2)
            max_Td = np.max(self.T2)
        #self.xs = np.logspace(np.log10(min_Td),np.log10(max_Td),npoints)
        if log_transform: 
            self.xs = np.log10(self.T1)#np.linspace(min_Td, max_Td,npoints)
        else:
            self.xs = self.T1
        if mode == 'train':
            if number_of_locations == 1:
                training_dictionary = load_obj(os.path.join(directory,'training_'+load_name+'_single'))
            elif number_of_locations == 2:
                training_dictionary = load_obj(os.path.join(directory,'training_'+load_name))
            else:
                print('Other number of locations is not supported yet')
            self.signals, self.spectrum_list, self.targets = training_dictionary['signals'], training_dictionary['spectrum_list'], training_dictionary['targets']
            if log_transform: 
                self.targets = np.log10(self.targets)
            self.targets_ps = self.creat_target_ps_1d(self.xs, np.array(self.targets), d, slope)
            self.signal_proxy = self.creat_signal_proxy(self.signals, self.dictionary)
            self.targets_class = training_dictionary['targets_class']
            self.targets_ps =  self.creat_target_ps_2d(self.targets_class, number_of_locations)
        elif mode == 'val':
            
            if number_of_locations == 1:
                testing_dictionary = load_obj(os.path.join(directory,'testing_'+load_name+'_single'))
            elif number_of_locations == 2:
                testing_dictionary = load_obj(os.path.join(directory,'testing_'+load_name))
            else:
                print('Other number of locations is not supported yet')
            n_example_val = int(0.7*(len(testing_dictionary['signals'])))
            self.signals, self.spectrum_list, self.targets = testing_dictionary['signals'][0:n_example_val], testing_dictionary['spectrum_list'][0:n_example_val], testing_dictionary['targets'][0:n_example_val]
            if log_transform: 
                self.targets = np.log10(self.targets)
            self.signal_proxy = self.creat_signal_proxy(self.signals, self.dictionary)
            self.targets_class = testing_dictionary['targets_class'][0:n_example_val]
            self.targets_ps =  self.creat_target_ps_2d(self.targets_class,number_of_locations)
        elif mode == 'test':
            if number_of_locations == 1:
                testing_dictionary = load_obj(os.path.join(directory,'testing_'+load_name+'_single'))
            elif number_of_locations == 2:
                testing_dictionary = load_obj(os.path.join(directory,'testing_'+load_name))
            else:
                print('Other number of locations is not supported yet')
            n_example_val = int(0.7*(len(testing_dictionary['signals'])))
            self.signals, self.spectrum_list, self.targets = testing_dictionary['signals'][n_example_val:], testing_dictionary['spectrum_list'][n_example_val:], testing_dictionary['targets'][n_example_val:]
            if log_transform: 
                self.targets = np.log10(self.targets)
            
            self.signal_proxy = self.creat_signal_proxy(self.signals, self.dictionary)
            self.targets_class = testing_dictionary['targets_class'][n_example_val:]
            self.targets_ps =  self.creat_target_ps_2d( self.targets_class,number_of_locations)


    def __getitem__(self,index):
        if self.SNR:
            y = self.signals[index,:]
            sigma = np.random.uniform(0,1)
            rnd = np.random.randn(len(y))*sigma
            noise = np.linalg.norm(y)/self.SNR*rnd/np.linalg.norm(rnd)
            y = y + noise
            return np.array(y), np.array(self.spectrum_list[index]), np.array(self.targets[index]), self.targets_ps[index], self.signal_proxy[index],self.targets_class[index],(np.array(self.targets_class[index])*2+1)/np.array([len(self.T1),len(self.T2)]) - 1#self.creat_signal_proxy_single(y, self.dictionary)
        else:
            return np.array(self.signals[index,:]), np.array(self.spectrum_list[index]), np.array(self.targets[index]), self.targets_ps[index], self.signal_proxy[index],self.targets_class[index],(np.array(self.targets_class[index])*2+1)/np.array([len(self.T1),len(self.T2)]) - 1#self.creat_signal_proxy_single(self.signals[index], self.dictionary)

    def __len__(self):
        return len(self.signals)

    def triangle_mrf_1d(self, f, xs, slope):
        n = f.shape[0]
        npoints = len(xs)
        ps = np.zeros((n, npoints))
        for i in range(f.shape[1]):
            ps += np.clip(1 - slope * np.abs(np.repeat(xs.reshape(1, npoints), n, 0) - f[:, i].reshape(n, 1)), 0, None)
        return ps

    def creat_target_ps_1d(self,xs, targets, d, slope):
        n_points = len(xs)
        
        targets_ps = np.zeros((targets.shape[0],n_points))
        for i in range(targets.shape[0]):
            Td = []
            for j in range(targets.shape[1]):
                Td.append(targets[i][j][d-1])
            targets_ps[i,:] = self.triangle_mrf_1d(np.array([Td]), xs, slope)
        return targets_ps

    def creat_signal_proxy(self, signals, dictionary):
        number_of_example = len(signals)
        proxy_len = dictionary.shape[1]
        signal_proxy = np.zeros((number_of_example, len(self.T1), len(self.T2)))
        for i in range(number_of_example):
            signal_proxy[i,:,:] = np.dot(dictionary.T, signals[i,:]).reshape(len(self.T1),-1)
        return signal_proxy

    def creat_signal_proxy_single(self, signal, dictionary):
        proxy_len = dictionary.shape[1]
        #signal_proxy = np.zeros((1, proxy_len))
        signal_proxy = np.dot(dictionary.T,signal)
        return signal_proxy
    def assign_values(self, values, lookuptable):
        T1 = np.unique(lookuptable[:,0])
        T2 = np.unique(lookuptable[:,1])
        T_space = np.array(np.meshgrid(T1, T2)).T.reshape(-1,2)
        T_space = [tuple(T_space[i,:]) for i in range(len(T_space))]
        T_space_dict = {}

        for ids in T_space:
            T_space_dict[ids] = np.nan
        lookuptable_ = [tuple(lookuptable[i,:]) for i in range(len(lookuptable))]
        lookuptable_dict = {}
        for i in range(len(lookuptable_)):
            ids = lookuptable_[i]
            lookuptable_dict[ids] = values[i]
        T_space_dict_new = {**T_space_dict, **lookuptable_dict} 
        # Transform from dict back to array
        dictlist = [] 
        for value in T_space_dict_new.values():
            dictlist.append(value)
        return np.array(dictlist)

    def creat_target_ps_2d(self, targets_class,number_of_locations):
        N = len(targets_class)
        targets_ps = np.zeros((N,number_of_locations,len(self.T1),len(self.T2)))
        for i in range(N):
            for j in range(number_of_locations):
                mat = np.zeros((len(self.T1),len(self.T2)))
                mat[int(targets_class[i][j][0]),int(targets_class[i][j][1])] = 0.5
                conv_signal = ss.convolve2d(mat,self.kernel,mode='same')
                #conv_signal2 = ss.convolve2d(mat2,self.kernel,mode='same')
                targets_ps[i,j,:,:] = conv_signal/np.sum(conv_signal)
                #targets_ps[i,1,:,:] = conv_signal2/np.sum(conv_signal2)
        return targets_ps


