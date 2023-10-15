import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate, utils
from collections import OrderedDict
from DCLS.construct.modules import Dcls1d


class SRNN(nn.Module):
    def __init__(self, beta_lsm, beta_lif, learn_beta, threshold, learn_threshold, N, all_to_all):
        super().__init__()

        spike_grad = surrogate.atan()
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.beta_lsm = beta_lsm
        self.beta_lif = beta_lif
        self.threshold = threshold
        self.N = N
        self.all_to_all = all_to_all

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(34*34*2, N)
        self.lsm = snn.RLeaky(beta=self.beta_lsm, all_to_all=self.all_to_all, spike_grad=spike_grad, learn_beta=self.learn_beta,
                                    learn_threshold=self.learn_threshold, linear_features=self.N, threshold=self.threshold)
        self.fc2 = nn.Linear(N, 10)
        self.lif1 = snn.Leaky(beta=self.beta_lif, spike_grad=spike_grad, learn_beta=self.learn_beta, output=True)
            

    def forward(self, data):
        spk_rec = []
        mem_rec = []

        spk_lsm, syn_lsm, mem_lsm = self.lsm.init_rsynaptic()
        mem_out = self.lif1.init_leaky()

        # print(data.shape)
        for step in range(data.size(1)):  # data.size(1) = number of time stepsss
            in_curr = self.fc1(self.flatten(data[:,step,:,:,:]))
            spk_lsm, mem_lsm = self.lsm(in_curr, spk_lsm, mem_lsm) 
            out_curr = self.fc2(spk_lsm)
            spk_out, mem_out = self.lif1(out_curr, mem_out)
            #print(spk_out.shape)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
            

        return torch.stack(spk_rec), torch.stack(mem_rec)
    

class SNN_Delay(nn.Module):

    def __init__(self, beta, learn_beta, threshold, learn_threshold, time_steps):

        super().__init__()

        self.spike_grad = surrogate.atan()
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.beta = beta
        self.threshold = threshold
        self.time_steps = time_steps

        self.left_pad = 250//self.time_steps

        self.max_delay = 250//self.time_steps
        self.max_delay = self.max_delay if self.max_delay%2==1 else self.max_delay+1

        self.siginit = self.max_delay//2
        
        # self.net = nn.Sequential(
        #             nn.Flatten(),
        #             nn.Linear(34*34*2, 300),
        #             snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold),
        #             nn.Linear(300, 300),
        #             snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold),
        #             nn.Linear(300, 100),
        #             snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True, output=False, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold),
        #             )

        self.flatten = nn.Flatten()
        self.dcls1 = Dcls1d(in_channels=34*34*2, out_channels=300, kernel_count=1, stride=1, padding=0, dilated_kernel_size=self.max_delay, groups=1, bias=True, padding_mode='zeros', version='gauss')
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)
        self.dcls2 = Dcls1d(in_channels=300, out_channels=10, kernel_count=1, stride=1, padding=0, dilated_kernel_size=self.max_delay, groups=1, bias=True, padding_mode='zeros', version='gauss')
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)
        # self.fc1 = nn.Linear(34, 10)
        # self.fc2 = nn.Linear(34*34*2, 10)
        # self.net = nn.Sequential(
        #             nn.Flatten(),
        #             Dcls1d(in_channels=34*34*2, out_channels=300, kernel_count=1, stride=1, padding=0, dilated_kernel_size=1, groups=1, bias=True, padding_mode='zeros'),
        # )

        # set sig paramaters of the dcls1 and dcls2 to siginit and required_grad=False\
        
        nn.init.constant_(self.dcls1.SIG, self.siginit)
        nn.init.constant_(self.dcls2.SIG, self.siginit)
        # print(self.dcls1.SIG.requires_grad)
        self.dcls1.SIG.requires_grad = False
        self.dcls2.SIG.requires_grad = False

        

    def forward(self, data):

        mem_1 = self.lif1.init_leaky()
        mem_2 = self.lif2.init_leaky()

        # print(self.dcls1.SIG.requires_grad)
        # self.dcls1.SIG.requires_grad = False
        # self.dcls2.SIG.requires_grad = False

        spk_rec_1 = [] 

        print(data_1.size())

        # # left pad the time_steps with T time steps of zeros
        data_1 = F.pad(data_1, (0,0,self.left_pad,0), 'constant', 0)
        print(data_1.size())

        # permute to [batch_size, features, time_steps]
        x_1 = data_1.permute(0,2,1)
        print(x_1.size())

        x_1 = self.dcls1(x_1)
        print(x_1.size())

        for step in range(x_1.size(2)):
            spk_out, mem_1 = self.lif1(x_1[:,:,step], mem_1)
            spk_rec_1.append(spk_out)

        in_1 = torch.stack(spk_rec_1, dim=2)
        print(in_1.size())

        in_1 = F.pad(in_1, (self.left_pad,0), 'constant', 0)
        print(in_1.size())

        in_1 = self.dcls2(in_1)
        print(in_1.size())


        spk_rec_2 = []
        mem_rec_2 = []
        for step in range(in_1.size(2)):
            spk_out, mem_2 = self.lif2(in_1[:,:,step], mem_2)
            spk_rec_2.append(spk_out)
            mem_rec_2.append(mem_2)


        return torch.stack(spk_rec_2), torch.stack(mem_rec_2)
    

class SNN_Delay_2(nn.Module):

    def __init__(self, beta, learn_beta, threshold, learn_threshold, time_steps):

        super().__init__()

        self.spike_grad = surrogate.atan()
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.beta = beta
        self.threshold = threshold
        self.time_steps = time_steps

        self.left_pad = 250//self.time_steps

        self.max_delay = 250//self.time_steps
        self.max_delay = self.max_delay if self.max_delay%2==1 else self.max_delay+1

        self.siginit = self.max_delay//2
        
        # self.net = nn.Sequential(
        #             nn.Flatten(),
        #             nn.Linear(34*34*2, 300),
        #             snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold),
        #             nn.Linear(300, 300),
        #             snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold),
        #             nn.Linear(300, 100),
        #             snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True, output=False, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold),
        #             )

        self.flatten = nn.Flatten()
        self.dcls1 = Dcls1d(in_channels=140, out_channels=256, kernel_count=1, stride=1, padding=0, dilated_kernel_size=self.max_delay, groups=1, bias=True, padding_mode='zeros', version='gauss')
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)
        self.dcls2 = Dcls1d(in_channels=256, out_channels=256, kernel_count=1, stride=1, padding=0, dilated_kernel_size=self.max_delay, groups=1, bias=True, padding_mode='zeros', version='gauss')
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)
        self.dcls3 = Dcls1d(in_channels=256, out_channels=20, kernel_count=1, stride=1, padding=0, dilated_kernel_size=self.max_delay, groups=1, bias=True, padding_mode='zeros', version='gauss')
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)
        # self.fc1 = nn.Linear(34, 10)
        # self.fc2 = nn.Linear(34*34*2, 10)
        # self.net = nn.Sequential(
        #             nn.Flatten(),
        #             Dcls1d(in_channels=34*34*2, out_channels=300, kernel_count=1, stride=1, padding=0, dilated_kernel_size=1, groups=1, bias=True, padding_mode='zeros'),
        # )

        # set sig paramaters of the dcls1 and dcls2 to siginit and required_grad=False\

        for m in self.modules():
            if isinstance(m, Dcls1d):
                print(m)
                nn.init.constant_(m.SIG, self.siginit)
                m.SIG.requires_grad = False



        

    def forward(self, data):

        mem_1 = self.lif1.init_leaky()
        mem_2 = self.lif2.init_leaky()
        mem_3 = self.lif3.init_leaky()

        spk_rec_1 = [] 
        data_1 = data
        # data: [batch_size, time_steps, features]

        # # left pad the time_steps with T time steps of zeros
        data_1 = F.pad(data_1, (0,0,self.left_pad,0), 'constant', 0)
        #print(data_1.size())

        # permute to [batch_size, features, time_steps]
        x_1 = data_1.permute(0,2,1)
        #print(x_1.size())

        x_1 = self.dcls1(x_1)
        #print(x_1.size())

        for step in range(x_1.size(2)):
            spk_out, mem_1 = self.lif1(x_1[:,:,step], mem_1)
            spk_rec_1.append(spk_out)

        in_1 = torch.stack(spk_rec_1, dim=2)
        #print(in_1.size())

        in_1 = F.pad(in_1, (self.left_pad,0), 'constant', 0)
        #print(in_1.size())

        in_1 = self.dcls2(in_1)
        #print(in_1.size())

        spk_rec_2 = []

        for step in range(in_1.size(2)):
            spk_out, mem_2 = self.lif2(in_1[:,:,step], mem_2)
            spk_rec_2.append(spk_out)
        
        in_2 = torch.stack(spk_rec_2, dim=2)

        in_2 = F.pad(in_2, (self.left_pad,0), 'constant', 0)

        in_2 = self.dcls3(in_2)

        spk_rec_3 = []
        mem_3_rec = []
        for step in range(in_2.size(2)):
            spk_out, mem_3 = self.lif3(in_2[:,:,step], mem_3)
            spk_rec_3.append(spk_out)
            mem_3_rec.append(mem_3)

        return torch.stack(spk_rec_3), torch.stack(mem_3_rec)
    
        
    

    def reset_model(self, train=True):

        # We use clamp_parameters of the Dcls1d modules
        # self.dcls1.clamp_parameters()
        # self.dcls2.clamp_parameters()
        # self.dcls3.clamp_parameters()

        # automate to do this for all dcls layers

        for m in self.modules():
            if isinstance(m, Dcls1d):
                m.clamp_parameters()


    def decrease_sig(self, epoch, num_epochs, time_steps):

        # Decreasing to 0.23 instead of 0.5

        final_epoch = num_epochs//4
        alpha = 0

        max_delay = 250//time_steps
        max_delay = max_delay if max_delay%2==1 else max_delay+1

        siginit = max_delay//2

        # take sig from last dcls layer  
        sig = self.dcls3.SIG[0,0,0,0].detach().cpu().item()

        if epoch < final_epoch and sig > 0.23:

            alpha = (0.23/siginit)**(1/(final_epoch))

            # self.dcls3.SIG *= alpha
            # self.dcls2.SIG *= alpha
            # self.dcls1.SIG *= alpha

            # automate to do this for all dcls layers

            for m in self.modules():
                if isinstance(m, Dcls1d):
                    m.SIG *= alpha


    def round_pos(self):
        with torch.no_grad():
                
            # self.dcls1.P.round_()
            # self.dcls2.P.round_()
            # self.dcls3.P.round_()
            # self.dcls1.clamp_parameters()
            # self.dcls2.clamp_parameters()
            # self.dcls3.clamp_parameters()

            # automate to do this for all dcls layers

            for m in self.modules():
                if isinstance(m, Dcls1d):
                    m.P.round_()
                    m.clamp_parameters()