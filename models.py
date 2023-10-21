import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate, utils
from collections import OrderedDict
from DCLS.construct.modules import *
    

class SNN(nn.Module):

    def __init__(self, beta, learn_beta, threshold, learn_threshold, time_steps):

        super().__init__()

        self.spike_grad = surrogate.atan()
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.beta = beta
        self.threshold = threshold
        self.time_steps = time_steps

        self.net = nn.Sequential(
                    nn.Linear(140, 256),
                    snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True, learn_beta=self.learn_beta),
                    nn.Linear(256, 256),
                    snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True, learn_beta=self.learn_beta),
                    nn.Linear(256, 20),
                    snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True, learn_beta=self.learn_beta, output=True)
                    )
        
    def forward(self, data):

        spk_rec = []
        mem_rec = []
        utils.reset(self.net)  # resets hidden states for all LIF neurons in net

        #print(data.size(1))
        for step in range(data.size(1)):  # data.size(1) = number of time steps
            spk_out, mem_out = self.net(data[:,step,:])
            #spk_out, mem_out = self.net(data[step])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)


class SNN_Delay(nn.Module):

    def __init__(self, beta, learn_beta, threshold, learn_threshold, time_step):

        super().__init__()

        self.spike_grad = surrogate.atan()
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.beta = beta
        self.threshold = threshold
        self.time_step = time_step

        self.left_pad = 250//self.time_step

        self.max_delay = 250//self.time_step
        self.max_delay = self.max_delay if self.max_delay%2==1 else self.max_delay+1

        self.siginit = self.max_delay//2

        self.flatten = nn.Flatten()
        self.dcls1 = Dcls1d(in_channels=140, out_channels=256, kernel_count=1, stride=1, padding=0, dilated_kernel_size=self.max_delay, groups=1, bias=True, padding_mode='zeros', version='gauss')
        self.dropout1 = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(256)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)
        self.dcls2 = Dcls1d(in_channels=256, out_channels=256, kernel_count=1, stride=1, padding=0, dilated_kernel_size=self.max_delay, groups=1, bias=True, padding_mode='zeros', version='gauss')
        self.dropout2 = nn.Dropout(p=0.4)
        self.bn2 = nn.BatchNorm1d(256)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)
        self.dcls3 = Dcls1d(in_channels=256, out_channels=20, kernel_count=1, stride=1, padding=0, dilated_kernel_size=self.max_delay, groups=1, bias=True, padding_mode='zeros', version='gauss')
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)

        # set sig parameters of the dcls layers to siginit and required_grad=False

        for m in self.modules():
            if isinstance(m, Dcls1d):
                nn.init.uniform_(m.P, a=-self.max_delay//2, b=self.max_delay//2)
                m.clamp_parameters()

        for m in self.modules():
            if isinstance(m, Dcls1d):
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

        x_1 = self.bn1(x_1)
        #print(x_1.size())

        for step in range(x_1.size(2)):
            spk_out, mem_1 = self.lif1(x_1[:,:,step], mem_1)
            spk_rec_1.append(spk_out)

        in_1 = torch.stack(spk_rec_1, dim=2)

        in_1 = self.dropout1(in_1)
        #print(in_1.size())

        in_1 = F.pad(in_1, (self.left_pad,0), 'constant', 0)
        #print(in_1.size())

        in_1 = self.dcls2(in_1)
        #print(in_1.size())

        in_1 = self.bn2(in_1)

        spk_rec_2 = []

        for step in range(in_1.size(2)):
            spk_out, mem_2 = self.lif2(in_1[:,:,step], mem_2)
            spk_rec_2.append(spk_out)
        
        in_2 = torch.stack(spk_rec_2, dim=2)

        in_2 = self.dropout2(in_2)

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

        # clamp parameters of dcls layers

        for m in self.modules():
            if isinstance(m, Dcls1d):
                m.clamp_parameters()


    def decrease_sig(self, epoch, num_epochs, time_step):

        # Decreasing to 0.23
        final_epoch = num_epochs//4
        alpha = 0

        max_delay = 250//time_step
        max_delay = max_delay if max_delay%2==1 else max_delay+1

        siginit = max_delay//2

        # take sig from last dcls layer  
        sig = self.dcls3.SIG[0,0,0,0].detach().cpu().item()

        if epoch < final_epoch and sig > 0.23:

            alpha = (0.23/siginit)**(1/(final_epoch))

            for m in self.modules():
                if isinstance(m, Dcls1d):
                    m.SIG *= alpha


    def round_pos(self):
        with torch.no_grad():

            for m in self.modules():
                if isinstance(m, Dcls1d):
                    m.P.round_()
                    m.clamp_parameters()


class SNN_Delay_Conv(nn.Module):

    def __init__(self, beta, learn_beta, threshold, learn_threshold, time_steps, surr):

        super().__init__()

        if surr == 'atan':
            self.spike_grad = surrogate.atan()
        elif surr == 'fast_sigmoid':
            self.spike_grad = surrogate.fast_sigmoid()

        # initialize the parameters
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.beta = beta
        self.threshold = threshold
        self.time_steps = time_steps

        # initialize the dcls parameters
        self.left_pad = 250//self.time_steps
        self.max_delay = 250//self.time_steps
        self.max_delay = self.max_delay if self.max_delay%2==1 else self.max_delay+1
        self.siginit = self.max_delay//2

        # define the network
        self.dcls3_1d_1 = Dcls3_1d(in_channels=2, out_channels=12, kernel_count=1, stride=1, version='gauss',
                                   padding=0, dilated_kernel_size=self.max_delay, dense_kernel_size=(5,5))
        self.mp_1 = nn.MaxPool2d(2)
        self.ln_1 = nn.LayerNorm((12, 15, 15), elementwise_affine=False)
        self.lif_1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)
        self.dropout_1 = nn.Dropout(p=0.4)
        self.dcls3_1d_2 = Dcls3_1d(in_channels=12, out_channels=32, kernel_count=1, stride=1, version='gauss',
                                    padding=0, dilated_kernel_size=self.max_delay, dense_kernel_size=(5,5))
        self.mp_2 = nn.MaxPool2d(2)
        self.ln_2 = nn.LayerNorm((32, 5, 5), elementwise_affine=False)
        self.lif_2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)
        self.dropout_2 = nn.Dropout(p=0.4)
        self.flatten = nn.Flatten(start_dim=1, end_dim=3)
        self.dcls1d_1 = Dcls1d(in_channels=32*5*5, out_channels=100, kernel_count=1, stride=1, padding=0, 
                               dilated_kernel_size=self.max_delay, version='gauss')
        self.lif_3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)
        self.dcls1d_2 = Dcls1d(in_channels=100, out_channels=10, kernel_count=1, stride=1, padding=0, 
                               dilated_kernel_size=self.max_delay, version='gauss')
        self.lif_4 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)

        # set P parameters of the dcls1d layers to random values between -max_delay//2 and max_delay//2
        for m in self.modules():
            if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                nn.init.uniform_(m.P, a=-self.max_delay//2, b=self.max_delay//2)
                m.clamp_parameters()


        # set sig parameters of the dcls layers to siginit and required_grad=False
        for m in self.modules():
            if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                print(m)
                nn.init.constant_(m.SIG, self.siginit)
                m.SIG.requires_grad = False


    def forward(self, data):

        # automate the code so as to pass through dcls2_1 to lif_3

        mem_1 = self.lif_1.init_leaky()
        mem_2 = self.lif_2.init_leaky()
        mem_3 = self.lif_3.init_leaky()
        mem_4 = self.lif_4.init_leaky()

        # data [batch_size, time_steps, C, H, W] to [batch_size, C, H, W, time_steps]
        data = data.permute(0,2,3,4,1)

        # left pad the time_steps with T time steps of zeros
        x_1 = F.pad(data, (self.left_pad,0), 'constant', 0)
        #print(x_1.size())

        x_1 = self.dcls3_1d_1(x_1)
        #print(x_1.size())

        spk_rec_1 = []

        for step in range(x_1.size(4)):
            in_1 = self.ln_1(self.mp_1(x_1[:,:,:,:,step]))
            spk_out, mem_1 = self.lif_1(in_1, mem_1)
            spk_rec_1.append(spk_out)

        x_2 = torch.stack(spk_rec_1, dim=4)
        #print(x_2.size())

        x_2 = F.pad(x_2, (self.left_pad,0), 'constant', 0)
        #print(x_2.size())

        x_2 = self.dcls3_1d_2(x_2)
        #print(x_2.size())

        spk_rec_2 = []

        for step in range(x_2.size(4)):
            in_2 = self.ln_2(self.mp_2(x_2[:,:,:,:,step]))
            spk_out, mem_2 = self.lif_2(in_2, mem_2)
            spk_rec_2.append(spk_out)

        x_3 = torch.stack(spk_rec_2, dim=4)
        #print(x_3.size())

        x_3 = F.pad(x_3, (self.left_pad,0), 'constant', 0)

        # now change x_3 from [batch_size, C, H, W, time_steps] to [batch_size, C*H*W, time_steps]

        x_3 = self.flatten(x_3)

        x_3 = self.dcls1d_1(x_3)
        #print(x_3.size())

        spk_rec_3 = []
        mem_rec_3 = []

        for step in range(x_3.size(2)):
            spk_out, mem_3 = self.lif_3(x_3[:,:,step], mem_3)
            spk_rec_3.append(spk_out)
            mem_rec_3.append(mem_3)

        x_4 = torch.stack(spk_rec_3, dim=2)
        #print(x_4.size())
              
        x_4 = F.pad(x_4, (self.left_pad,0), 'constant', 0)

        x_4 = self.dcls1d_2(x_4)
        #print(x_4.size())

        spk_rec_4 = []
        mem_rec_4 = []

        for step in range(x_4.size(2)):
            spk_out, mem_4 = self.lif_4(x_4[:,:,step], mem_4)
            spk_rec_4.append(spk_out)
            mem_rec_4.append(mem_4)

        #print(torch.stack(spk_rec_4).size())

        #exit()

        return torch.stack(spk_rec_4), torch.stack(mem_rec_4)

        #output of the form [time_steps, batch_size, features]


    def reset_model(self, train):

        # clamp parameters of dcls layers

        for m in self.modules():
            if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                m.clamp_parameters()


    def decrease_sig(self, epoch, num_epochs, time_steps):

        # Decreasing to 0.23
        final_epoch = num_epochs//4
        alpha = 0

        max_delay = 250//time_steps
        max_delay = max_delay if max_delay%2==1 else max_delay+1

        siginit = max_delay//2

        # take sig from last dcls layer  
        sig = self.dcls1d_2.SIG[0,0,0,0].detach().cpu().item()

        if epoch < final_epoch and sig > 0.23:

            alpha = (0.23/siginit)**(1/(final_epoch))

            for m in self.modules():
                if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                    m.SIG *= alpha


    def round_pos(self):
        with torch.no_grad():

            for m in self.modules():
                if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                    m.P.round_()
                    m.clamp_parameters()


class SNN_Delay_Conv_Small(nn.Module):

    def __init__(self, beta, learn_beta, threshold, learn_threshold, time_steps, surr):

        super().__init__()

        if surr == 'atan':
            self.spike_grad = surrogate.atan()
        elif surr == 'fast_sigmoid':
            self.spike_grad = surrogate.fast_sigmoid()

        # initialize the parameters
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.beta = beta
        self.threshold = threshold
        self.time_steps = time_steps

        # initialize the dcls parameters
        # self.left_pad = 250//self.time_steps
        # self.max_delay = 250//self.time_steps

        self.left_pad = self.time_steps
        self.max_delay = self.time_steps

        # self.left_pad = 20
        # self.max_delay = 20

        self.max_delay = self.max_delay if self.max_delay%2==1 else self.max_delay+1
        self.siginit = self.max_delay//2

        # define the network
        self.dcls3_1d_1 = Dcls3_1d(in_channels=2, out_channels=12, kernel_count=1, stride=1, version='gauss',
                                   padding=0, dilated_kernel_size=self.max_delay, dense_kernel_size=(5,5))
        self.mp_1 = nn.MaxPool2d(2)
        self.bn_1 = nn.BatchNorm2d(12)
        self.lif_1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, 
                               threshold=self.threshold, learn_threshold=self.learn_threshold)
        self.dcls3_1d_2 = Dcls3_1d(in_channels=12, out_channels=32, kernel_count=1, stride=1, version='gauss',
                                    padding=0, dilated_kernel_size=self.max_delay, dense_kernel_size=(5,5))
        self.mp_2 = nn.MaxPool2d(2)
        self.bn_2 = nn.BatchNorm2d(32)
        self.lif_2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, 
                               threshold=self.threshold, learn_threshold=self.learn_threshold)
        self.flatten = nn.Flatten(start_dim=1, end_dim=3)
        self.dcls1d_1 = Dcls1d(in_channels=32*5*5, out_channels=100, kernel_count=1, stride=1, padding=0, 
                               dilated_kernel_size=self.max_delay, version='gauss')
        self.bn_3 = nn.BatchNorm1d(100)
        self.lif_3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, 
                               threshold=self.threshold, learn_threshold=self.learn_threshold)
        self.dcls1d_2 = Dcls1d(in_channels=100, out_channels=10, kernel_count=1, stride=1, padding=0, 
                               dilated_kernel_size=self.max_delay, version='gauss')
        self.lif_4 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, learn_beta=self.learn_beta, 
                               threshold=self.threshold, learn_threshold=self.learn_threshold)

        # set P parameters of the dcls1d layers to random values between -max_delay//2 and max_delay//2
        for m in self.modules():
            if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                nn.init.uniform_(m.P, a=-self.max_delay//2, b=self.max_delay//2)
                m.clamp_parameters()


        # set sig parameters of the dcls layers to siginit and required_grad=False
        for m in self.modules():
            if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                print(m)
                nn.init.constant_(m.SIG, self.siginit)
                m.SIG.requires_grad = False


    def forward(self, data):

        # automate the code so as to pass through dcls2_1 to lif_3

        mem_1 = self.lif_1.init_leaky()
        mem_2 = self.lif_2.init_leaky()
        mem_3 = self.lif_3.init_leaky()
        mem_4 = self.lif_4.init_leaky()

        # data [batch_size, time_steps, C, H, W] to [batch_size, C, H, W, time_steps]
        data = data.permute(0,2,3,4,1)

        # left pad the time_steps with T time steps of zeros
        x_1 = F.pad(data, (self.left_pad,0), 'constant', 0)
        #print(x_1.size())

        x_1 = self.dcls3_1d_1(x_1)
        #print(x_1.size())

        spk_rec_1 = []

        for step in range(x_1.size(4)):
            in_1 = self.bn_1(self.mp_1(x_1[:,:,:,:,step]))
            spk_out, mem_1 = self.lif_1(in_1, mem_1)
            spk_rec_1.append(spk_out)

        x_2 = torch.stack(spk_rec_1, dim=4)
        #print(x_2.size())

        x_2 = F.pad(x_2, (self.left_pad,0), 'constant', 0)
        #print(x_2.size())

        x_2 = self.dcls3_1d_2(x_2)
        #print(x_2.size())

        spk_rec_2 = []

        for step in range(x_2.size(4)):
            in_2 = self.bn_2(self.mp_2(x_2[:,:,:,:,step]))
            spk_out, mem_2 = self.lif_2(in_2, mem_2)
            spk_rec_2.append(spk_out)

        x_3 = torch.stack(spk_rec_2, dim=4)
        #print(x_3.size())

        x_3 = F.pad(x_3, (self.left_pad,0), 'constant', 0)

        # now change x_3 from [batch_size, C, H, W, time_steps] to [batch_size, C*H*W, time_steps]

        #print(x_3.size())
        x_3 = self.flatten(x_3)
        #print(x_3.size())

        x_3 = self.dcls1d_1(x_3)
        #print(x_3.size())

        spk_rec_3 = []
        mem_rec_3 = []

        for step in range(x_3.size(2)):
            in_3 = self.bn_3(x_3[:,:,step])
            spk_out, mem_3 = self.lif_3(in_3, mem_3)
            spk_rec_3.append(spk_out)
            mem_rec_3.append(mem_3)

        x_4 = torch.stack(spk_rec_3, dim=2)
        #print(x_4.size())
              
        x_4 = F.pad(x_4, (self.left_pad,0), 'constant', 0)

        x_4 = self.dcls1d_2(x_4)
        #print(x_4.size())

        spk_rec_4 = []
        mem_rec_4 = []

        for step in range(x_4.size(2)):
            spk_out, mem_4 = self.lif_4(x_4[:,:,step], mem_4)
            spk_rec_4.append(spk_out)
            mem_rec_4.append(mem_4)

        #print(torch.stack(spk_rec_4).size())

        #exit()

        return torch.stack(spk_rec_4), torch.stack(mem_rec_4)

        #output of the form [time_steps, batch_size, features]


    def reset_model(self, train):

        # clamp parameters of dcls layers

        for m in self.modules():
            if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                m.clamp_parameters()


    def decrease_sig(self, epoch, num_epochs, time_steps):

        # Decreasing to 0.23
        final_epoch = num_epochs//4
        alpha = 0

        # max_delay = 250//time_steps
        max_delay = time_steps
        max_delay = max_delay if max_delay%2==1 else max_delay+1

        siginit = max_delay//2

        # take sig from last dcls layer  
        sig = self.dcls1d_2.SIG[0,0,0,0].detach().cpu().item()

        if epoch < final_epoch and sig > 0.23:

            alpha = (0.23/siginit)**(1/(final_epoch))

            for m in self.modules():
                if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                    m.SIG *= alpha


    def round_pos(self):
        with torch.no_grad():

            for m in self.modules():
                if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                    m.P.round_()
                    m.clamp_parameters()