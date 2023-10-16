import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate, utils
from collections import OrderedDict
from DCLS.construct.modules import Dcls1d
    

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

        self.flatten = nn.Flatten()
        self.dcls1 = Dcls1d(in_channels=700, out_channels=256, kernel_count=1, stride=1, padding=0, dilated_kernel_size=self.max_delay, groups=1, bias=True, padding_mode='zeros', version='gauss')
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
                print(m)
                nn.init.constant_(m.SIG, self.siginit)
                m.SIG.requires_grad = False

    def forward(self, data):

        mem_1 = self.lif1.init_leaky()
        mem_2 = self.lif2.init_leaky()
        mem_3 = self.lif3.init_leaky()

        spk_rec_1 = [] 
        data_1 = data

        # if using tonic dataset, convert from [256, 10, 1, 700] to [256, 10, 700]
        # data_1 = data_1.squeeze(2)    


        # data: [batch_size, time_steps, features]
        #print(data_1.size())
        # # left pad the time_steps with T time steps of zeros
        data_1 = F.pad(data_1, (0,0,self.left_pad,0), 'constant', 0)
        #print(data_1.size())
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
    
    # def __init__(self, beta, learn_beta, threshold, learn_threshold, time_steps, num_layers=3, in_channels=140, out_channels=20):
    #     super().__init__()

    #     self.spike_grad = surrogate.atan()
    #     self.learn_beta = learn_beta
    #     self.learn_threshold = learn_threshold
    #     self.beta = beta
    #     self.threshold = threshold
    #     self.time_steps = time_steps
    #     self.num_layers = num_layers
    #     self.in_channels = in_channels
    #     self.out_channels = out_channels

    #     self.left_pad = 250 // self.time_steps

    #     self.max_delay = 250 // self.time_steps
    #     self.max_delay = self.max_delay if self.max_delay % 2 == 1 else self.max_delay + 1

    #     self.siginit = self.max_delay // 2

    #     self.layers = nn.ModuleList([
    #         Dcls1d(in_channels=self.in_channels if i == 0 else self.out_channels,
    #                out_channels=self.out_channels if i == self.num_layers - 1 else 256,
    #                kernel_count=1, stride=1, padding=0, dilated_kernel_size=self.max_delay, groups=1, bias=True,
    #                padding_mode='zeros', version='gauss')
    #         for i in range(self.num_layers)
    #     ] + [
    #         nn.Dropout(p=0.4) for _ in range(self.num_layers - 1)
    #     ] + [
    #         nn.BatchNorm1d(self.out_channels if self.num_layers > 1 else 256)
    #     ] + [
    #         snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True, output=i == self.num_layers - 1,
    #                   learn_beta=self.learn_beta, threshold=self.threshold, learn_threshold=self.learn_threshold)
    #         for i in range(self.num_layers)
    #     ])

    #     # set sig paramaters of the dcls1 and dcls2 to siginit and required_grad=False
    #     for m in self.layers:
    #         if isinstance(m, Dcls1d):
    #             nn.init.constant_(m.SIG, self.siginit)
    #             m.SIG.requires_grad = False

    # def forward(self, data):
    #     mems = [lif.init_leaky() for lif in self.layers if isinstance(lif, snn.Leaky)]
    #     mem_rec = []

    #     x = data
    #     for layer in self.layers:
    #         if isinstance(layer, Dcls1d):
    #             x = layer(x)
    #         else:
    #             if isinstance(layer, snn.Leaky):
    #                 spk_rec = []
    #                 for step in range(x.size(2)):
    #                     spk_out, mem = layer(x[:, :, step], mems.pop(0))
    #                     spk_rec.append(spk_out)
    #                     mems.append(mem)
    #                 x = torch.stack(spk_rec, dim=2)
    #                 if layer.output:
    #                     mem_rec = mems[-1].recording
    #             else:
    #                 x = layer(x)

    #     return x, mem_rec
        
    

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

            for m in self.modules():
                if isinstance(m, Dcls1d):
                    m.P.round_()
                    m.clamp_parameters()