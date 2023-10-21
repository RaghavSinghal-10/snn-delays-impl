import torch
import snntorch as snn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import tonic
import argparse
import wandb
from models import *


wandb.init(project='snn delay with conv')

parser = argparse.ArgumentParser(description='Evaluate SNN with delay conv')

parser.add_argument('--batch_size_train', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--batch_size_test', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs for training the linear classifier')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers for loading the data')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for training the linear classifier')
parser.add_argument('--device', default='cuda', type=str, help='device to be used for training the linear classifier')
parser.add_argument('--learn_beta', action='store_true', help='learn beta for the LIF neurons in the backbone')
parser.add_argument('--learn_threshold', action='store_true', help='learn threshold for the LIF neurons in the backbone')
parser.add_argument('--threshold', default=1.0, type=float, help='threshold for the LIF neurons in the linear classifier')
parser.add_argument("--surr", default="atan", type=str, help="surrogate neuron: atan, fast_sigmoid")

parser.add_argument('--label_percent', default=100, type=float, help='percentage of labeled data to be used for training the linear classifier')
parser.add_argument('--beta', default=0.5, type=float, help='beta for the LIF neurons in the backbone')
parser.add_argument('--time_steps', default=30, type=int, help='number of time steps for the SNN')

args = parser.parse_args()

wandb.config.update(args)


def main():

    print(args)

    device = args.device if torch.cuda.is_available() else "cpu"

    model = SNN_Delay_Conv_Small(beta=args.beta, threshold=args.threshold, surr = args.surr, learn_beta=args.learn_beta, 
                           learn_threshold=args.learn_threshold, time_steps=args.time_steps).to(device)

    #print trainable parameters of the model
    print("Parameters of the model: ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

    print(model)
    
    sensor_size = tonic.datasets.NMNIST.sensor_size

    transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.time_steps)
            ])

    full_dataset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=transform)
    testset = tonic.datasets.NMNIST(save_to='./data', train=False, transform=transform)

    if args.label_percent == 100:
        trainset = full_dataset
    else:
        # use a random 10% of the labeled data
        trainset, _ = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*args.label_percent/100), len(full_dataset)-int(len(full_dataset)*args.label_percent/100)])    
    
    print("trainset: ", len(trainset))
    print("testset: ", len(testset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers, drop_last=False)

    criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    p_params = []
    w_params = []

    for name, param in model.named_parameters():
        if param.requires_grad and '.P' in name:     
            p_params.append(param)
            print("p params", name)
        elif param.requires_grad:
            w_params.append(param)
            print("w params", name)

    optimizer_p = torch.optim.Adam(model.parameters(), lr=100*args.lr, betas=(0.9, 0.999))
    optimizer_w = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))


    loss_history = []
    accuracy_history = []

    model.train()

    for epoch in range(args.num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, (input, labels) in enumerate(trainloader, 0):
            
            labels = labels.type(torch.LongTensor)

            input = input.to(device)
            labels = labels.to(device)

            input = input.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

            optimizer_p.zero_grad()
            optimizer_w.zero_grad()

            #output of the form [time_steps, batch_size, features]
            spk_rec, mem_rec = model(input)

            loss = criterion(spk_rec, labels)
            loss.backward()

            optimizer_w.step()
            optimizer_p.step()

            model.reset_model(train=True)

            # print statistics
            running_loss += loss.detach().item()

            wandb.log({"train_loss": loss.detach().item()})

        #print epoch loss
        loss_history.append(running_loss / len(trainloader))
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

        total = 0
        running_acc = 0.0

        model.decrease_sig(epoch=i, num_epochs=args.num_epochs, time_steps=args.time_steps)


        model.eval()

        with torch.no_grad():

            for m in model.modules():
                if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                    m.SIG *= 0
                    m.version = 'max'
                    m.DCK.version = 'max'

            model.round_pos()

            for i, (images, labels) in enumerate(testloader, 0):
                
                labels = labels.type(torch.LongTensor)

                images = images.to(device)
                labels = labels.to(device)

                images = images.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
                #labels = labels.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.FloatTensor)    

                spk_rec, mem_rec = model(images)

                acc = SF.accuracy_rate(spk_rec, labels)*100

                model.reset_model(train=False)

                total += labels.size(0)
                running_acc += acc

            for m in model.modules():
                if isinstance(m, Dcls1d) or isinstance(m, Dcls3_1d):
                    m.version = 'gauss'
                    m.DCK.version = 'gauss'

            accuracy = running_acc / len(testloader)
            accuracy_history.append(accuracy)
            wandb.log({"test_acc": accuracy})

            print("Total: {}, Accuracy: {}".format(total, accuracy))
          
if __name__ == '__main__':
    main()
