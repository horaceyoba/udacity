# Imports here

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from workspace_utils import active_session
from collections import OrderedDict
import matplotlib.pyplot as plt
from torchvision import datasets, transforms,models

from input_args import get_input_args


# Main program function defined below
def architecture(arch):
    if(arch=='vgg16'):
        model=models.vgg16_bn(pretrained=True)
        param='classifier'
        input_size=25088
    elif (arch=='densenet'):
        model=models.densenet161(pretrained=True)
        param='classifier'
        input_size=2208
    elif (arch=='resnet'):
        model=models.resnet18(pretrained=True)
        param='fc'
        input_size=512
    elif (arch=='alexnet'):
        model=models.alexnet(pretrained=True)
        param='classifier'
        input_size=9216
    elif (arch=='squeezenet'):
        model=models.squeezenet1_0(pretrained=True)
        param='classifier'
        input_size=512
    elif (arch=='inception'):
        model=models.inception_v3(pretrained=True)
        param='fc'
        input_size=2048
    else:
        model=models.vgg16_bn(pretrained=True)
        param='classifier'
        input_size=25088
    }
    return model,param,input_size

def main():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    #Valid Datasets and loaders
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    #Test Datasets and loaders
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    in_arg = get_input_args()
    model,param,input_size=architecture(in_arg.arch)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(input_size, in_arg.hidden_unit1)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(0.5)),
                            ('fc2', nn.Linear(in_arg.hidden_unit1,in_arg.hidden_unit2)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(0.5)),
                            ('fc3', nn.Linear(in_arg.hidden_unit2,in_arg.hidden_unit3)),
                            ('relu3', nn.ReLU()),
                            ('dropout3', nn.Dropout(0.5)),
                            ('fc4', nn.Linear(in_arg.hidden_unit3, 102)),                      
                            ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    if (param=='classifier'):
        model.classifier = classifier
    else:
        model.fc=classifier
        
    with active_session():
    # at beginning of the script
        device = torch.device("cpu")
        if "gpu" in in_arg:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for device in ['cpu', 'cuda']:
            criterion = nn.NLLLoss()
            # Only train the classifier parameters, feature parameters are frozen
            optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
            model.to(device)
            epochs = in_arg.epochs
            train_losses= []
            steps = 0
            for e in range(epochs):
                running_loss = 0
                model.train()
                for ii, (inputs, labels) in enumerate(trainloader):
                    steps += 1
                    optimizer.zero_grad()
                    # Move input and label tensors to the GPU
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                else:
                    train_losses.append(running_loss/len(trainloader))
                    print("Epoch: {}/{}.. ".format(e+1, epochs),"Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)))
                    test_loss, accuracy=validation(model, testloader, criterion,device)
    # TODO: Do validation on the test set
    def validation(model, testloader, criterion,device):
        # at beginning of the script
        device = torch.device("cpu")
        if "gpu" in in_arg:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        test_losses = []
        test_loss = 0
        accuracy = 0
        model.to(device)
        # Turn off gradients for validation, saves memory and computations
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                log_ps = model.forward(inputs)
                test_loss1=criterion(log_ps, labels)
                test_loss += test_loss1.item()
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                test_losses.append(test_loss/len(testloader))
        print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
        return test_loss, accuracy
    # TODO: Save the checkpoint 
    checkpoint = {
              'state_dict': model.state_dict(),
              'class_to_idx':train_data.class_to_idx,
              'optimizer': optimizer.state_dict(),
              'model': model,
              'batch_size': 64,
              'epoch':in_arg.epochs,
              'loss':test_loss/len(testloader),
              'accuracy':accuracy/len(testloader)
             }

    torch.save(checkpoint, in_arg.save_dir+'/checkpoint.pth')

# Call to main function to run the program
if __name__ == "__main__":
    main()
