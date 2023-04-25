################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################
import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup. 
    This will serve as the encoder for our Image Captioning problem.
    '''

    def __init__(self, outputs, dropout=False, dropout_param = 0):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments
        
        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        super(CustomCNN, self).__init__()
        self.outputs = outputs

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if dropout:
            self.fc1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=128, out_features=1024),
                nn.ReLU())
            self.fc2 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=1024, out_features=1024),
                nn.ReLU())
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(in_features=128, out_features=1024),
                nn.ReLU())
            self.fc2 = nn.Sequential(
                nn.Linear(in_features=1024, out_features=1024),
                nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=self.outputs)
        )

    def forward(self, x):
        '''
        Pass the input through each layer defined in the __init__() function
        in order.

        Parameters:
            x => Input to the CNN
        '''
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.pool(output)
        # use -1 to let reshape() function to determine the dim
        output = torch.squeeze(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

class Resnet50(nn.Module):
    def __init__(self, outputs):
        super(Resnet50, self).__init__()
        self.outputs = outputs
        resnet = resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, outputs)
    
    def forward(self, x):
        out = self.resnet(x)
        out = self.fc(out.view(out.size(0), -1))
        return out

class CNN_LSTM(nn.Module):
    '''
    An encoder decoder architecture.
    Contains a reference to the CNN encoder based on model_type config value.
    Contains an LSTM implemented using PyTorch modules. This will serve as the decoder for our Image Captioning problem.
    '''

    def __init__(self, config_data, vocab):
        '''
        Initialize the embedding layer, LSTM, and anything else you might need.
        '''
        super(CNN_LSTM, self).__init__()
        self.vocab = vocab
        self.hidden_size = config_data['model']['hidden_size']
        self.embedding_size = config_data['model']['embedding_size']
        self.model_type = config_data['model']['model_type']
        self.max_length = config_data['generation']['max_length']
        self.deterministic = config_data['generation']['deterministic']
        self.temp = config_data['generation']['temperature']
        self.features = None
        
        # specify device
        self.device = torch.device('cuda' if torch.has_cuda else 'cpu')
        # LSTM cells
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=2, batch_first=True)
        # lower dimensional embedding
        # Word embeddings: similar words have a similar encoding
        self.embed = nn.Embedding(num_embeddings=self.vocab.idx, embedding_dim=self.embedding_size)
        # fully connected layer
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab.idx)
        # softmax output
        self.softmax = nn.Softmax(dim=2)
        if self.model_type == "Custom":
            self.dropout = config_data['model']['dropout']
            if self.dropout: 
                self.dropout_param = config_data['model']['dropout_param']
                self.encoder = CustomCNN(self.embedding_size, self.dropout, self.dropout_param).to(self.device)
            else:
                self.encoder = CustomCNN(self.embedding_size).to(self.device)
        if self.model_type == "Resnet":
            self.encoder = Resnet50(self.embedding_size).to(self.device)

    def forward(self, images, captions, teacher_forcing=True, deterministic=None, temp=None):
        '''
        Forward function for this model.
        If teacher forcing is true:
            - Pass encoded images to the LSTM at first time step.
            - Pass each encoded caption one word at a time to the LSTM at every time step after the first one.
        Else:
            - Pass encoded images to the LSTM at first time step.
            - Pass output from previous time step through the LSTM at subsequent time steps
            - Generate predicted caption from the output based on whether we are generating them deterministically or not.
        '''
        # Optimization
        images = images#.to(self.device)
        captions = captions#.to(self.device)
        # stores features from encoder
        features = self.encoder(images)
        self.features = features
        # captions embedded; size = [64, 23, 300]
        embedded_captions = self.embed(captions)
        # If teacher forcing is true

        if teacher_forcing:
            
            all_features = torch.cat((torch.unsqueeze(features, dim=1),
                                      embedded_captions[:, :-1, :]), 1)
            output, hidden_states = self.lstm(all_features)
            logits = self.fc(output)
            return logits

        # else
        # outputs = torch.empty((features.size(0), self.max_length, self.hidden_size)).to(self.device)
        # stores created words 
        deterministic = self.deterministic if deterministic is None else deterministic
        temp = self.temp if temp is None else temp
        word_lst = []
        embed_words, hidden_states = None, None
        # feed the lstm of embedded sample words
        for i in range(self.max_length):
            if i == 0:
                output, hidden_states = self.lstm(features.unsqueeze(1))
            else:
                if embed_words is None:
                    raise Exception("Sorry, input could not be None")
                output, hidden_states = self.lstm(embed_words.unsqueeze(1), hidden_states)
            logits = self.fc(output)
            # stores the word (batch size) then embed them
            # check deterministic or based on probability
            if deterministic:
                weights = self.softmax(logits).squeeze(1)
                words = torch.argmax(weights, dim=1)
            else:
                logits = logits / temp
                weights = self.softmax(logits)
#                 temp_device = torch.device('cuda' if torch.has_cuda else 'cpu')
                words = torch.multinomial(weights.to(self.device).squeeze(1), 1).view(-1).to(self.device)
            # stores
            embed_words = self.embed(words)         # for next lstm input
            word_lst.append(words.cpu().detach().numpy())
            # outputs[:, i, :] = output

        # logits = self.fc(outputs)
        return word_lst


def get_model(config_data, vocab):
    '''
    Return the LSTM model
    '''
    return CNN_LSTM(config_data, vocab)
