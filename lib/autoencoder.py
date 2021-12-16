import torch.nn as nn
import numpy as np
import torch
from lib.utils import save_checkpoint_allin1
import os


class Decoder_conv(nn.Module):
    # Decoder block with convolutional layers
    def __init__(self, latent_dim, fmri_dims, mask):
        super(Decoder_conv, self).__init__()

        n_voxels_conv2_output = int(np.product(np.array(fmri_dims) / 4))
        # a fc layer from latent_dim to fmri conv dim
        self.fc_decoder = nn.Linear(latent_dim, n_voxels_conv2_output * 64)

        # conv_transpose layers
        self.decoder_t_conv1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.decoder_t_conv2 = nn.ConvTranspose3d(32, 1, 2, stride=2)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.25)

        self.mask = mask
        
    def forward(self, z, compressed_shape):
        z = self.leaky_relu(self.fc_decoder(z))
        z = z.view(compressed_shape)
        z = self.leaky_relu(self.decoder_t_conv1(z))
        z = self.dropout(z)
        z = self.decoder_t_conv2(z)

        return z

class Encoder_conv(nn.Module):
    # Encoder block with conv layers
    def __init__(self, fmri_dims, latent_dim, mask ):
        super(Encoder_conv, self).__init__()
        self.encoder_conv1 = nn.Conv3d(1, 32, 3, padding=1) #in_channels=1, out_channels=32 (3x3x3)kernel
        self.encoder_conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.25)
        
        n_voxels_conv2_output = int(np.product(np.array(fmri_dims) / 4))
        self.fc_encoder = nn.Linear(64 * n_voxels_conv2_output, latent_dim)

        self.mask = mask

    def forward(self, x):
        x = self.leaky_relu(self.encoder_conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.leaky_relu(self.encoder_conv2(x))
        x = self.pool(x)

        compressed_shape = x.size()
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc_encoder(x))
        hidden_rep = x.clone().detach()
        

        return x, hidden_rep, compressed_shape

class Encoder_basic(nn.Module):
    # Basic MLP encoder block
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Encoder_basic,self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.leaky_relu = nn.LeakyReLU(0.1)
        # self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        hidden1 = self.leaky_relu(self.linear(x))
        hidden2 = self.leaky_relu(self.linear2(hidden1))
        hidden3 = self.leaky_relu(self.linear3(hidden2))

        return hidden3

class Encoder_basic2(nn.Module):
    # Basic MLP encoder block
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, bottleneck_dim):
        super(Encoder_basic2,self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.linear4 = nn.Linear(hidden_dim3, bottleneck_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
        # self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        hidden1 = self.leaky_relu(self.linear(x))
        hidden2 = self.leaky_relu(self.linear2(hidden1))
        hidden3 = self.leaky_relu(self.linear3(hidden2))
        hidden4 = self.leaky_relu(self.linear4(hidden3))
        return hidden4


class Decoder_basic(nn.Module):
    # Basic MLP decoder block
    def __init__(self, hidden_dim3, hidden_dim2, hidden_dim1, output_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim3, hidden_dim2)
        self.linear2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.linear3 = nn.Linear(hidden_dim1, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self,x):
        hidden2 = self.leaky_relu(self.linear(x))
        hidden1 = self.leaky_relu(self.linear2(hidden2))
        output = self.leaky_relu(self.linear3(hidden1))
        return output

class Decoder_basic2(nn.Module):
    # Basic MLP decoder block
    def __init__(self, bottleneck_dim, hidden_dim3, hidden_dim2, hidden_dim1, output_dim):
        super(Decoder_basic2, self).__init__()
        self.linear = nn.Linear(bottleneck_dim, hidden_dim3)
        self.linear2 = nn.Linear(hidden_dim3, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim1)
        self.linear4 = nn.Linear(hidden_dim1, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self,x):
        hidden3 = self.leaky_relu(self.linear(x))
        hidden2 = self.leaky_relu(self.linear2(hidden3))
        hidden1 = self.leaky_relu(self.linear3(hidden2))
        output = self.leaky_relu(self.linear4(hidden1))
        return output


class Decoder_Manifold(nn.Module):
    # Decoder with individual bottleneck manifold embedding layer
    def __init__(self, manifold_dim, hidden_dim3, hidden_dim2, hidden_dim1, output_dim):
        """
        :param manifold_dim: the bottleneck manifold dimensions, e.g. phate/tphate dim
        :param hidden_dim3: the output dim from encoder, this is reused to get symmetric encoder/decoder
        :param hidden_dim2: symmetric to encoder hidden_dim2
        :param hidden_dim1: symmetric to encoder hidden_dim1
        :param output_dim: the reconstructed dim same as encoder input_dim
        """
        super(Decoder_Manifold, self).__init__()
        self.linear_m = nn.Linear(hidden_dim3, manifold_dim)
        self.linear = nn.Linear(manifold_dim, hidden_dim3 )
        self.linear_1 = nn.Linear(hidden_dim3, hidden_dim2)
        self.linear_2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.linear_3 = nn.Linear(hidden_dim1, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self, x):
        z = self.linear_m(x)
        hidden3 = self.leaky_relu(self.linear(z))
        hidden2 = self.leaky_relu(self.linear_1(hidden3))
        hidden1 = self.leaky_relu(self.linear_2(hidden2))
        output = self.leaky_relu(self.linear_3(hidden1))
        return z, output


class Encoder_basic_btlnk(nn.Module):
    # add a bottleneck layer to the encoder, so that the encoder output the same bottleneck dim as the manifold
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, manifold_dim):
        super(Encoder_basic_btlnk,self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.linear_m = nn.Linear(hidden_dim3, manifold_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
        # self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        hidden1 = self.leaky_relu(self.linear(x))
        hidden2 = self.leaky_relu(self.linear2(hidden1))
        hidden3 = self.leaky_relu(self.linear3(hidden2))
        hidden_m = self.leaky_relu(self.linear_m(hidden3))

        return hidden_m


class Decoder_Manifold_btlnk(nn.Module):
    # Decoder with individual bottleneck manifold embedding layer
    def __init__(self, manifold_dim, hidden_dim3, hidden_dim2, hidden_dim1, output_dim):
        """
        :param manifold_dim: the bottleneck manifold dimensions, e.g. phate/tphate dim
        :param hidden_dim3: symmetric to encoder hidden_dim3
        :param hidden_dim2: symmetric to encoder hidden_dim2
        :param hidden_dim1: symmetric to encoder hidden_dim1
        :param output_dim: the reconstructed dim same as encoder input_dim
        """
        super(Decoder_Manifold_btlnk, self).__init__()
        self.linear_m = nn.Linear(manifold_dim, manifold_dim) #the input to this is the same bottleneck dimension of manifold dim
        self.linear = nn.Linear(manifold_dim, hidden_dim3)
        self.linear_1 = nn.Linear(hidden_dim3, hidden_dim2)
        self.linear_2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.linear_3 = nn.Linear(hidden_dim1, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self, x):
        z = self.linear_m(x)
        hidden3 = self.leaky_relu(self.linear(z))
        hidden2 = self.leaky_relu(self.linear_1(hidden3))
        hidden1 = self.leaky_relu(self.linear_2(hidden2))
        output = self.leaky_relu(self.linear_3(hidden1))
        return z, output

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x)
        return x

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x)
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_dim, classification_dim=2, hidden2=16) -> None:
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden2)
        self.linear2 = nn.Linear(hidden2, classification_dim)

        # self.linear = nn.Linear(hidden_dim, classification_dim)

        self.log_soft_max = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        # output = self.log_soft_max(self.linear(x))

        x = self.linear(x)
        output = self.log_soft_max(self.linear2(x))

        return output


# simple encoder+decoder
def train_ae(encoder, decoder, dataloader, n_epochs, lr, device):
    criterion = nn.MSELoss()
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr)

    mask = torch.from_numpy(decoder.mask).to(device)

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for data in dataloader:
            optimizer.zero_grad()
            data = data.unsqueeze(1).float()
            data = data.to(device)
            hidden, _, comp_shape = encoder(data)
            outputs = decoder(hidden, comp_shape)

            loss = criterion((outputs * mask).float(), data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(dataloader)
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.4f}")

    return encoder, decoder


def eval_ae(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()
    criterion = nn.MSELoss()

    mask = torch.from_numpy(decoder.mask).to(device)

    eval_loss=0.0
    for data in dataloader:
        data = data.unsqueeze(1).float()
        data = data.to(device)
        hidden, _, comp_shape = encoder(data)
        outputs = decoder(hidden, comp_shape)
        loss = criterion((outputs * mask).float(), data)
        eval_loss+=loss.item()*data.size(0)

    eval_loss = eval_loss/len(dataloader)
    print('Evaluation loss: ', eval_loss    )
    return eval_loss

def train_one_decoder(i, encoder, decoders, optimizer, dataloader, epoch, lr,criterion, device ):
    # first set each decoder require or not require grad 
    for j in range(len(decoders)):
        if j==i:
            for p in decoders[j].parameters():
                p.requires_grad=True
        else:
            for p in decoders[j].parameters():
                p.requires_grad=False
    train_loss = 0.0
    mask = torch.from_numpy(decoders[i].mask).to(device)
    for data in dataloader:
        optimizer.zero_grad()
        data = data.unsqueeze(1).float()
        data = data.to(device)
        hidden,_, comp_shape = encoder(data)
        outputs = decoders[i](hidden, comp_shape)
        loss = criterion((outputs * mask).float(), data)
        loss.backward()
        optimizer.step() # batch gradient descent
        train_loss+=loss.item()*data.size(0)
    train_loss = train_loss/len(dataloader)
    
    print(f"Pt: {i+1} \tEpoch: {epoch} \tTraining Loss: {train_loss:.4f}")
    
    return train_loss
        
#
# from lib.helper import adjust_learning_rate # reduce learning rate by 5% at each call.
# # Because training for each patient one epoch overall is 17 epochs update for encoder. Need to adjust the learning rate
# def train_ae_multidecoder(encoder, decoders, optimizer, dataloaders, n_epochs, lr, device, savepath = None, save_freq = 10,
#                           lr_decay=True):
#     if lr_decay:
#         logging.info('lr decay used, decay by 0.95 each pt epoch update')
#     else:
#         logging.info('lr decay not used')
#     criterion = nn.MSELoss()
#
#     if savepath is not None:
#         model_namelist = ['encoder']+ ['decoder_%d'%i for i in range(1,len(decoders)+1)]
#         save_checkpoint_allin1([encoder]+decoders, model_namelist,
#                                         optimizer,
#                                        os.path.join(savepath,'ae_e%d.pt'%0),
#                                        0)
#     print('saved initial checkpoint')
#     all_train_losses = []
#     for epoch in range(1, n_epochs + 1):
#         train_loss = 0.0
#         for i in range(len(decoders)):
#             loss = train_one_decoder(i, encoder, decoders, optimizer, dataloaders[i], epoch, lr, criterion, device)
#             all_train_losses.append(loss)
#             train_loss += loss
#
#             if lr_decay:
#                 # adjust learning rate after each patient epoch, reduce by 5%
#                 new_lr = adjust_learning_rate(lr,optimizer, (epoch-1)*17+i+1)
#                 if new_lr <= lr*0.1:
#                     lr_decay = False
#                     logging.info('epoch%d: stopped further lr decay after lr is 10% of initial lr'%epoch)
#
#         train_loss = train_loss / len(decoders)
#         print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.4f}")
#         if savepath is not None:
#             if epoch%save_freq==0 or epoch==n_epochs:
#                 model_namelist = ['encoder']+ ['decoder_%d'%i for i in range(1,len(decoders)+1)]
#                 save_checkpoint_allin1([encoder]+decoders, model_namelist,
#                                         optimizer,
#                                        os.path.join(savepath,'ae_e%d.pt'%epoch),
#                                        epoch)
#     if savepath is not None:
#         np.save(os.path.join(savepath, 'all_train_losses.npy'), np.array(all_train_losses))
#     return encoder, decoders
