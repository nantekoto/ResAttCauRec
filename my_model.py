from fightingcv_attention.attention.CBAM import CBAMBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

class CauRecNet(nn.Module):
    
    '''
    causality matrix as cell state, CI-LSTM route, all site version, different cell_state within batch
    input: batch_size * seq_len * feature_num
    output: batch_size * 1
    '''

    def __init__(self, len_input = 15, len_output = 1, input_feature = 12, num_h1 = 64, num_h2 = 128):
        super().__init__()


        self.fc1 = nn.Linear(input_feature*8, num_h1)
        self.fc2 = nn.Linear(input_feature*8, num_h2)

        self.len_input = len_input
        self.len_output = len_output
        self.num_h1 = num_h1
        self.num_h2 = num_h2
        self.lstm0 = nn.LSTMCell(input_size = input_feature, hidden_size  =  num_h1)
        self.lstm1 = nn.LSTMCell(input_size = num_h1, hidden_size = num_h2)
        self.dropout = nn.Dropout(p=0.2)
        self.dense1 = nn.Linear(num_h2, num_h1)
        self.dense2 = nn.Linear(num_h1, len_output)

        
    def forward(self, input_seq, cell_state):
        
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # causality matrix as cell state, CI-LSTM route, all site version, different cell_state within batch
        device = input_seq.device

        h_l0 = torch.zeros(batch_size, self.num_h1).to(device)

        c_l0 = self.fc1(cell_state)

        h_l1 = torch.zeros(batch_size, self.num_h2).to(device)

        c_l1 = self.fc2(cell_state)

        for t in range(seq_len):
            h_l0, c_l0_res = self.lstm0(input_seq[:, t, :], (h_l0, c_l0))

            c_l0 = c_l0 + c_l0_res

            h_l0, c_l0 = self.dropout(h_l0), self.dropout(c_l0)

            h_l1, c_l1_res = self.lstm1(h_l0, (h_l1, c_l1))

            c_l1 = c_l1 + c_l1_res

            h_l1, c_l1 = self.dropout(h_l1), self.dropout(c_l1)
        pred = self.dense2(self.dense1(h_l1))
        return pred

            
    # if cell_state == None:
        #   # no causality, standard LSTM route
        #   h_l0 = torch.zeros(batch_size, self.num_h1).to(device)
        #   c_l0 = torch.zeros(batch_size, self.num_h1).to(device)
        #   h_l1 = torch.zeros(batch_size, self.num_h2).to(device)
        #   c_l1 = torch.zeros(batch_size, self.num_h2).to(device)
        #   for t in range(seq_len):
        #       h_l0, c_l0 = self.lstm0(input_seq[:, t, :], (h_l0, c_l0))
        #       h_l0, c_l0 = self.dropout(h_l0), self.dropout(c_l0)
        #       h_l1, c_l1 = self.lstm1(h_l0, (h_l1, c_l1))
        #       h_l1, c_l1 = self.dropout(h_l1), self.dropout(c_l1)
        #   pred = self.dense2(self.dense1(h_l1))
        #   return pred

#         if len(cell_state.shape) == 1:
#           # causality matrix as cell state, CI-LSTM route, one site version, sharing the same cell_state
#           h_l0 = torch.zeros(batch_size, self.num_h1).to(device)

#           c_l0 = self.fc1(cell_state).repeat(batch_size, 1)

#           h_l1 = torch.zeros(batch_size, self.num_h2).to(device)

#           c_l1 = self.fc2(cell_state).repeat(batch_size, 1)

#           for t in range(seq_len):
#               h_l0, c_l0_res = self.lstm0(input_seq[:, t, :], (h_l0, c_l0))

#               c_l0 = c_l0 + c_l0_res

#               h_l0, c_l0 = self.dropout(h_l0), self.dropout(c_l0)

#               h_l1, c_l1_res = self.lstm1(h_l0, (h_l1, c_l1))

#               c_l1 = c_l1 + c_l1_res

#               h_l1, c_l1 = self.dropout(h_l1), self.dropout(c_l1)
#           pred = self.dense2(self.dense1(h_l1))
#           return pred



class RecNet(nn.Module):
    '''
    classic recurrent network with 2-layer lstm basicblock
    input: batch_size * seq_len * feature_num
    output: batch_size * 1
    '''

    def __init__(self, len_input = 15, len_output = 1, input_feature = 12, num_h1 = 64, num_h2 = 128):
        super().__init__()


        self.fc1 = nn.Linear(input_feature*8, num_h1)
        self.fc2 = nn.Linear(input_feature*8, num_h2)

        self.len_input = len_input
        self.len_output = len_output
        self.num_h1 = num_h1
        self.num_h2 = num_h2
        self.lstm0 = nn.LSTMCell(input_size = input_feature, hidden_size  =  num_h1)
        self.lstm1 = nn.LSTMCell(input_size = num_h1, hidden_size = num_h2)
        self.dropout = nn.Dropout(p=0.2)
        self.dense1 = nn.Linear(num_h2, num_h1)
        self.dense2 = nn.Linear(num_h1, len_output)

    def forward(self, input_seq, cell_state = None):
        # no causality, standard LSTM route
        
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]        
        device = input_seq.device
        
        h_l0 = torch.zeros(batch_size, self.num_h1).to(device)
        c_l0 = torch.zeros(batch_size, self.num_h1).to(device)
        h_l1 = torch.zeros(batch_size, self.num_h2).to(device)
        c_l1 = torch.zeros(batch_size, self.num_h2).to(device)
        for t in range(seq_len):
            h_l0, c_l0 = self.lstm0(input_seq[:, t, :], (h_l0, c_l0))
            h_l0, c_l0 = self.dropout(h_l0), self.dropout(c_l0)
            h_l1, c_l1 = self.lstm1(h_l0, (h_l1, c_l1))
            h_l1, c_l1 = self.dropout(h_l1), self.dropout(c_l1)
        pred = self.dense2(self.dense1(h_l1))
        return pred        


class ResAttBlock(nn.Module):
    '''
    classic resnet basicblock with attention block attached 
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResAttBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #        
        self.cbam = CBAMBlock(channel=out_channels,reduction=8,kernel_size=7)

        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResBlock(nn.Module):
    '''
    classic resnet basicblock 
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #        
        # self.cbam = CBAMBlock(channel=out_channels,reduction=8,kernel_size=7)

        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = self.cbam(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out    
    
    

class ResAttNet(nn.Module):
    '''
    this model is to fuse data in area into one point
    input: batch_size * feature_num * area_len * area_wid
    output: batch_size * feature_num 
    '''
    def __init__(self, in_channel=12, out_dim=12):
        super(ResAttNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        #self.layer3 = self._make_layer(128, 256, 2, stride=2)
        #self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, out_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResAttBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResAttBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet(nn.Module):
    '''
    this model is to fuse data in area into one point
    input: batch_size * feature_num * area_len * area_wid
    output: batch_size * feature_num 
    '''
    def __init__(self, in_channel=12, out_dim=12):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        #self.layer3 = self._make_layer(128, 256, 2, stride=2)
        #self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, out_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    
'''
complete models below
'''    
class ResAttCauRec(nn.Module):
    '''
    residual attention causal recurrent model, full model
    '''
    def __init__(self, len_input = 15, ):
        super(ResAttCauRec, self).__init__()
        self.len_input = len_input
    
        self.resattnet = ResAttNet()
        self.cilstm = CauRecNet(len_input = len_input,)

    
    def cell_state_in(self, cell_state):
        self.cell_state = cell_state

    def forward(self, x, cell_state = None):
        if cell_state is not None:
            self.cell_state = cell_state

        batch_size, len_input, *feature_dims = x.size()
        input_tensor_reshaped = x.view(-1, *feature_dims)
        output_tensor_reshaped = self.resattnet(input_tensor_reshaped)
        output_tensor = output_tensor_reshaped.view(batch_size, len_input, *output_tensor_reshaped.size()[1:])
        
        out = self.cilstm(output_tensor,  self.cell_state[:batch_size])
        
        return out
    

class ResAttRec(nn.Module):
    '''
    residual attention recurrent model, without causal module
    '''
    
    def __init__(self, len_input = 15, ):
        super(ResAttRec, self).__init__()
        self.len_input = len_input
    
        self.resattnet = ResAttNet()
        self.lstm = RecNet(len_input = len_input,)

    def forward(self, x, cell_state):
        
        batch_size, len_input, *feature_dims = x.size()
        input_tensor_reshaped = x.view(-1, *feature_dims)
        output_tensor_reshaped = self.resattnet(input_tensor_reshaped)
        output_tensor = output_tensor_reshaped.view(batch_size, len_input, *output_tensor_reshaped.size()[1:])
        
        out = self.lstm(output_tensor )
        
        return out

    
class ResAtt(nn.Module):
    '''
    residual attention model, without causal recurrent module
    additional fc layers to get output
    '''
    def __init__(self, len_input = 15, ):
        super(ResAtt, self).__init__()
        self.len_input = len_input
    
        self.resattnet = ResAttNet()
        # self.cilstm = CauRecNet(len_input = len_input,)
        self.fc1 = nn.Linear(len_input * 12, 1)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(64, 1)

    def forward(self, x, cell_state):
        
        batch_size, len_input, *feature_dims = x.size()
        input_tensor_reshaped = x.view(-1, *feature_dims)
        output_tensor_reshaped = self.resattnet(input_tensor_reshaped)
        output_tensor = output_tensor_reshaped.view(batch_size, -1)
        
        out = self.fc1(output_tensor)
        
        return out

    
class ResCauRec(nn.Module):
    '''
    residual causal recurrent model, without attention module
    '''
    def __init__(self, len_input = 15, ):
        super(ResCauRec, self).__init__()
        self.len_input = len_input
    
        self.resattnet = ResNet()
        self.cilstm = CauRecNet(len_input = len_input,)

    def forward(self, x, cell_state):
        
        batch_size, len_input, *feature_dims = x.size()
        input_tensor_reshaped = x.view(-1, *feature_dims)
        output_tensor_reshaped = self.resattnet(input_tensor_reshaped)
        output_tensor = output_tensor_reshaped.view(batch_size, len_input, *output_tensor_reshaped.size()[1:])
        
        out = self.cilstm(output_tensor, cell_state)
        
        return out

    
class CauRec(nn.Module):
    '''
    causal recurrent model, without residual attention module
    the average value of the nearest four grids as input of CauRec.
    '''

    def __init__(self, len_input = 15, ):
        super(CauRec, self).__init__()
        self.len_input = len_input
    
        # self.resattnet = ResAttNet()
        self.cilstm = CauRecNet(len_input = len_input,)

    def forward(self, x, cell_state):
        
        # batch_size, len_input, *feature_dims = x.size()
        # input_tensor_reshaped = x.view(-1, *feature_dims)
        # output_tensor_reshaped = self.resattnet(input_tensor_reshaped)
        #output_tensor = output_tensor_reshaped.view(batch_size, len_input, *output_tensor_reshaped.size()[1:])
        
        x = torch.mean(torch.flatten(x[:,:,:,4:6,4:6], start_dim = 3), dim = 3)
        out = self.cilstm(x, cell_state)
        
        return out
    
class ResRec(nn.Module):
    '''
    residual recurrent model, without attention and causal modules
    '''
    
    def __init__(self, len_input = 15, ):
        super().__init__()
        self.len_input = len_input
    
        self.resattnet = ResNet()
        self.lstm = RecNet(len_input = len_input,)

    def forward(self, x, cell_state):
        
        batch_size, len_input, *feature_dims = x.size()
        input_tensor_reshaped = x.view(-1, *feature_dims)
        output_tensor_reshaped = self.resattnet(input_tensor_reshaped)
        output_tensor = output_tensor_reshaped.view(batch_size, len_input, *output_tensor_reshaped.size()[1:])
        
        out = self.lstm(output_tensor )
        
        return out

    
class Res(nn.Module):
    '''
    residual model, without attention and causal recurrent modules
    additional fc layers to get output
    '''
    def __init__(self, len_input = 15, ):
        super().__init__()
        self.len_input = len_input
    
        self.resattnet = ResNet()
        # self.cilstm = CauRecNet(len_input = len_input,)
        self.fc1 = nn.Linear(len_input * 12, 1)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(64, 1)

    def forward(self, x, cell_state):
        
        batch_size, len_input, *feature_dims = x.size()
        input_tensor_reshaped = x.view(-1, *feature_dims)
        output_tensor_reshaped = self.resattnet(input_tensor_reshaped)
        output_tensor = output_tensor_reshaped.view(batch_size, -1)
        
        out = self.fc1(output_tensor)
        
        return out
    
    


class Rec(nn.Module):
    '''
    recurrent model, without residual attention and causal modules
    the average value of the nearest four grids as input.
    '''

    def __init__(self, len_input = 15, ):
        super().__init__()
        self.len_input = len_input
    
        # self.resattnet = ResAttNet()
        self.cilstm = RecNet(len_input = len_input,)

    def forward(self, x, cell_state):
        
        # batch_size, len_input, *feature_dims = x.size()
        # input_tensor_reshaped = x.view(-1, *feature_dims)
        # output_tensor_reshaped = self.resattnet(input_tensor_reshaped)
        #output_tensor = output_tensor_reshaped.view(batch_size, len_input, *output_tensor_reshaped.size()[1:])
        
        x = torch.mean(torch.flatten(x[:,:,:,4:6,4:6], start_dim = 3), dim = 3)
        out = self.cilstm(x, cell_state)
        
        return out


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class UNet3D(nn.Module):
    def __init__(self, in_channels = 12, features=[24, 32, 40, 64]):
        """
        Initialize the 3D U-Net for regression.
        
        Args:
            in_channels (int): Number of input channels
            features (list): List of feature sizes for each level of the U-Net
        """
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1])

        # Decoder
        for feature in reversed(features[:-1]):
            self.decoder.append(
                nn.ConvTranspose3d(
                    in_channels, feature, kernel_size=2, stride=2
                )
            )
            self.decoder.append(DoubleConv3D(feature * 2, feature))
            in_channels = feature

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(features[0], 1)  # Scalar output

    def forward(self, x, cell_state):
        # Input: [B, T, C, W, H] → reshape to [B, C, T, W, H]
        x = x.permute(0, 2, 1, 3, 4)

        # Encoder
        skip_connections = []
        for encoder in self.encoder[:-1]:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.encoder[-1](x)
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx+1](x)

        x = self.global_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, C, 1, 1, 1] → [B, C]
        return self.fc(x)  # [B, 1]