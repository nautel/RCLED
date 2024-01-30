import torch
import torch.nn as nn
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available else "cpu"


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to("cuda")
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to("cuda")
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to("cuda")
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to("cuda"),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to("cuda"))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input_matrix):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input_matrix
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        #        return outputs
        return outputs, (x, new_c)


def attention(ConvLstm_out):
    attention_w = []
    for k in range(5):
        attention_w.append(torch.sum(torch.mul(ConvLstm_out[k], ConvLstm_out[-1])) / 5)
    m = nn.Softmax(dim=0)
    attention_w = torch.reshape(m(torch.stack(attention_w)), (-1, 5))
    cl_out_shape = ConvLstm_out.shape
    ConvLstm_out = torch.reshape(ConvLstm_out, (5, -1))
    convLstmOut = torch.matmul(attention_w, ConvLstm_out)
    convLstmOut = torch.reshape(convLstmOut, (cl_out_shape[1], cl_out_shape[2], cl_out_shape[3]))
    return convLstmOut


class CnnEncoder(nn.Module):
    def __init__(self, in_channels_encoder, paddings):
        super(CnnEncoder, self).__init__()
        # đầu vào, đầu ra, kernel, stride, padding
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_encoder, 32, kernel_size=3, stride=1, padding=paddings[0]),
            nn.SELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=paddings[1]),
            nn.SELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=(2, 2), padding=paddings[2]),
            nn.SELU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=(2, 2), padding=paddings[3]),
            nn.SELU()
        )

    def forward(self, X):
        conv1_out = self.conv1(X)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        return conv1_out, conv2_out, conv3_out, conv4_out


class Conv_LSTM(nn.Module):
    def __init__(self):
        super(Conv_LSTM, self).__init__()
        self.conv1_lstm = ConvLSTM(input_channels=32, hidden_channels=[32],
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv2_lstm = ConvLSTM(input_channels=64, hidden_channels=[64],
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv3_lstm = ConvLSTM(input_channels=128, hidden_channels=[128],
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv4_lstm = ConvLSTM(input_channels=256, hidden_channels=[256],
                                   kernel_size=3, step=5, effective_step=[4])

    def forward(self, conv1_out, conv2_out,
                conv3_out, conv4_out):
        conv1_lstm_out_att_5_step = []
        conv2_lstm_out_att_5_step = []
        conv3_lstm_out_att_5_step = []
        conv4_lstm_out_att_5_step = []
        for i in range(5):
            conv1_lstm_out = self.conv1_lstm(conv1_out)
            conv1_lstm_out_att = attention(conv1_lstm_out[0][0])
            conv1_lstm_out_att_5_step.append(conv1_lstm_out_att.unsqueeze(0))

            conv2_lstm_out = self.conv2_lstm(conv2_out)
            conv2_lstm_out_att = attention(conv2_lstm_out[0][0])
            conv2_lstm_out_att_5_step.append(conv2_lstm_out_att.unsqueeze(0))

            conv3_lstm_out = self.conv3_lstm(conv3_out)
            conv3_lstm_out_att = attention(conv3_lstm_out[0][0])
            conv3_lstm_out_att_5_step.append(conv3_lstm_out_att.unsqueeze(0))

            conv4_lstm_out = self.conv4_lstm(conv4_out)
            conv4_lstm_out_att = attention(conv4_lstm_out[0][0])
            conv4_lstm_out_att_5_step.append(conv4_lstm_out_att.unsqueeze(0))

        return torch.cat(conv1_lstm_out_att_5_step, dim=0), \
               torch.cat(conv2_lstm_out_att_5_step, dim=0), \
               torch.cat(conv3_lstm_out_att_5_step, dim=0), \
               torch.cat(conv4_lstm_out_att_5_step, dim=0)


class CnnDecoder(nn.Module):
    def __init__(self, in_channels, paddings, output_paddings):
        super(CnnDecoder, self).__init__()
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2, padding=paddings[0],
                               output_padding=output_paddings[0]),
            nn.SELU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=paddings[1],
                               output_padding=output_paddings[1]),
            nn.SELU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=paddings[2],
                               output_padding=output_paddings[2]),
            nn.SELU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=paddings[3],
                               output_padding=output_paddings[3]),
            nn.SELU()
        )

    def forward(self, conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out):
        deconv4 = self.deconv4(conv4_lstm_out)
        deconv4_concat = torch.cat((deconv4, conv3_lstm_out), dim=1)
        deconv3 = self.deconv3(deconv4_concat)
        deconv3_concat = torch.cat((deconv3, conv2_lstm_out), dim=1)
        deconv2 = self.deconv2(deconv3_concat)
        deconv2_concat = torch.cat((deconv2, conv1_lstm_out), dim=1)
        deconv1 = self.deconv1(deconv2_concat)
        return deconv1


class RCLEDmodel(nn.Module):
    # Autoencoder model
    def __init__(self,
                 num_vars,
                 in_channels_ENCODER,
                 in_channels_DECODER
                 ):
        self.dtype = torch.float32
        super().__init__()
        if num_vars == 25:
            ENCODER_paddings = [1, 1, 1, 1]

            DECODER_paddings = [1, 1, 1, 1]
            DECODER_out_paddings = [1, 1, 0, 0]
        if num_vars == 30:
            ENCODER_paddings = [1, 1, 1, 0]

            DECODER_paddings = [0, 1, 1, 1]
            DECODER_out_paddings = [0, 1, 1, 0]
        if num_vars == 55:
            ENCODER_paddings = [1, 1, 1, 1]

            DECODER_paddings = [1, 1, 1, 1]
            DECODER_out_paddings = [1, 0, 0, 0]

        self.cnn_encoder = CnnEncoder(in_channels_ENCODER, ENCODER_paddings)
        self.conv_lstm = Conv_LSTM()
        self.cnn_decoder = CnnDecoder(in_channels_DECODER, DECODER_paddings, DECODER_out_paddings)

    def forward(self, x):
        conv1_out, conv2_out, conv3_out, conv4_out = self.cnn_encoder(x)
        conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out = self.conv_lstm(
            conv1_out, conv2_out, conv3_out, conv4_out)

        gen_x = self.cnn_decoder(conv1_lstm_out, conv2_lstm_out,
                                 conv3_lstm_out, conv4_lstm_out)
        return gen_x


if __name__ == '__main__':
    print('Convolutional LSTM Encoder Decoder')
