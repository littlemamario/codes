class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False, normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        score = self.conv2(x, edge_index).squeeze()
        return x, score


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)

    class Multi_Head_Attention(nn.Module):
        def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
            super().__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class Attention_block(nn.Module):
        def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                     norm_layer=nn.LayerNorm):
            super().__init__()
            self.norm1 = norm_layer(dim)
            self.attn = Multi_Head_Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        def forward(self, x, score):
            x = score + self.drop_path(self.attn(self.norm1(x)))
            # x = x + self.drop_path(self.attn(x))
            return x

        class pipeline(nn.Module):
            def __init__(self, tcn_kernel_size, gcn_hidden_channels, gcn_out_channels, se_in_channels, se_reduction):
                super().__init__()

                self.tcn_kernel_size = tcn_kernel_size
                self.gcn_hidden_channels = gcn_hidden_channels
                self.gcn_out_channels = gcn_out_channels

                self.tcn = TemporalConvNet(num_inputs=64, num_channels=[64, 64, 64], kernel_size=self.tcn_kernel_size)
                self.gcn = GCN(in_channels=520, hidden_channels=self.gcn_hidden_channels,
                               out_channels=self.gcn_out_channels)
                self.multi_head = Attention_block(dim=520, drop=0., attn_drop=0.)

                self.clf = nn.Sequential(
                    # nn.Linear(64 * self.gcn_out_channels, 256),
                    nn.Linear(64 * 648, 256),
                    nn.ELU(),
                    nn.Linear(256, 256),
                    nn.ELU(),
                    nn.Linear(256, 4),
                )

            def forward(self, data):
                '''
                '''

                x = data.x
                # print(x)
                # print('x.shape', x.shape)
                edge_index = data.edge_index
                edge_weight = data.edge_weight
                # x = x.unsqueeze(0)
                x_tcn = x.reshape(-1, 64, x.shape[-1])
                # x_tcn = x_tcn.permute(0, 2, 1)
                x_tcn = self.multi_head(x_tcn)
                # x_tcn = x_tcn.permute(0, 2, 1)
                # print(x)
                # print('x_tcn.shape before tcn', x_tcn.shape)
                x_tcn = self.tcn(x_tcn)
                # print(x)
                # print('x_tcn.shape after tcn', x_tcn.shape)

                # x = x.reshape(-1, x.shape[-1])
                # print(x)
                # print('x.shape before gcn', x.shape)
                # print("in model", edge_index.max() < x.size(0))
                x_gcn = self.gcn(x, edge_index, edge_weight)
                # print(x)
                # print('x_gcn.shape after gcn', x_gcn.shape)

                x_gcn = x_gcn.reshape(-1, 64, self.gcn_out_channels)
                x = torch.cat((x_tcn, x_gcn), dim=2)
                # print(x)
                # print('x.shape after cat', x.shape)
                # x = x.reshape(-1, x.shape[-1])
                x = x.unsqueeze(-1)
                # print(x)
                # print('x.shape before se', x.shape)
                x = x.reshape(-1, 64, 648)
                # print(x)
                # print('x.shape before flatten', x.shape)
                x = torch.flatten(x, start_dim=1, end_dim=- 1)
                # print(x)
                # print('x.shape after flatten', x.shape)

                logits = self.clf(x)
                # print(logits.shape)

                return logits