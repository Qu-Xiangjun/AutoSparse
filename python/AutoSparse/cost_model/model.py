import torch
import torch.nn as nn
from torch.nn.init import normal_
import torch.nn.functional as F
import logging
try:
    import MinkowskiEngine as ME
except ImportError:
    logging.warning("MinkowskiEngine is not installed.")
from typing import *
from AutoSparse.model import cuda_device_id
from AutoSparse.cost_model.tokenizer import Tokenizer
from AutoSparse.cost_model.config import Config


class LambdaRankingLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(
            torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)
        ) * torch.abs(G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10.0, sigma=1.0):
        device = self.device
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros(
            (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device
        )
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.0)
        y_true_sorted.clamp_(min=0.0)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1.0 + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(
            ((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1
        ).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(
            min=-1e8, max=1e8
        )
        scores_diffs[torch.isnan(scores_diffs)] = 0.0
        weighted_probas = (
            torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights
        ).clamp(min=eps)
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss


class SparseMatrixEmbed_WACO_NET(nn.Module):

    def __init__(self, in_channels=1, middle_channels=32, out_feature=128, D=2):
        nn.Module.__init__(self)
        self.inplanes = middle_channels
        self.layer1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=5, stride=1, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            # 这些卷积是不是只设置了步长比较大，但没设置Dilate 的空洞卷积，是否可以使用空洞卷积呢？
            # 另外是否需要bias 呢？
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer6 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer7 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer8 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer9 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer10 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer11 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer12 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer13 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer14 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )

        self.glob_pool = nn.Sequential(
            ME.MinkowskiGlobalAvgPooling(), ME.MinkowskiToFeature()
        )

        self.shape_feature = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.matrix_embedding = nn.Sequential(
            nn.Linear(self.inplanes * 14, 256),
            nn.ReLU(),
            nn.Linear(256, out_feature),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

    def forward(self, x1):
        """
        Parameters
        ----------
        x1 : ME.SparseTensor
            Sparse matrix.
        x2 : torch.Tensor
            Normalized shape info: (row_num, col_num, nnz)
        """
        y1 = self.layer1(x1)  # [batch, H, W] -> [batch, H, W]
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y5 = self.layer5(y4)
        y6 = self.layer6(y5)
        y7 = self.layer7(y6)
        y8 = self.layer8(y7)
        y9 = self.layer9(y8)
        y10 = self.layer10(y9)
        y11 = self.layer11(y10)
        y12 = self.layer12(y11)
        y13 = self.layer13(y12)
        y14 = self.layer14(y13)

        y1 = self.glob_pool(y1)  # [batch, H, W] -> [batch, H, W]
        y2 = self.glob_pool(y2)
        y3 = self.glob_pool(y3)
        y4 = self.glob_pool(y4)
        y5 = self.glob_pool(y5)
        y6 = self.glob_pool(y6)
        y7 = self.glob_pool(y7)
        y8 = self.glob_pool(y8)
        y9 = self.glob_pool(y9)
        y10 = self.glob_pool(y10)
        y11 = self.glob_pool(y11)
        y12 = self.glob_pool(y12)
        y13 = self.glob_pool(y13)
        y14 = self.glob_pool(y14)

        # y = F.normalize(torch.cat((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14), dim=1))
        y = torch.cat(
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14), dim=-1
        )

        # 这里不类似WACO NET将shape info嵌入matrix embedding，只用其卷积部分即可
        y = self.matrix_embedding(y)
        # y = F.normalize(y)

        return y


class SparseMatrixEmbedNet(nn.Module):
    """Extract sparse feature from sparse matrix data."""

    def __init__(self, in_channels=1, middle_channels=128, out_feature=128, D=2):
        """
        Parameters
        ----------
        in_channels : int, optional
            _description_, by default 1
        middle_channels : int, optional
            Use some middle hid channel to contain more feature info, by default 32
        out_feature : int, optional
            _description_, by default 1
        D : int, optional
            Sparse matirx dimension count, by default 2
        """
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_feature = out_feature
        self.D = D
        self.layer1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, middle_channels, kernel_size=5, stride=1, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )

        # 使用 HDC Block 的空洞卷积来做提取
        self.layer2 = nn.Sequential(
            ME.MinkowskiConvolution(
                middle_channels,
                int(middle_channels/4),
                kernel_size=3,
                stride=1,
                dilation=1,
                dimension=D,
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            ME.MinkowskiConvolution(
                int(middle_channels/4),
                int(middle_channels/4),
                kernel_size=3,
                stride=1,
                dilation=2,
                dimension=D,
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            ME.MinkowskiConvolution(
                int(middle_channels/4),
                int(middle_channels/4),
                kernel_size=3,
                stride=1,
                dilation=3,
                dimension=D,
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            ME.MinkowskiConvolution(
                int(middle_channels/4), middle_channels, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer6 = nn.Sequential(
            ME.MinkowskiConvolution(
                middle_channels, middle_channels, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer7 = nn.Sequential(
            ME.MinkowskiConvolution(
                middle_channels, middle_channels, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer8 = nn.Sequential(
            ME.MinkowskiConvolution(
                middle_channels,
                int(middle_channels/4),
                kernel_size=3,
                stride=1,
                dilation=1,
                dimension=D,
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer9 = nn.Sequential(
            ME.MinkowskiConvolution(
                int(middle_channels/4),
                int(middle_channels/4),
                kernel_size=3,
                stride=1,
                dilation=2,
                dimension=D,
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer10 = nn.Sequential(
            ME.MinkowskiConvolution(
                int(middle_channels/4),
                int(middle_channels/4),
                kernel_size=3,
                stride=1,
                dilation=3,
                dimension=D,
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer11 = nn.Sequential(
            ME.MinkowskiConvolution(
                int(middle_channels/4), middle_channels, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer12 = nn.Sequential(
            ME.MinkowskiConvolution(
                middle_channels, middle_channels, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer13 = nn.Sequential(
            ME.MinkowskiConvolution(
                middle_channels, middle_channels, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.glob_pool = nn.Sequential(
            ME.MinkowskiGlobalAvgPooling(), ME.MinkowskiToFeature()
        )

        # 类似 WACO NET 合并多个中间卷积结果
        self.feature1 = nn.Sequential(
            nn.Linear(middle_channels * 7 + int(middle_channels/4)*6, 512), nn.ReLU(), nn.Linear(512, out_feature)
        )

        # 只使用简单的卷积结果，寄希望于充足的channel
        self.feature2 = nn.Sequential(
            nn.Linear(middle_channels, 512), nn.ReLU(), nn.Linear(512, out_feature)
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

    def forward1(self, x):
        """Same with WACO_NET using results of multi conv layer."""
        y1 = self.layer1(x)  # [batch, H, W] -> [batch, H, W]
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y5 = self.layer5(y4)
        y6 = self.layer6(y5)
        y7 = self.layer7(y6)
        y8 = self.layer8(y7)
        y9 = self.layer9(y8)
        y10 = self.layer10(y9)
        y11 = self.layer11(y10)
        y12 = self.layer12(y11)
        y13 = self.layer13(y12)

        y1 = self.glob_pool(y1)  # [batch, H, W] -> [batch, H, W]
        y2 = self.glob_pool(y2)
        y3 = self.glob_pool(y3)
        y4 = self.glob_pool(y4)
        y5 = self.glob_pool(y5)
        y6 = self.glob_pool(y6)
        y7 = self.glob_pool(y7)
        y8 = self.glob_pool(y8)
        y9 = self.glob_pool(y9)
        y10 = self.glob_pool(y10)
        y11 = self.glob_pool(y11)
        y12 = self.glob_pool(y12)
        y13 = self.glob_pool(y13)

        y = self.feature1(
            torch.cat((y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13), dim=-1)
        )

        return y

    def forward2(self, x):
        """Only rely on last conv layer result."""
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y5 = self.layer5(y4)
        y6 = self.layer6(y5)
        y7 = self.layer7(y6)
        y8 = self.layer8(y7)
        y9 = self.layer9(y8)
        y10 = self.layer10(y9)
        y11 = self.layer11(y10)
        y12 = self.layer12(y11)
        y13 = self.layer13(y12)
        y13 = self.glob_pool(y13)
        y = self.feature2(y13)
        return y


class AutoSparseNet(nn.Module):
    """Cost model for AutoSparse, using conv net embedding sparse matrix,
    and using transformer decoder to preidict program ranking.
    """

    def __init__(self, config: Config, eval = False):
        nn.Module.__init__(self)
        self.in_channels = config.in_channels
        self.device = config.device
        self.embedding_size = config.token_embedding_size
        self.middle_channels = config.middle_channel_num
        self.tensor_name_set = config.tensor_name_set
        self.D = config.D
        self.is_waco_net = config.is_waco_net
        self.is_net_forward1 = config.is_net_forward1

        # TODO: 这里改造为判断CUDA才创建一下的变量
        if torch.cuda.is_available():
            self.conv_sparse_feature_waco = SparseMatrixEmbed_WACO_NET(
                self.in_channels,
                self.middle_channels,
                self.embedding_size - len(Tokenizer.PRIMITIVES),
                self.D,
            )
            self.conv_sparse_feature = SparseMatrixEmbedNet(
                self.in_channels,
                self.middle_channels,
                self.embedding_size - len(Tokenizer.PRIMITIVES),
                self.D,
            )

        self.tokenizer = Tokenizer(self.embedding_size, self.tensor_name_set)

        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.res1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )
        self.res2 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if eval == False:
            self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if not isinstance(m, ME.MinkowskiConvolution):
                for p in m.parameters():
                    if p.dim() > 1:
                        normal_(p, mean=0.0, std=0.02)

    def embed_sparse_matirx_WACO(self, x):
        return self.conv_sparse_feature_waco(x)

    def embed_sparse_matirx(self, x):
        if self.is_net_forward1:
            return self.conv_sparse_feature.forward1(
                x
            )  # TODO:  这里可以更换卷积提取特征的网络方式
        return self.conv_sparse_feature.forward2(x)

    def forward(
        self,
        schedules: Union[str, List[str]],
        sparse_tensor_info: Tuple[float],
        sparse_matrix,
    ):
        input_seq = self.tokenizer(schedules, sparse_tensor_info).to(
            self.device
        )  # [32, 11, 128]
        one_hot_vec = (
            torch.Tensor(self.tokenizer.primitives_one_hot["sparse_feature"].copy())
            .reshape(1, 1, -1)
            .to(self.device)
        )
        if self.is_waco_net:
            embeded_sparse_feature = self.embed_sparse_matirx_WACO(
                sparse_matrix
            ).reshape(
                1, 1, -1
            )  # 1,1,128
        else:
            embeded_sparse_feature = self.embed_sparse_matirx(sparse_matrix).reshape(
                1, 1, -1
            )  # 1,1,128
        embeded_sparse_feature = torch.cat(
            (one_hot_vec, embeded_sparse_feature), dim=-1
        )
        embeded_sparse_feature = embeded_sparse_feature.expand(
            input_seq.size(0), -1, -1
        )

        input_seq_full = torch.cat(  # TODO： 这里sparse feature没有加one-hot码
            (input_seq, embeded_sparse_feature), dim=1
        )  # [32, 12, 128]

        encoder_seq = self.encoder(input_seq_full).transpose(
            0, 1
        )  # batch first is False
        output, attention_mask = self.attention(encoder_seq, encoder_seq, encoder_seq)
        output = output + self.res1(output)
        output = output + self.res2(output)
        output = self.decoder(output)  # shape [seq_len, batch, 1]

        output = output.sum(0).squeeze()

        return output

    def forward_in_query(
        self,
        schedules: Union[str, List[str]],
        sparse_tensor_info: torch.Tensor,
        sparse_matrix_embeded_feature: torch.Tensor,
    ) -> torch.Tensor:
        input_seq = self.tokenizer(schedules, sparse_tensor_info).to(self.device)
        one_hot_vec = (
            torch.Tensor(self.tokenizer.primitives_one_hot["sparse_feature"].copy())
            .reshape(1, 1, -1)
            .to(self.device)
        )
        embeded_sparse_feature = sparse_matrix_embeded_feature.reshape(
            1, 1, -1
        )
        embeded_sparse_feature = torch.cat(
            (one_hot_vec, embeded_sparse_feature), dim=-1
        )
        embeded_sparse_feature = embeded_sparse_feature.expand(
            input_seq.size(0), -1, -1
        )

        input_seq_full = torch.cat((input_seq, embeded_sparse_feature), dim=1)

        encoder_seq = self.encoder(input_seq_full).transpose(
            0, 1
        )  # batch first is False
        output, attention_mask = self.attention(encoder_seq, encoder_seq, encoder_seq)
        output = output + self.res1(output)
        output = output + self.res2(output)
        output = self.decoder(output)  # shape [seq_len, batch, 1]

        output = output.sum(0).squeeze()

        return output
