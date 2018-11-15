import torch
import torch.nn as nn
from trnsf_Models import Encoder, Decoder


class Model(nn.Module):
    """
    - A VGG-style 3D CNN with 11 layers.
    - Kernel size is kept 3 for all three dimensions - (time, H, W)
      except the first layer has kernel size of (3, 5, 5)
    - Time dimension is preserved with `padding=1` and `stride=1`, and is
      averaged at the end

    Arguments:
    - Input: a (batch_size, 3, sequence_length, W, H) tensor
    - Returns: a (batch_size, 512) sized tensor
    """

    def __init__(
        self, 
        column_units,
        num_classes,
        n_src_vocab=100000,
        n_tgt_vocab=100000,
        transformer_d_model=512, 
        transformer_d_inner=2048, 
        transformer_dropout=0.1,
        transformer_n_layers=6, 
        transformer_n_head=8, 
        transformer_d_k=64, 
        transformer_len_max_seq=1000,
        transformer_tgt_emb_prj_weight_sharing=True):

        super(Model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block4 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        # self.block5 = nn.Sequential(
        #     nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(inplace=True),
        # )

        self.block5 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(3, 1, 1), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(3, 1, 1), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        self.transformer_encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=transformer_len_max_seq,
            d_word_vec=transformer_d_model, d_model=transformer_d_model, d_inner=transformer_d_inner,
            n_layers=transformer_n_layers, n_head=transformer_n_head, d_k=transformer_d_k, d_v=transformer_d_v,
            dropout=transformer_dropout)


        self.transformer_decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=transformer_len_max_seq,
            d_word_vec=transformer_d_model, d_model=transformer_d_model, d_inner=transformer_d_inner,
            n_layers=transformer_n_layers, n_head=transformer_n_head, d_k=transformer_d_k, d_v=transformer_d_v,
            dropout=transformer_dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.transformer_decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        self.classifier = nn.Sequential(
            nn.Linear(256*288*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # get convolution column features

        x = self.block1(x)
        # print(x.size())
        x = self.block2(x)
        # print(x.size())
        x = self.block3(x)
        # print(x.size())
        x = self.block4(x)
        # print(x.size())
        x = self.block5(x)
        # print(x.size())

        # flatten
        x = x.view(-1,256,288)
        # print(x.size())

        src_pos = torch.FloatTensor([list(range(288))]*list(x.shape)[0])

        enc_output, *_ = self.encoder(x, src_pos)
        #dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        #seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        flattened = enc_output.view(-1,256*288*2)

        result_pre_softmax = self.classifier(enc_output)

        return result_pre_softmax#, seq_logit.view(-1, seq_logit.size(2))


if __name__ == "__main__":
    num_classes = 174
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 72, 84, 84))
    model = Model(512)

    output = model(input_tensor)
    print(output.size())
