from src.models import get_CTN
import torch


def test_CTN():
    model = get_CTN(
        in_channels=3,
        out_channels=3,
        hist_channels=8,
        encoder_name="Original",
        enc_hidden_list=[8, 8, 8, 8, 8],
        dec_hidden_list=[8, 8, 8, 8, 8],
        use_dropout=True,
    )

    img = torch.randn((2, 3, 64, 64))
    hist_enc1 = torch.randn((2, 8, 64, 64))
    hist_enc2 = torch.randn((2, 8, 64, 64))

    out = model(img, hist_enc1, hist_enc2)

    model2 = get_CTN(
        in_channels=3,
        out_channels=3,
        hist_channels=64,
        encoder_name="mobilenetv3_small_100",
        enc_hidden_list=[16, 16, 24, 48, 576],
        dec_hidden_list=[576, 48, 48, 64, 64],
        use_dropout=True,
    )
