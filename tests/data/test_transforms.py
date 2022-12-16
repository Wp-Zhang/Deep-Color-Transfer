import src.data.transforms as t
from PIL import Image
import numpy as np


def test_LAB_Trans():
    img = Image.open("tests/test_data/test/in_imgs/1.jpg").convert("RGB")
    transform = t.get_transform_lab()

    lab_img = transform(img)
    re_trans_img = t.post_process_img(lab_img)
    re_trans_img = (re_trans_img * 255).astype(int)

    assert np.abs((np.array(img) - re_trans_img)).mean() < 0.5


def test_HUE_shift():
    img = Image.open("tests/test_data/test/in_imgs/1.jpg").convert("RGB")
    transform = t.get_transform_hueshiftlab()

    _ = transform(img)
