import os
import requests
from pathlib import Path
import rawpy
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed


def dng_to_png(img_path: str):
    raw = rawpy.imread(img_path)
    rgb = raw.postprocess(use_camera_wb=True)
    new_name = img_path[:-3] + "jpg"
    Image.fromarray(rgb).save(new_name, optimize=True)
    os.remove(img_path)


def tif_to_png(img_path: str):
    img = Image.open(img_path)
    img.save(img_path[:-3] + "jpg")
    os.remove(img_path)


def download_img(img_name: str, data_dir: Path):
    try:
        dng_url = f"https://data.csail.mit.edu/graphics/fivek/img/dng/{img_name}.dng"
        dng_file = requests.get(dng_url, allow_redirects=True)
        dng_path = data_dir / "raw" / (img_name + ".dng")
        open(dng_path, "wb").write(dng_file.content)
        dng_to_png(str(dng_path))

        tif_base = "https://data.csail.mit.edu/graphics/fivek/img/tiff16"
        for expert in ["a", "b", "c", "d", "e"]:
            url = f"{tif_base}_{expert}/{img_name}.tif"
            path = data_dir / expert / (img_name + ".tif")
            tif_file = requests.get(url, allow_redirects=True)
            open(path, "wb").write(tif_file.content)
            tif_to_png(str(path))

    except:
        print(img_name)


def download_dataset(info_dir: Path, store_dir: Path, n_jobs: int = 8):
    # * Create folders
    dng_dir = store_dir / "raw"
    tif_dirs = [store_dir / s for s in ["a", "b", "c", "d", "e"]]

    dng_dir.mkdir(parents=True, exist_ok=True)
    for path in tif_dirs:
        path.mkdir(parents=True, exist_ok=True)

    # * Get image info
    with open(info_dir / "filesAdobe.txt", "r") as f:
        f1 = f.read().split("\n")
    with open(info_dir / "filesAdobeMIT.txt", "r") as f:
        f2 = f.read().split("\n")
    names = [x for x in set(f1 + f2) if x != ""]

    # * Download imgs
    parallel = Parallel(n_jobs, backend="multiprocessing")
    parallel(delayed(download_img)(name, store_dir) for name in tqdm(names))


if __name__ == "__main__":
    info_dir = Path("data/raw")
    store_dir = Path("data/raw/adobe_5k")
    download_dataset(info_dir, store_dir)
