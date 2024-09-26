from pathlib import Path
import nibabel as nib
from scipy import stats
import skimage.exposure as exp
from matplotlib import pyplot as plt


if __name__ == "__main__":
    data_path = "<INPUT_PATH>"
    img_folder = "img/"
    output_path = "<OUTPUT_PATH>"

    for p in Path(data_path + img_folder).rglob('*.nii.gz'):
        print(p)
        img = nib.load(p)
        data = img.get_fdata()

        data = data / 255

        data = exp.equalize_adapthist(data, clip_limit=0.01)

        out_img = nib.Nifti1Image(data, img.affine, img.header)
        out_path = output_path + p.relative_to(data_path).as_posix()
        print(out_path)
        nib.save(out_img, out_path)

        print("\n")
