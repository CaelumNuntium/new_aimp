print("Loading modules...\nPlease wait...")

import pathlib
import argparse
import os
import math
import numpy
import warnings
import cv2
from astropy.io import fits
from astropy import visualization
from astropy import stats
from photutils import detection
from photutils import aperture
from matplotlib import pyplot as plt


def make_bias(orig_files):
    if len(orig_files) > 0:
        bias_data = numpy.ndarray(shape=shape, dtype=datatype)
        numpy.median([im[0].data for im in bias_files], axis=0, out=bias_data)
    else:
        bias_data = numpy.zeros(shape=shape, dtype=datatype)
    bias_header = fits.Header()
    bias_header["IMAGETYP"] = "bias"
    bias_header["EXPTIME"] = 0.0
    bias_hdu = fits.PrimaryHDU(data=bias_data, header=bias_header)
    return fits.HDUList([bias_hdu])


def make_dark(orig_files, bias, exptime):
    bias_data = bias[0].data
    if len(orig_files) > 0:
        dark_data = numpy.ndarray(shape=shape, dtype=datatype)
        numpy.median([(im[0].data - bias_data) * (exptime / im[0].header["EXPTIME"]) for im in dark_files], axis=0, out=dark_data)
    else:
        dark_data = numpy.zeros(shape=shape, dtype=datatype)
    dark_header = fits.Header()
    dark_header["EXPTIME"] = exptime
    dark_hdu = fits.PrimaryHDU(data=dark_data, header=dark_header)
    return fits.HDUList([dark_hdu])


def make_flat(orig_files, bias, dark, exptime):
    bias_data = bias[0].data
    dark_data = dark[0].data
    if len(orig_files) > 0:
        flat_data = numpy.ndarray(shape=shape, dtype=datatype)
        numpy.median(
            [(im[0].data - bias_data) * (exptime / im[0].header["EXPTIME"]) - dark_data for im in flat_files],
            axis=0, out=flat_data)
        valmax = numpy.max(flat_data)
        flat_data = flat_data / valmax
    else:
        flat_data = numpy.ones(shape=shape, dtype=datatype)
    flat_header = fits.Header()
    flat_header["IMAGETYP"] = "obj"
    flat_header["OBJECT"] = "sunsky"
    flat_header["EXPTIME"] = exptime
    flat_hdu = fits.PrimaryHDU(data=flat_data, header=flat_header)
    return fits.HDUList([flat_hdu])


def align_images(obj_images, ref_stars=[]):
    all_sources = []
    n_rows = math.floor(math.sqrt(len(obj_images) / 2))
    n_cols = math.ceil(len(obj_images) / n_rows)
    for ii in range(len(obj_images)):
        im = obj_images[ii]
        mean, median, std = stats.sigma_clipped_stats(im, sigma=3.0)
        detector = detection.DAOStarFinder(fwhm=7.0, threshold=7.0 * std)
        sources = detector(im - median)
        sources.add_index("id")
        all_sources.append(sources)
        positions = numpy.transpose((sources["xcentroid"], sources["ycentroid"]))
        if len(ref_stars) == 0:
            apertures = aperture.CircularAperture(positions, r=8.0)
            norm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.LogStretch())
            plt.subplot(n_rows, n_cols, ii + 1)
            plt.imshow(im, cmap="Greys", origin="lower", norm=norm, interpolation="nearest")
            apertures.plot(color="blue", lw=1.5, alpha=0.5)
            for star in sources:
                plt.annotate(f"{int(star['id'])}", xy=(float(star["xcentroid"]) + 6, float(star["ycentroid"]) + 6))
    ref_points_ids = []
    if len(ref_stars) == 0:
        print("\nMultiple object images selected. For correct alignment, please, select reference stars.")
        print("Select and write down the same stars on the images below (minimum 3 stars required)")
        print("Press Enter to show the images:")
        input()
        plt.show()
        print("Please, select the same stars on the images; write down their numbers in the correct order")
        for ii in range(len(obj_images)):
            print(f"Reference stars ids on image {ii}:")
            ref_points_ids.append([int(s) for s in input().split()])
    elif len(ref_stars) != len(obj_images):
        print("Error: Reference stars must be selected on all images!")
        exit(4)
    else:
        ref_points_ids = ref_stars
    n_points = len(ref_points_ids[0])
    if n_points < 3:
        print("Error: Minimum 3 reference stars required!")
    for pid in ref_points_ids:
        if len(pid) != n_points:
            print("Error: The number of reference stars must be the same on all images!")
            exit(4)
    dst_points_ids = ref_points_ids[0]
    dst_points = [
        (float((all_sources[0].loc["id", sid])["xcentroid"]), float((all_sources[0].loc["id", sid])["ycentroid"])) for
        sid in dst_points_ids]
    print("Selected stars:")
    print("  Image 0:")
    for j in range(len(dst_points_ids)):
        print(f"    # {dst_points_ids[j]}: {dst_points[j]}")
    for ii in range(1, len(ref_points_ids)):
        scr_points_ids = ref_points_ids[ii]
        scr_points = [
            (float((all_sources[ii].loc["id", sid])["xcentroid"]), float((all_sources[ii].loc["id", sid])["ycentroid"]))
            for sid in scr_points_ids]
        print(f"  Image {ii}:")
        for j in range(len(scr_points_ids)):
            print(f"    # {scr_points_ids[j]}: {scr_points[j]}")
        tfm = numpy.float32([[1, 0, 0], [0, 1, 0]])
        A = []
        b = []
        for sp, dp in zip(scr_points, dst_points):
            A.append([sp[0], 0, sp[1], 0, 1, 0])
            A.append([0, sp[0], 0, sp[1], 0, 1])
            b.append(dp[0])
            b.append(dp[1])
        result, residuals, rank, s = numpy.linalg.lstsq(numpy.array(A), numpy.array(b), rcond=None)
        a0, a1, a2, a3, a4, a5 = result
        tfm = numpy.float32([[a0, a2, a4], [a1, a3, a5]])
        print("Found affine transform:")
        print(f"determinant = {numpy.linalg.det(tfm[:2, :2])}")
        print(f"x_shift = {tfm[0, 2]}")
        print(f"y_shift = {tfm[1, 2]}")
        tmp = numpy.ndarray(shape=(obj_images[ii].shape[0], obj_images[ii].shape[1], 3))
        tmp[:, :, 0] = obj_images[ii]
        tmp[:, :, 1] = numpy.zeros(obj_images[ii].shape)
        tmp[:, :, 2] = numpy.zeros(obj_images[ii].shape)
        tmp = cv2.warpAffine(tmp, tfm, (obj_images[ii].shape[1], obj_images[ii].shape[0]), borderValue=numpy.nan)
        obj_images[ii] = tmp[:, :, 0]


attributes = ["DATE-OBS", "TELESCOP", "OBJECT", "IMAGETYP", "EXPTIME", "RA", "DEC"]
attrs_to_save = ["DATE-OBS", "TELESCOP", "INSTRUME", "OBJECT", "PROG-ID", "OBSERVAT", "DETECTOR"]
fits_files = []
bias_nums = []
dark_nums = []
flat_nums = []
have_dark, have_flat = False, False
datatype = numpy.dtype('f4')

print("OK")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", action="store", default=".", help="Input directory")
args = parser.parse_args()

fits_names = [str(_) for _ in pathlib.Path(args.dir).glob("**/*.fits") if _ not in pathlib.Path(args.dir).glob("**/aimp/**/*.fits")] + [str(_) for _ in pathlib.Path(args.dir).glob("**/*.fts") if _ not in pathlib.Path(args.dir).glob("**/aimp/**/*.fts")] + [str(_) for _ in pathlib.Path(args.dir).glob("**/*.fit") if _ not in pathlib.Path(args.dir).glob("**/aimp/**/*.fit")]
fits_names = [_ for _ in fits_names]
print(f"{len(fits_names)} FITS files found in {args.dir}")
i = 0
for f in fits_names:
    img = fits.open(f)
    fits_files.append(img)
    i = i + 1
    print(f"{i}) {f}:")
    for j in range(len(img)):
        hdu = img[j]
        if j == 0:
            print("  Primary HDU:")
        else:
            print(f"  HDU{j}:")
        for attr in attributes:
            if attr in hdu.header:
                print(f"\t    {attr} = {hdu.header[attr]}")
    if "IMAGETYP" in img[0].header and (img[0].header["IMAGETYP"] == "bias" or img[0].header["IMAGETYP"] == "Bias Frame"):
        bias_nums.append(i)
    if "OBJECT" in img[0].header and img[0].header["OBJECT"] == "sunsky":
        flat_nums.append(i)
    if "IMAGETYP" in img[0].header and img[0].header["IMAGETYP"] == "Dark Frame":
        dark_nums.append(i)
print("\n\nWarning: ONLY Primary HDUs will be processed in the next steps!")

print(f"\nFound {len(bias_nums)} BIAS frames: ", end="")
for _ in bias_nums:
    print(f"{_}", end=" ")
print("\nDo you want to select other frames (1) or Continue (press Enter)")
ans = input()
if ans.isdigit():
    if int(ans) == 1:
        print("\n\nSelect BIAS frames:")
        bias_nums = [int(_) for _ in input().split()]

print(f"\nFound {len(dark_nums)} DARK frames: ", end="")
for _ in dark_nums:
    print(f"{_}", end=" ")
print("\nDo you want to select other frames (1) or Select DARK image (2) or Continue (press Enter)")
ans = input()
if ans.isdigit():
    if int(ans) == 1:
        print("\nSelect DARK frames:")
        dark_nums = [int(_) for _ in input().split()]
    if int(ans) == 2:
        print("\nSelect DARK image. This image will be used without preprocessing.")
        have_dark = True
        dark_nums = [int(input())]

print(f"\n\nFound {len(flat_nums)} FLAT frames: ", end="")
for _ in flat_nums:
    print(f"{_}", end=" ")
print("\nDo you want to select other frames (1) or Select FLAT image (2) or Continue (press Enter)")
ans = input()
if ans.isdigit():
    if int(ans) == 1:
        print("\nSelect FLAT frames:")
        flat_nums = [int(_) for _ in input().split()]
    if int(ans) == 2:
        print("\nSelect FLAT image. This image will be used without preprocessing.")
        have_flat = True
        flat_nums = [int(input())]

print("\nSelect frames with object:")
obj_nums = [int(_) for _ in input().split()]
obj_files = [fits_files[i - 1] for i in obj_nums]
if len(obj_nums) == 0:
    print("\nWarning: No frames selected!")
else:
    print("\nSelected files:")
    for _ in obj_nums:
        print(fits_names[_ - 1])

if len(obj_files) > 0:
    shape = obj_files[0][0].data.shape
elif len(bias_nums) > 0:
    shape = fits_files[bias_nums[0]][0].data.shape
elif len(dark_nums) > 0:
    shape = fits_files[dark_nums[0]][0].data.shape
elif len(flat_nums) > 0:
    shape = fits_files[flat_nums[0]][0].data.shape
else:
    shape = (0, 0)
    print("Error: No files selected!")
    exit(7)
for i in range(len(obj_files)):
    if obj_files[i - 1][0].data.shape != shape:
        print(f"Error: Data shape conflict in {fits_names[obj_nums[0]]} and {fits_names[obj_nums[i]]}")
        exit(2)
for i in bias_nums:
    if fits_files[i - 1][0].data.shape != shape:
        print(f"Warning: Shape of BIAS frame in {fits_names[i]} is not equal to the object frame shape! This frame will be ignored.")
        bias_nums.remove(i)
bias_files = [fits_files[i - 1] for i in bias_nums]
for i in dark_nums:
    if fits_files[i - 1][0].data.shape != shape:
        print(f"Warning: Shape of DARK frame in {fits_names[i]} is not equal to the object frame shape! This frame will be ignored.")
        dark_nums.remove(i)
dark_files = [fits_files[i - 1] for i in dark_nums]
for i in flat_nums:
    if fits_files[i - 1][0].data.shape != shape:
        print(f"Warning: Shape of FLAT frame in {fits_names[i]} is not equal to the object frame shape! This frame will be ignored.")
        flat_nums.remove(i)
flat_files = [fits_files[i - 1] for i in flat_nums]

if len(obj_files) > 0:
    exptime = max([img[0].header["EXPTIME"] for img in obj_files])
elif len(dark_files) > 0 or len(flat_files) > 0:
    exptime = max([max([img[0].header["EXPTIME"] for img in dark_files] + [0.0]), max([img[0].header["EXPTIME"] for img in flat_files] + [0.0])])
else:
    exptime = 0.0

if not os.path.exists(args.dir + "/aimp/tmp"):
    os.makedirs(args.dir + "/aimp/tmp")

bias_fits = make_bias(bias_files)
bias_fits.writeto(args.dir + "/aimp/tmp/mean_bias.fits", overwrite=True)

if not have_dark:
    dark_fits = make_dark(dark_files, bias_fits, exptime)
    dark_fits.writeto(args.dir + "/aimp/tmp/mean_dark.fits", overwrite=True)
else:
    if len(dark_files) == 0:
        print("Error: Problem occurred in DARK image")
        exit(3)
    dark_fits = dark_files[0]

if not have_flat:
    flat_fits = make_flat(flat_files, bias_fits, dark_fits, exptime)
    flat_fits.writeto(args.dir + "/aimp/tmp/mean_flat.fits", overwrite=True)
else:
    if len(flat_files) == 0:
        print("Error: Problem occurred in FLAT image")
        exit(4)
    flat_fits = flat_files[0]

if len(obj_files) > 0:
    bias_data = bias_fits[0].data
    dark_data = dark_fits[0].data
    flat_data = flat_fits[0].data
    obj_data = numpy.ndarray(shape=shape, dtype=datatype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if numpy.sum(numpy.where(flat_data < 0.05, 1, 0)) > 0:
            print("Warning: FLAT contains very small elements (<0.05). Corresponding pixels of the result will be NaN")
        obj_images = [numpy.where(flat_data < 0.05, numpy.NaN, ((img[0].data - bias_data) * (exptime / img[0].header["EXPTIME"]) - dark_data) / flat_data) for img in obj_files]
    obj_header = fits.Header()
    obj_header["IMAGETYP"] = "obj"
    obj_header["EXPTIME"] = sum([img[0].header["EXPTIME"] for img in obj_files])
    if "IMSCALE" in obj_files[0][0].header:
        scale = obj_files[0][0].header["IMSCALE"]
    else:
        scale = 1.0
    for img in obj_files:
        if "IMSCALE" in img[0].header and img[0].header["IMSCALE"] != scale:
            print("Error: Image scale must be the same!")
            exit(5)
    obj_header["IMSCALE"] = scale
    for attr in attrs_to_save:
        if attr in obj_files[0][0].header:
            attr0 = obj_files[0][0].header[attr]
            attr_stat = True
        else:
            attr0 = "NONE"
            attr_stat = False
        for img in obj_files:
            if attr in img[0].header and img[0].header[attr] != attr0:
                print(f"Warning: Non equal values of {attr} attribute")
                attr_stat = False
        if attr_stat:
            obj_header[attr] = attr0
    z = []
    for img in obj_files:
        if "Z" in img[0].header:
            z.append(float(img[0].header["Z"]))
    if len(z) > 0:
        obj_header["Z"] = sum(z) / len(z)

    align_images(obj_images)

    numpy.sum(obj_images, axis=0, out=obj_data)

    obj_hdu = fits.PrimaryHDU(data=obj_data, header=obj_header)

    print("Result file:")
    filename = input()
    if not (filename.endswith(".fts") or filename.endswith(".fits") or filename.endswith(".fit")):
        filename = filename + ".fits"
    fits.HDUList([obj_hdu]).writeto(args.dir + "/aimp/" + filename, overwrite=True)
    print(f"Result saved to: {args.dir + '/aimp/' + filename}")
print(f"BIAS, DARK and FLAT frames saved to directory: {args.dir + '/aimp/tmp'}")
