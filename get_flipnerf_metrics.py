import argparse
import os
import json
import numpy as np
import multiprocessing as mp
from sklearn.isotonic import IsotonicRegression
from flipnerf_metrics import *

import configs_cal
from eval_cal import get_cdf_params


def get_args_parser():
    parser = argparse.ArgumentParser("Calibrating NeRF Uncertainties")
    parser.add_argument(
        "--scene",
        default="fern",
        help="name of LLFF scene out of {room, fern, flower, fortress, horns, leaves, orchids, trex}",
    )
    parser.add_argument(
        "--disable_lpips", action="store_true", help="disable lpips for memory reasons"
    )
    parser.add_argument(
        "--gin_config",
        default="../mini-project-2/configs/llff_fern_hold_out_rays.gin",
        help="name of Gin config file for pretrained FlipNeRF model",
    )
    parser.add_argument(
        "--cal_model_dir",
        default="../mini-project-2/checkpoints/llff3/fern_hold_out_rays",
        help="directory containing model checkpoint for calibration",
    )
    parser.add_argument(
        "--test_model_dir",
        default="../mini-project-2/checkpoints/llff3/fern",
        help="directory containing model checkpoint for testing",
    )
    parser.add_argument(
        "--data_split_file",
        default="llff_data_splits.json",
        help="name of JSON file with training, test, and calibration splits",
    )
    parser.add_argument(
        "--output_dir",
        default="./logs/fern",
        help="directory where to save output",
    )
    parser.add_argument(
        "--num_procs",
        default=48,
        type=int,
        help="number of processes to use for multiprocessing",
    )

    return parser


def cdf(pi, mu, beta, c):
    """
    [pi]: of shape (image height, image width, # of mixture components)
    => (378, 504, 128)
    [mu]: of shape (image height, image width, # of mixture components, # of channels)
    => (378, 504, 128, 3)
    [beta]: of shape (image height, image width, # of mixture components, # of channels)
    => (378, 504, 128, 3)
    [c]: of shape (image height, image width, # of channels)
    => (378, 504, 3)

    Equation 4 of paper
    """

    num_mix_components = pi.shape[-1]
    num_channels = c.shape[-1]

    # Reshape [pi] and [c] to match the shapes of [mu] and [beta].
    pi = np.stack((pi,) * num_channels, axis=-1)
    c = np.stack((c,) * num_mix_components, axis=-2)

    # Compute the cdf parametrized by [mu] and [beta] at the color [c].
    return np.sum(
        pi
        * ((1 / 2) + (1 / 2) * np.sign(c - mu) * (1 - np.exp(-np.abs(c - mu) / beta))),
        axis=-2,
    )


def get_emp_confs(p, num_procs):
    global get_emp_conf
    hat_p = mp.Array("d", np.zeros(p.shape), lock=False)

    def get_emp_conf(ind):
        hat_p[ind] = np.sum(p <= p[ind]) / len(p)

    proc_pool = mp.Pool(num_procs)
    proc_pool.map(get_emp_conf, range(len(p)))
    proc_pool.close()
    proc_pool.join()

    return np.array(hat_p)


# Parse commandline arguments.
args = get_args_parser()
args = args.parse_args()

# Make output directory for storing data.
os.makedirs(args.output_dir, exist_ok=True)

# Get train, test, and calibration indices for the specified scene.
all_data_splits = json.load(open(args.data_split_file))
scene_data_split = all_data_splits[args.scene]
train_inds = scene_data_split["train"]
test_inds = scene_data_split["test"]
cal_inds = train_inds

# Load config file for the pretrained FlipNeRF model.
config = configs_cal.load_config(args.gin_config, save_config=True)

# Save CDF parameters for test data from FlipNeRF model trained on 3 views.
config.checkpoint_dir = args.test_model_dir
(preds_test, betas_test, mus_test, pis_test, gts_test) = get_cdf_params(
    config, test_inds
)
# Save CDF parameters for calibration data from FlipNeRF model trained on the masked views.
config.checkpoint_dir = args.cal_model_dir
(preds_cal, betas_cal, mus_cal, pis_cal, gts_cal) = get_cdf_params(config, cal_inds)
# Load the hold-out masks.
mask = np.ones(
    (len(gts_cal), gts_cal[0].shape[0], gts_cal[0].shape[1], gts_cal[0].shape[2])
)
patch_size = 63
with open("./train_masks.json") as fp:
    train_masks = json.load(fp)
for ind in range(len(gts_cal)):
    left_corners = train_masks["mask_" + str(ind + 1)]["left_corners"]
    for corner in left_corners:
        mask[
            ind,
            corner[1] : (corner[1] + patch_size),
            corner[0] : (corner[0] + patch_size),
            :,
        ] = 0
np.save("./masked_image.npy", mask * gts_cal)
mask = np.array(mask, dtype=bool)

# Construct the calibration sets D^{R}, D^{G}, and D^{B}.

# a) Compute expected confidence levels.
p_r = []
p_g = []
p_b = []

for img_ind in range(len(cal_inds)):
    expected_cdf_vals = cdf(
        pis_cal[img_ind], mus_cal[img_ind], betas_cal[img_ind], gts_cal[img_ind]
    )
    p_r = p_r + list(expected_cdf_vals[:, :, 0][~mask[img_ind, :, :, 0]])
    p_g = p_g + list(expected_cdf_vals[:, :, 1][~mask[img_ind, :, :, 1]])
    p_b = p_b + list(expected_cdf_vals[:, :, 2][~mask[img_ind, :, :, 2]])

p_r = np.sort(p_r)
p_g = np.sort(p_g)
p_b = np.sort(p_b)

# b) Compute empirical confidence levels.
hat_p_r = get_emp_confs(p_r, args.num_procs)
hat_p_g = get_emp_confs(p_g, args.num_procs)
hat_p_b = get_emp_confs(p_b, args.num_procs)

# Increase precision of data for regression. [IsotonicRegression] casts result to lower-precision type, so the below code ensures the inputs are both of the same, high-precision type.
p_r = p_r.astype(np.float64)
hat_p_r = hat_p_r.astype(np.float64)
p_g = p_g.astype(np.float64)
hat_p_g = hat_p_g.astype(np.float64)
p_b = p_b.astype(np.float64)
hat_p_b = hat_p_b.astype(np.float64)
print("Length of cal. set expected confs:")
print(len(p_r))
print("Length of cal. set empirical confs:")
print(len(hat_p_r))
D_R = (p_r, hat_p_r)
D_G = (p_g, hat_p_g)
D_B = (p_b, hat_p_b)

# Train auxiliary models A^{R}, A^{G}, and A^{B} on calibration sets D^{R}, D^{G}, and D^{B}.
A_R = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
    D_R[0], D_R[1]
)
A_G = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
    D_G[0], D_G[1]
)
A_B = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
    D_B[0], D_B[1]
)

# Compute average image quality metrics.
preds = np.array(preds_test)
betas = np.array(betas_test)
mus = np.array(mus_test)
pis = np.array(pis_test)
gts = np.array(gts_test)

avg_psnr = 0
avg_ssim = 0
avg_lpips = 0
avg_geom_err = 0
for image_ind in range(gts.shape[0]):
    avg_psnr += get_psnr(preds[image_ind], gts[image_ind])
    avg_ssim += get_ssim(preds[image_ind], gts[image_ind])
    if not args.disable_lpips:
        avg_lpips += get_lpips(preds[image_ind], gts[image_ind])
        avg_geom_err += get_avg_err(preds[image_ind], gts[image_ind])

avg_psnr = avg_psnr / gts.shape[0]
avg_ssim = avg_ssim / gts.shape[0]
avg_lpips = avg_lpips / gts.shape[0]
avg_geom_err = avg_geom_err / gts.shape[0]

print("PSNR:")
print(avg_psnr)
print("SSIM:")
print(avg_ssim)
print("LPIPS:")
print(avg_lpips)
print("Geom. Avg. Err.:")
print(avg_geom_err)

# Compute uncertainty metrics.
nll_uncal = get_nll(gts, mus, betas, pis)
print("NLL (Uncal.):")
print(nll_uncal)
nll_cal = get_nll_chain_rule(gts, mus, betas, pis, A_R, A_G, A_B)
print("NLL (Cal.):")
print(nll_cal)
cal_err_uncal = get_cal_err(
    gts,
    mus,
    betas,
    pis,
    A_R,
    A_G,
    A_B,
    False,
    args.output_dir + "/flipnerf-cal-err-uncal.png",
    num_procs=args.num_procs,
)
print("Cal. Err. (Uncal.):")
print(cal_err_uncal)
cal_err_cal = get_cal_err(
    gts,
    mus,
    betas,
    pis,
    A_R,
    A_G,
    A_B,
    True,
    args.output_dir + "/flipnerf-cal-err-cal.png",
    num_procs=args.num_procs,
)
print("Cal. Err. (Cal.):")
print(cal_err_cal)
ause_uncal = get_ause(
    preds,
    gts,
    mus,
    betas,
    pis,
    A_R,
    A_G,
    A_B,
    False,
    args.output_dir + "/flipnerf-sparse-curves-uncal.png",
    num_procs=args.num_procs,
)
print("AUSE (Uncal.):")
print(ause_uncal)
ause_cal = get_ause(
    preds,
    gts,
    mus,
    betas,
    pis,
    A_R,
    A_G,
    A_B,
    True,
    args.output_dir + "/flipnerf-sparse-curves-cal.png",
    num_procs=args.num_procs,
)
print("AUSE (Cal.):")
print(ause_cal)

# Print Summary.
print("SUMMARY")
print()
print("Image Quality:")
print("PSNR:")
print(avg_psnr)
print("SSIM:")
print(avg_ssim)
print("LPIPS:")
print(avg_lpips)
print("Geom. Avg. Err.:")
print(avg_geom_err)
print()
print("Uncertainty:")
print("Cal. Err. (Uncal.):")
print(cal_err_uncal)
print("Cal. Err. (Cal.):")
print(cal_err_cal)
print("AUSE (Uncal.):")
print(ause_uncal)
print("AUSE (Cal.):")
print(ause_cal)
print("NLL (Uncal.):")
print(nll_uncal)
print("NLL (Cal.):")
print(nll_cal)
