import argparse
import os
import json
import numpy as np
import multiprocessing as mp
from sklearn.isotonic import IsotonicRegression
from flipnerf_metrics import *

import configs_cal
from eval_ronnie import get_cdf_params


def get_args_parser():
    parser = argparse.ArgumentParser("Calibrating NeRF Uncertainties")
    parser.add_argument(
        "--gin_config",
        default="./configs/blender4.gin",
        help="name of Gin config file for pretrained FlipNeRF model",
    )
    parser.add_argument(
        "--output_dir",
        default="./logs/lego",
        help="directory where to save output",
    )
    parser.add_argument(
        "--exclude_white", action="store_true", help="exclude white background from calibration"
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

# Load config file for the pretrained FlipNeRF model.
config = configs_cal.load_config(args.gin_config, save_config=True)

# Save CDF parameters for test data from FlipNeRF model.
(preds_test, betas_test, mus_test, pis_test, gts_test) = get_cdf_params(
    config
)

preds_test = np.array(preds_test)
betas_test = np.array(betas_test)
mus_test = np.array(mus_test)
pis_test = np.array(pis_test)
gts_test = np.array(gts_test)

# Construct the calibration sets D^{R}, D^{G}, and D^{B}.

# a) Compute expected confidence levels.
p_r = []
p_g = []
p_b = []

for img_ind in range(len(pis_test)):
    expected_cdf_vals = cdf(
        pis_test[img_ind], mus_test[img_ind], betas_test[img_ind], gts_test[img_ind]
    )
    p_r = p_r + list(expected_cdf_vals[:, :, 0].flatten())
    p_g = p_g + list(expected_cdf_vals[:, :, 1].flatten())
    p_b = p_b + list(expected_cdf_vals[:, :, 2].flatten())

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

# Save data.
np.save(args.output_dir + "/" + "p_r.npy", p_r)
np.save(args.output_dir + "/" + "p_g.npy", p_g)
np.save(args.output_dir + "/" + "p_b.npy", p_b)
np.save(args.output_dir + "/" + "hat_p_r.npy", hat_p_r)
np.save(args.output_dir + "/" + "hat_p_g.npy", hat_p_g)
np.save(args.output_dir + "/" + "hat_p_b.npy", hat_p_b)
np.save(args.output_dir + "/" + "preds.npy", preds_test)
np.save(args.output_dir + "/" + "gts.npy", gts_test)

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

# Get uncertainty masks.
cal_uncerts = get_uncerts(mus_test, betas_test, pis_test, A_R, A_G, A_B, True, num_procs=args.num_procs)

uncal_uncerts = get_uncerts(mus_test, betas_test, pis_test, A_R, A_G, A_B, False, num_procs=args.num_procs)

np.save(args.output_dir + "/cal_masks.npy", cal_uncerts.reshape((preds_test.shape[0], preds_test.shape[1], preds_test.shape[2])))
np.save(args.output_dir + "/uncal_masks.npy", uncal_uncerts.reshape((preds_test.shape[0], preds_test.shape[1], preds_test.shape[2])))

create_and_save_fig_rgb(
    p_r,
    hat_p_r,
    p_g,
    hat_p_g,
    p_b,
    hat_p_b,
    args.output_dir + "/cal_curve.png",
)



