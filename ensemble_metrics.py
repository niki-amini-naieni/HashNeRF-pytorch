import numpy as np
from skimage.metrics import structural_similarity
from scipy.misc import derivative
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing as mp
import math

EPS = 1e-12

"""
- preds, gts, and other inputs are the aggregated ones for the whole ensemble of M models of shape (# test images x H x W x C), C is 3 for preds and gts but 1 for acc
- assume everything is given to you as numpy arrays on cpu with last channel color
"""


def create_and_save_fig_rgb(
    p_r,
    hat_p_r,
    p_g,
    hat_p_g,
    p_b,
    hat_p_b,
    f_name,
):
    # Set up axis settings
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)

    # Plot uncalibrated expected and empirical confidence levels.
    ax.plot(p_r, hat_p_r, "r", linewidth=2, linestyle="-")
    ax.plot(p_g + 0.4, hat_p_g, "g", linewidth=2, linestyle="-")
    ax.plot(p_b + 0.8, hat_p_b, "b", linewidth=2, linestyle="-")

    # Plot perfectly calibrated lines.
    ax.plot(
        np.linspace(0, 1, 5),
        np.linspace(0, 1, 5),
        linewidth=2,
        linestyle="--",
        color=(0, 0, 0),
        alpha=0.5,
    )
    ax.plot(
        np.linspace(0.4, 1.4, 5),
        np.linspace(0, 1, 5),
        linewidth=2,
        linestyle="--",
        color=(0, 0, 0),
        alpha=0.5,
    )
    ax.plot(
        np.linspace(0.8, 1.8, 5),
        np.linspace(0, 1, 5),
        linewidth=2,
        linestyle="--",
        color=(0, 0, 0),
        alpha=0.5,
    )

    # Label axes.
    ax.set_xlabel("Expected Confidence Level")
    ax.set_ylabel("Observed Confidence Level")
    fig.savefig(f_name)
    plt.close(fig)


def get_emp_confs(p, num_procs=12):
    global get_emp_conf
    hat_p = mp.Array("d", np.zeros(p.shape), lock=False)

    def get_emp_conf(ind):
        hat_p[ind] = np.sum(p <= p[ind]) / len(p)

    proc_pool = mp.Pool(num_procs)
    proc_pool.map(get_emp_conf, range(len(p)))
    proc_pool.close()
    proc_pool.join()

    return np.array(hat_p)


# From: https://github.com/BayesRays/BayesRays/blob/main/bayesrays/metrics/ause.py
def calc_ause(unc_vec, err_vec, err_type="rmse"):
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    # Sort the error
    err_vec_sorted, _ = np.sort(err_vec)

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ause_err = []
    for r in ratio_removed:
        err_slice = err_vec_sorted[0 : int((1 - r) * n_valid_pixels)]
        if err_type == "rmse":
            ause_err.append(np.sqrt(err_slice.mean()))
        elif err_type == "mae" or err_type == "mse":
            ause_err.append(err_slice.mean())

    ###########################################

    # Sort by variance
    _, var_vec_sorted_idxs = np.sort(unc_vec)
    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]
    ause_err_by_var = []
    for r in ratio_removed:

        err_slice = err_vec_sorted_by_var[0 : int((1 - r) * n_valid_pixels)]
        if err_type == "rmse":
            ause_err_by_var.append(np.sqrt(err_slice.mean()))
        elif err_type == "mae" or err_type == "mse":
            ause_err_by_var.append(err_slice.mean())

    ause_err = np.array(ause_err)
    ause_err_by_var = np.array(ause_err_by_var)

    ause = np.trapz(ause_err_by_var - ause_err, ratio_removed)
    return ratio_removed, ause_err, ause_err_by_var, ause


def norm_cdf(x, mean, var):
    return (1 / 2) * (1 + math.erf((x - mean) / np.sqrt(2 * var)))


def mse_to_psnr(mse):
    """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
    return -10.0 / np.log(10.0) * np.log(mse)


def psnr_to_mse(psnr):
    """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
    return np.exp(-0.1 * np.log(10.0) * psnr)


def ssim_fn(x, y):
    return structural_similarity(
        x,
        y,
        multichannel=True,
        data_range=1.0,
        win_size=11,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        K1=0.01,
        K2=0.03,
    )


def load_lpips():
    graph = tf.compat.v1.Graph()
    session = tf.compat.v1.Session(graph=graph)
    with graph.as_default():
        input1 = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
        input2 = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
        with tf.compat.v1.gfile.Open("alex_net.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

            target = tf.compat.v1.transpose(
                (input1[tf.compat.v1.newaxis] * 2.0) - 1.0, [0, 3, 1, 2]
            )
            pred = tf.compat.v1.transpose(
                (input2[tf.compat.v1.newaxis] * 2.0) - 1.0, [0, 3, 1, 2]
            )
            tf.compat.v1.import_graph_def(
                graph_def, input_map={"0:0": target, "1:0": pred}
            )
            distance = graph.get_operations()[-1].outputs[0]

    def lpips_distance(img1, img2):
        with graph.as_default():
            return session.run(distance, {input1: img1, input2: img2})[0, 0, 0, 0]

    return lpips_distance


lpips_fn = load_lpips()
print("Activate LPIPS calculation with AlexNet.")


def compute_avg_error(psnr, ssim, lpips):
    """The 'average' error used in the paper."""
    mse = psnr_to_mse(psnr)
    dssim = np.sqrt(1 - ssim)
    return np.exp(np.mean(np.log(np.array([mse, dssim, lpips]))))


def get_psnr(preds, gts):
    return float(mse_to_psnr(((preds - gts) ** 2).mean()))


def get_ssim(preds, gts):
    return float(ssim_fn(preds, gts))


def get_lpips(preds, gts):
    return float(lpips_fn(preds, gts))


def get_avg_err(preds, gts):
    return float(
        compute_avg_error(
            psnr=get_psnr(preds, gts),
            ssim=get_ssim(preds, gts),
            lpips=get_lpips(preds, gts),
        )
    )


def get_nll(preds, gts, vars, accs, add_epistem):
    gts = gts.reshape(-1, 3)
    preds = preds.reshape(-1, 3)
    vars = vars.reshape(-1, 3)
    accs = accs.reshape(-1)
    log_pdf_vals = []
    for px_ind in range(vars.shape[0]):
        gt = gts[px_ind]
        mu = preds[px_ind]
        if add_epistem:
            var = np.mean(vars[px_ind], axis=-1)
            epistem = (1 - accs[px_ind]) ** 2
            var = var + epistem
        else:
            var = vars[px_ind]

        log_pdf = np.log(
            (np.exp(-0.5 * (gt - mu) ** 2 / var) / np.sqrt(var * 2.0 * np.pi)).prod()
            + EPS
        )
        log_pdf_vals.append(-log_pdf)
    return np.mean(log_pdf_vals)


def get_nll_finite_diff(preds, gts, vars, accs, add_epistem):
    gts = gts.reshape(-1, 3)
    preds = preds.reshape(-1, 3)
    vars = vars.reshape(-1, 3)
    accs = accs.reshape(-1)
    log_pdf_vals = []
    for px_ind in range(vars.shape[0]):
        gt = gts[px_ind]
        mu = preds[px_ind]
        if add_epistem:
            var = np.mean(vars[px_ind], axis=-1)
            epistem = (1 - accs[px_ind]) ** 2
            var = var + epistem
            cdf_r = lambda r: norm_cdf(r, mu[0], var)
            cdf_g = lambda g: norm_cdf(g, mu[1], var)
            cdf_b = lambda b: norm_cdf(b, mu[2], var)
        else:
            var = vars[px_ind]
            cdf_r = lambda r: norm_cdf(r, mu[0], var[0])
            cdf_g = lambda g: norm_cdf(g, mu[1], var[1])
            cdf_b = lambda b: norm_cdf(b, mu[2], var[2])

        log_pdf = np.log(
            derivative(cdf_r, gt[0], dx=1e-6)
            * derivative(cdf_g, gt[1], dx=1e-6)
            * derivative(cdf_b, gt[2], dx=1e-6)
            + EPS
        )

        log_pdf_vals.append(-log_pdf)
    return np.mean(log_pdf_vals)


def get_cal_err(preds, gts, vars, accs, add_epistem, f_name):
    preds = preds.reshape(-1, 3)
    gts = gts.reshape(-1, 3)
    vars = vars.reshape(-1, 3)
    accs = accs.reshape(-1, 3)
    p_r = []
    p_g = []
    p_b = []

    for px_ind in range(vars.shape[0]):
        gt = gts[px_ind]
        mu = preds[px_ind]
        if add_epistem:
            var = np.mean(vars[px_ind], axis=-1)
            epistem = (1 - accs[px_ind]) ** 2
            var = var + epistem
            p_r.append(norm_cdf(gt[0], mu[0], var))
            p_g.append(norm_cdf(gt[1], mu[1], var))
            p_b.append(norm_cdf(gt[2], mu[2], var))
        else:
            var = vars[px_ind]
            p_r.append(norm_cdf(gt[0], mu[0], var[0]))
            p_g.append(norm_cdf(gt[1], mu[1], var[1]))
            p_b.append(norm_cdf(gt[2], mu[2], var[2]))

    hat_p_r = get_emp_confs(p_r)
    hat_p_g = get_emp_confs(p_g)
    hat_p_b = get_emp_confs(p_b)

    # Create and save calibration plot.
    create_and_save_fig_rgb(p_r, hat_p_r, p_g, hat_p_g, p_b, hat_p_b, f_name)

    return (
        np.mean((p_r - hat_p_r) ** 2),
        np.mean((p_g - hat_p_g) ** 2),
        np.mean((p_b - hat_p_b) ** 2),
    )


def get_ause(preds, gts, vars, accs, add_epistem):
    squared_errs = (preds - gts) ** 2
    squared_errs = squared_errs.reshape(-1, 3)
    squared_errs = np.mean(
        squared_errs, axis=-1
    )  # Use per-pixel mse over color channels
    vars = np.mean(vars.reshape(-1, 3), axis=-1)
    accs = accs.reshape(-1)
    if add_epistem:
        epistem = (1 - accs) ** 2
        vars = vars + epistem

    _, _, _, ause = calc_ause(vars, squared_errs)
    return ause
