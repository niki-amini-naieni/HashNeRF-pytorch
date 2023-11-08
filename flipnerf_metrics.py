import numpy as np
from skimage.metrics import structural_similarity
from scipy.misc import derivative
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["agg.path.chunksize"] = 1e9
import multiprocessing as mp
from scipy.interpolate import interp1d
import tensorflow as tf

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


def adjust_for_quantile(ps, cs):
    ps = np.copy(ps)
    cs = np.copy(cs)
    # Assumes ps and cs are sorted in increasing order.
    for ind in range(len(ps)):
        if ind > 0:
            if ps[ind - 1] == ps[ind]:
                cs[ind - 1] = cs[ind]

    (cs, inds) = np.unique(cs, return_index=True)

    return (ps[inds], cs)


def get_uncerts(mus, betas, pis, A_R, A_G, A_B, cal, num_procs=12):

    global get_uncert
    num_mix_comps = mus.shape[-2]
    mus = mus.reshape(-1, num_mix_comps, 3)
    betas = betas.reshape(-1, num_mix_comps, 3)
    pis = pis.reshape(-1, num_mix_comps)
    uncerts = mp.Array("d", np.zeros(pis.shape[0]), lock=False)

    def get_uncert(px_ind):
        mu = mus[px_ind]
        beta = betas[px_ind]
        pi = pis[px_ind]
        cdf_r_uncal = lambda x: cdf(x, mu[:, 0], beta[:, 0], pi
            )
        cdf_g_uncal = lambda x: cdf(x, mu[:, 1], beta[:, 1], pi
            )
        cdf_b_uncal = lambda x: cdf(x, mu[:, 2], beta[:, 2], pi
            )
        if cal:
            cdf_r = lambda x: A_R.predict(cdf_r_uncal(x))
            cdf_g = lambda x: A_G.predict(cdf_g_uncal(x))
            cdf_b = lambda x: A_B.predict(cdf_b_uncal(x))
        else:
            cdf_r = cdf_r_uncal
            cdf_g = cdf_g_uncal
            cdf_b = cdf_b_uncal

        xs = np.linspace(-2, 2, 200).reshape((200, 1))
        ys = cdf_r(xs)
        print(xs.shape)
        print(ys.shape)
        xs = xs[:, 0]
        (ys, xs) = adjust_for_quantile(ys, xs)
        i_cdf_r = interp1d(ys, xs)
        interquart_r = i_cdf_r(0.75) - i_cdf_r(0.25)

        xs = np.linspace(-2, 2, 200).reshape((200, 1))
        ys = cdf_g(xs)
        xs = xs[:, 0]
        (ys, xs) = adjust_for_quantile(ys, xs)
        i_cdf_g = interp1d(ys, xs)
        interquart_g = i_cdf_g(0.75) - i_cdf_g(0.25)

        xs = np.linspace(-2, 2, 200).reshape((200, 1))
        ys = cdf_b(xs)
        xs = xs[:, 0]
        (ys, xs) = adjust_for_quantile(ys, xs)
        i_cdf_b = interp1d(ys, xs)
        interquart_b = i_cdf_b(0.75) - i_cdf_b(0.25)

        uncerts[px_ind] = (interquart_r + interquart_g + interquart_b) / 3

    proc_pool = mp.Pool(num_procs)
    proc_pool.map(get_uncert, range(pis.shape[0]))
    proc_pool.close()
    proc_pool.join()

    return np.array(uncerts)


# From: https://github.com/BayesRays/BayesRays/blob/main/bayesrays/metrics/ause.py
def calc_ause(unc_vec, err_vec, err_type="rmse"):
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    # Sort the error
    err_vec_sorted = np.sort(err_vec)

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
    var_vec_sorted_idxs = np.argsort(unc_vec)
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


def cdf(x, mus, betas, pis):
    return np.sum(
        pis * (0.5 + 0.5 * np.sign(x - mus) * (1 - np.exp(-np.abs(x - mus) / betas)))
    )


def pdf(x, mus, betas, pis):
    return np.sum(pis * (1 / (2 * betas)) * np.exp(-np.abs(x - mus) / betas))


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
    # Make sure tf not using gpu due to memory limits.
    # Set CPU as available physical device
    my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
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
    lpips_fn = load_lpips()
    print("Activate LPIPS calculation with AlexNet.")
    return float(lpips_fn(preds, gts))


def get_avg_err(preds, gts):
    return float(
        compute_avg_error(
            psnr=get_psnr(preds, gts),
            ssim=get_ssim(preds, gts),
            lpips=get_lpips(preds, gts),
        )
    )


# Only available for uncalibrated flipnerf.
def get_nll(gts, mus, betas, pis, num_procs):
    global get_nll_loc
    num_mix_comps = mus.shape[-2]
    gts = gts.reshape(-1, 3)
    mus = mus.reshape(-1, num_mix_comps, 3)
    betas = betas.reshape(-1, num_mix_comps, 3)
    pis = pis.reshape(-1, num_mix_comps)
    log_pdf_vals = mp.Array("d", np.zeros(pis.shape[0]))

    def get_nll_loc(px_ind):
        gt = gts[px_ind]
        mu = mus[px_ind]
        beta = betas[px_ind]
        pi = pis[px_ind]
        log_pdf = np.log(
            pdf(gt[0], mu[:, 0], beta[:, 0], pi)
            * pdf(gt[1], mu[:, 1], beta[:, 1], pi)
            * pdf(gt[2], mu[:, 2], beta[:, 2], pi)
            + EPS
        )
        log_pdf_vals[px_ind] = -log_pdf

    proc_pool = mp.Pool(num_procs)
    proc_pool.map(get_nll_loc, range(pis.shape[0]))
    proc_pool.close()
    proc_pool.join()

    log_pdf_vals = np.array(log_pdf_vals)

    return np.mean(log_pdf_vals)


def get_nll_finite_diff(gts, mus, betas, pis, A_R, A_G, A_B, cal):
    num_mix_comps = mus.shape[-2]
    gts = gts.reshape(-1, 3)
    mus = mus.reshape(-1, num_mix_comps, 3)
    betas = betas.reshape(-1, num_mix_comps, 3)
    pis = pis.reshape(-1, num_mix_comps)
    log_pdf_vals = []
    for px_ind in range(betas.shape[0]):
        gt = gts[px_ind]
        mu = mus[px_ind]
        beta = betas[px_ind]
        pi = pis[px_ind]
        cdf_r_uncal = lambda x: cdf(x, mu[:, 0], beta[:, 0], pi)
        cdf_g_uncal = lambda x: cdf(x, mu[:, 1], beta[:, 1], pi)
        cdf_b_uncal = lambda x: cdf(x, mu[:, 2], beta[:, 2], pi)
        if cal:
            cdf_r = lambda x: A_R.predict([cdf_r_uncal(x)])[0]
            cdf_g = lambda x: A_G.predict([cdf_g_uncal(x)])[0]
            cdf_b = lambda x: A_B.predict([cdf_b_uncal(x)])[0]
        else:
            cdf_r = cdf_r_uncal
            cdf_g = cdf_g_uncal
            cdf_b = cdf_b_uncal

        log_pdf = np.log(
            derivative(cdf_r, gt[0], dx=1e-5)
            * derivative(cdf_g, gt[1], dx=1e-5)
            * derivative(cdf_b, gt[2], dx=1e-5)
            + EPS
        )
        log_pdf_vals.append(-log_pdf)
    return np.mean(log_pdf_vals)


def deriv(A, p):
    f = lambda x: A.predict([x])[0]
    return derivative(f, p, dx=1e-5)


def get_nll_chain_rule(gts, mus, betas, pis, A_R, A_G, A_B, num_procs):
    global get_nll_loc
    num_mix_comps = mus.shape[-2]
    gts = gts.reshape(-1, 3)
    mus = mus.reshape(-1, num_mix_comps, 3)
    betas = betas.reshape(-1, num_mix_comps, 3)
    pis = pis.reshape(-1, num_mix_comps)
    log_pdf_vals = mp.Array("d", np.zeros(pis.shape[0]))

    def get_nll_loc(px_ind):
        gt = gts[px_ind]
        mu = mus[px_ind]
        beta = betas[px_ind]
        pi = pis[px_ind]
        p_r = cdf(gt[0], mu[:, 0], beta[:, 0], pi)
        p_g = cdf(gt[1], mu[:, 1], beta[:, 1], pi)
        p_b = cdf(gt[2], mu[:, 2], beta[:, 2], pi)
        log_pdf = np.log(
            deriv(A_R, p_r)
            * pdf(gt[0], mu[:, 0], beta[:, 0], pi)
            * deriv(A_G, p_g)
            * pdf(gt[1], mu[:, 1], beta[:, 1], pi)
            * deriv(A_B, p_b)
            * pdf(gt[2], mu[:, 2], beta[:, 2], pi)
            + EPS
        )
        log_pdf_vals[px_ind] = -log_pdf

    proc_pool = mp.Pool(num_procs)
    proc_pool.map(get_nll_loc, range(pis.shape[0]))
    proc_pool.close()
    proc_pool.join()

    log_pdf_vals = np.array(log_pdf_vals)

    return np.mean(log_pdf_vals)


def get_cal_err(gts, mus, betas, pis, A_R, A_G, A_B, cal, f_name, num_procs):
    num_mix_comps = mus.shape[-2]
    gts = gts.reshape(-1, 3)
    mus = mus.reshape(-1, num_mix_comps, 3)
    betas = betas.reshape(-1, num_mix_comps, 3)
    pis = pis.reshape(-1, num_mix_comps)
    p_r = []
    p_g = []
    p_b = []

    for px_ind in range(betas.shape[0]):
        gt = gts[px_ind]
        mu = mus[px_ind]
        beta = betas[px_ind]
        pi = pis[px_ind]
        cdf_r_uncal = lambda x: cdf(x, mu[:, 0], beta[:, 0], pi)
        cdf_g_uncal = lambda x: cdf(x, mu[:, 1], beta[:, 1], pi)
        cdf_b_uncal = lambda x: cdf(x, mu[:, 2], beta[:, 2], pi)
        if cal:
            cdf_r = lambda x: A_R.predict([cdf_r_uncal(x)])[0]
            cdf_g = lambda x: A_G.predict([cdf_g_uncal(x)])[0]
            cdf_b = lambda x: A_B.predict([cdf_b_uncal(x)])[0]
        else:
            cdf_r = cdf_r_uncal
            cdf_g = cdf_g_uncal
            cdf_b = cdf_b_uncal
        p_r.append(cdf_r(gt[0]))
        p_g.append(cdf_g(gt[1]))
        p_b.append(cdf_b(gt[2]))

    p_r = np.sort(np.array(p_r))
    p_g = np.sort(np.array(p_g))
    p_b = np.sort(np.array(p_b))
    hat_p_r = get_emp_confs(p_r, num_procs)
    hat_p_g = get_emp_confs(p_g, num_procs)
    hat_p_b = get_emp_confs(p_b, num_procs)

    # Create and save calibration plot.
    create_and_save_fig_rgb(p_r, hat_p_r, p_g, hat_p_g, p_b, hat_p_b, f_name)

    return (
        np.mean((p_r - hat_p_r) ** 2),
        np.mean((p_g - hat_p_g) ** 2),
        np.mean((p_b - hat_p_b) ** 2),
    )


def get_ause(preds, gts, mus, betas, pis, A_R, A_G, A_B, cal, f_name, num_procs):
    squared_errs = (preds - gts) ** 2
    squared_errs = squared_errs.reshape(-1, 3)
    squared_errs = np.mean(
        squared_errs, axis=-1
    )  # Use per-pixel mse over color channels
    uncerts = get_uncerts(mus, betas, pis, A_R, A_G, A_B, cal, num_procs=num_procs)

    ratio_removed, ause_err, ause_err_by_var, ause = calc_ause(uncerts, squared_errs)

    # Plot and save results.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # Plot sparsification curves.
    ax.plot(ratio_removed, ause_err, "g", linewidth=2, linestyle="-")
    ax.plot(ratio_removed, ause_err_by_var, "r", linewidth=2, linestyle="-")
    # Label axes.
    ax.set_xlabel("Ratio Removed")
    ax.set_ylabel("RMSE")
    fig.savefig(f_name)
    plt.close(fig)
    return ause
