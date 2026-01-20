import sys
import numpy as np
from time import time
from multiprocessing import Pool
import pandas as pd


from methods.method_fngw import FngwEstimator
from utils.metabolites_utils import center_gram_matrix, normalize_gram_matrix
from utils.load_data import load_dataset_kernel_graph
from utils.diffusion import diffuse
import argparse


def exp_fngw_diffuse(
    n_tr,
    n_val,
    L,
    tau,
    alpha,
    beta,
    n_bary,
    n_c_max,
    num_thread,
    dataset_file_name,
    edge_info,
):

    # Load data
    t0 = time()
    D_tr, D_te = load_dataset_kernel_graph(n_tr - n_val, file_name=dataset_file_name)
    K, Y = D_tr
    K_tr_te, K_te_te, Y_te = D_te
    n = K_tr_te.shape[0]
    K_tr_te, K_te_te = K_tr_te[:, :n_val], K_te_te[:n_val, :n_val]
    Y_te = [
        Y_te[0][:n_val],
        Y_te[1][:n_val],
        Y_te[2][:n_val],
        Y_te[3][:n_val],
        Y_te[4][:n_val],
    ]
    print(f"Load time: {time() - t0}", flush=True)

    # Input pre-processing
    t0 = time()
    center, normalize = True, True
    if center:
        K_tr_te = center_gram_matrix(K_tr_te, K, K_tr_te, K)
        K = center_gram_matrix(K)
    if normalize:
        K_tr_te = normalize_gram_matrix(K_tr_te, K, K_te_te)
        K = normalize_gram_matrix(K)
    print(f"Pre-processing time: {time() - t0}", flush=True)

    # Train
    t0 = time()
    clf = FngwEstimator(alpha=alpha, beta=beta)
    clf.ground_metric = "diffuse"
    clf.tau = tau
    Y = diffuse(Y, clf.tau)
    clf.train(K, Y, L)
    print(f"Train time: {time() - t0}", flush=True)

    # Predict
    t0 = time()
    fgw, topk, n_pred = clf.predict(
        K_tr_te,
        n_bary=n_bary,
        Y_te=Y_te,
        n_c_max=n_c_max,
        num_thread=num_thread,
        edge_info=edge_info,
    )
    print(f"Test time: {time() - t0}", flush=True)

    print(
        f"{(n_tr, n_val, L, tau, alpha, beta, n_bary, n_c_max)}, mean fgw : {fgw}, topk = {topk}",
        flush=True,
    )

    return fgw[0], topk, n, n_pred


def main(edge_info="mix"):
    # Selection of the hyperparameters by taking the ones with the best validation scores
    n_tr, n_val = 3000, 600  # 3000 - 600 = 2400 train / 600 val
    # n_tr, n_val = 1000, 3 # debug
    n_c_max = 1e6  # do not consider test points with more than n_c_max candidates

    # Define the grids of hyper-parameters
    L = 1e-4
    tau = 0.6
    n_bary = 5

    num_thread = 28
    dataset_file_name = f"spectrum_graph_dataset_bond{edge_info}.pickle"
    # edge_info = 'mix'

    print(edge_info)

    path = f"exper_res/fngw_cv_{edge_info}.csv"

    exp = exp_fngw_diffuse

    alphas = [1e-1, 0.33, 0.5, 0.67]
    betas = [1e-1, 0.33, 0.5, 0.67]

    alpha_betas = []
    for alpha in alphas:
        for beta in betas:
            if alpha + beta < 1:
                alpha_betas.append((alpha, beta))

    # alpha_betas = [(0.5, 0.33), (0.67, 0.1)] #for test
    print(alpha_betas)
    Sfngw = np.zeros(len(alpha_betas))
    Stopk = np.zeros((len(alpha_betas), 3))

    Larg = []

    for alpha, beta in alpha_betas:
        Larg.append(
            (
                n_tr,
                n_val,
                L,
                tau,
                alpha,
                beta,
                n_bary,
                n_c_max,
                num_thread,
                dataset_file_name,
                edge_info,
            )
        )

    # n_pool = len(Larg)

    t0 = time()

    R = []
    for arg in Larg:
        R.append(exp(*arg))

    # if __name__ == '__main__':
    #     with Pool(n_pool) as p:
    #         R = p.starmap(exp, Larg)

    for i, res in enumerate(R):
        fngw, topk, n_train, n_pred = res
        Sfngw[i] = fngw
        Stopk[i, 0] = topk[0]
        Stopk[i, 1] = topk[1]
        Stopk[i, 2] = topk[2]

    alphas = [item[0] for item in alpha_betas]
    betas = [item[1] for item in alpha_betas]

    res_data = {
        "alpha": alphas,
        "beta": betas,
        "fngw": Sfngw,
        "Top-1": Stopk[:, 0],
        "Top-10": Stopk[:, 1],
        "Top-20": Stopk[:, 2],
    }

    print(f"selection time (with multiprocessing): {time() - t0}")

    df = pd.DataFrame(data=res_data)

    df.to_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross validation")

    parser.add_argument(
        "--edge_info",
        type=str,
        help="Edge information to use, one of 'type', 'stereo' or 'mix'",
    )
    args = parser.parse_args()
    main(args.edge_info)
