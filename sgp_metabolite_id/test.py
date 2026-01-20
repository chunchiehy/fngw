from time import time
import argparse
from utils.load_data import load_dataset_kernel_graph
from utils.metabolites_utils import center_gram_matrix, normalize_gram_matrix
from methods.method_fngw import FngwEstimator
from utils.diffusion import diffuse

import time


def main(alpha=0.1, beta=0.1, edge_info="type"):
    t_start = time.time()

    print(f"edge info: {edge_info}")
    dataset_kernel_graph = f"spectrum_graph_dataset_bond{edge_info}.pickle"

    # 1) Load data
    n_tr = 3000
    n_te = 1148
    D_tr, D_te = load_dataset_kernel_graph(n_tr, file_name=dataset_kernel_graph)
    K_tr, Y_tr = D_tr
    K_tr_te, K_te_te, Y_te = D_te

    # 2) Input pre-processing
    center, normalize = True, True
    if center:
        K_tr_te = center_gram_matrix(K_tr_te, K_tr, K_tr_te, K_tr)
        K_tr = center_gram_matrix(K_tr)
    if normalize:
        K_tr_te = normalize_gram_matrix(K_tr_te, K_tr, K_te_te)
        K_tr = normalize_gram_matrix(K_tr)

    # 3) Train

    print(f"alpha: {alpha}, beta: {beta}")

    clf = FngwEstimator(alpha=alpha, beta=beta)
    clf.ground_metric = "diffuse"
    L = 1e-4  # kernel ridge regularization parameter
    clf.tau = 0.6  # the bigger tau is the more the neighbor atoms have similar feature. This impact the FGW's ground metric.
    Y_Tr = diffuse(Y_tr, clf.tau)
    clf.train(K_tr, Y_tr, L)

    # 4) Predict and compute the test scores
    n_bary = 5  # Number of kept alpha_i(x) when predicting
    n_c_max = 500000000  # Do not predict test input with more than n_c_max candidates
    fgw, topk, n_pred = clf.predict(
        K_tr_te,
        n_bary=n_bary,
        Y_te=Y_te,
        n_c_max=n_c_max,
        num_thread=28,
        edge_info=edge_info,
    )

    print(f"fgw: {fgw}, topk: {topk}, n_pred: {n_pred}")

    t_end = time.time()

    print(f"Run time: {t_end - t_start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross validation")

    parser.add_argument(
        "--edge_info",
        type=str,
        help="Edge information to use, one of 'type', 'stereo' or 'mix'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
    )
    parser.add_argument(
        "--beta",
        type=float,
    )
    args = parser.parse_args()

    main(alpha=args.alpha, beta=args.beta, edge_info=args.edge_info)
