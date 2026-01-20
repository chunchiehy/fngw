import numpy as np
import ot
from utils.load_data import load_candidate_inchi, inchi_to_graph
from time import time
from utils.diffusion import diffuse

from ot.utils import dist

from fngw import fused_network_gromov_wasserstein2

from multiprocessing import Pool
import functools

class FngwEstimator:

    def __init__(self, alpha=0.33, beta=0.33):
        self.n = None
        self.M = None
        self.K = None
        self.k = None
        self.Y = None
        self.Features = None
        self.alpha = alpha
        self.beta = beta
        self.tau = None
        self.w = None
        self.ground_metric = None
        print("alpha: ", self.alpha)
        print("beta: ", self.beta)

    def train(self, K, Y, L):

        self.n = K.shape[0]
        self.M = np.linalg.inv(K + self.n * L * np.eye(self.n))
        self.K = K
        self.Y = Y

    def train_candidate_gwd(self, Y_c, Cs, Features, Es, num_thread=None):

        Cs_cand, Ls_cand, Es_cand, _, _ = Y_c
        # print(len(Es_cand))
        # print(len(Ls_cand))
        #print(Es_cand)
        # print(Ls_cand[0].shape)
        # print(Cs_cand[0].shape)
        # print(Es_cand[0].shape)

        #print("Es_cand", Es_cand)
        n_c = len(Cs_cand)
        n_tr = len(Features)
        D = np.zeros((n_tr, n_c))

        for i in range(n_tr):
            G1s = [[Cs[i].copy(), Features[i][:, : -5].copy(), Es[i].copy()] for _ in range(n_c)]
            G2s = [[Cs_cand[j], Ls_cand[j][:, : -5], Es_cand[j]] for j in range(n_c)]
            if num_thread is None:
                ds = []
                for (G1, G2) in zip(G1s, G2s):
                    d = fngw_distance(G1, G2, alpha=self.alpha, beta=self.beta)
                    ds.append(d)
            
            else:
                pool_func = functools.partial(fngw_distance,
                                            alpha=self.alpha,
                                            beta=self.beta)
                with Pool(num_thread) as p:
                    ds = p.starmap(pool_func, zip(G1s, G2s))
            
            #print(ds)
            D[i] = np.array(ds)

        return D

    def predict(self, K_tr_te, n_bary, Y_te, n_c_max=200, num_thread=None, edge_info='type'):

        n_te = K_tr_te.shape[1]
        A = self.M.dot(K_tr_te)

        # Compute n_te barycenters with n_bary points max for each barycenter
        mean_topk = np.array([0., 0., 0.])
        mean_fgw = np.array([0., 0.])
        n_pred = 0
        error_type = []
        idxs_predict = []
        mfs_predict = []
        for i in range(n_te):

            # predict weights and take greatest weight graphs
            lambdas = A[:, i]
            idxs = np.argsort(lambdas)[-n_bary:]
            lambdas = [A[j, i] for j in idxs]
            Cs = [self.Y[0][idx]for idx in idxs]
            Features = [self.Y[1][idx] for idx in idxs]
            Es = [self.Y[2][idx] for idx in idxs]

            # load candidate
            try:
                #print(Y_te[4][i])
                In = load_candidate_inchi(Y_te[4][i])
            except FileNotFoundError:
                print('FILE NOT FOUND')
                error_type.append('filenotfound')
                continue

            if In == -1:
                error_type.append('In')
                continue
            try:
                Y_c = [inchi_to_graph(inc, edge_info=edge_info) for inc in In]
                Y_c = [[g[0] for g in Y_c], [g[1] for g in Y_c], [g[3] for g in Y_c], [], In]
            except TypeError:
                error_type.append('type')
                continue

            # compute score
            t0 = time()
            n_c = len(In)
            if n_c > n_c_max:
                error_type.append('too big')
                continue
            if n_c < 1:
                error_type.append('too small')
                continue

            idxs_predict.append(i)
            mfs_predict.append(Y_te[4][i])

        mean_fgw = mean_fgw / n_pred
        mean_topk = mean_topk / n_pred * 100
        print(f'n prediction: {n_pred}', flush=True)

        print(error_type)

        return mean_fgw, mean_topk, n_pred

    def post_process(self, C_bary, F_bary, N_edges):

        # most probable atom predicted
        F = np.zeros(F_bary.shape)
        for i in range(F_bary.shape[0]):
            u = np.zeros(F_bary.shape[1])
            a = np.argmax(F_bary[i])
            u[a] = 1
            F[i] = u

        # keep N largest edges
        ind = np.unravel_index(np.argsort(C_bary, axis=None), C_bary.shape)
        ind_max = (ind[0][-2 * N_edges:], ind[1][-2 * N_edges:])
        C = np.zeros(C_bary.shape)
        C[ind_max] = 1

        return C, F

    # def fgw_distance(self, G1, G2):

    #     n1 = len(G1[1])
    #     n2 = len(G2[1])
    #     p1 = ot.unif(n1)
    #     p2 = ot.unif(n2)
    #     loss_fun = 'square_loss'
    #     Y_norms = np.linalg.norm(G1[1], axis=1) ** 2
    #     Z_norms = np.linalg.norm(G2[1], axis=1) ** 2
    #     scalar_products = G1[1].dot(G2[1].T)
    #     M = Y_norms.reshape(-1, 1) + Z_norms.reshape(1, -1) - 2 * scalar_products
    #     d = ot.gromov.fused_gromov_wasserstein2(M, G1[0], G2[0], p1, p2,
    #                                             loss_fun=loss_fun, alpha=self.alpha)

    #     return d
    
    def fngw_distance(self, G1, G2):
        C1 = G1[2]
        F1 = G1[1]
        A1 = G1[0]

        C2 = G2[2]
        F2 = G2[1]
        A2 = G2[0]

        #print(C2)
        #print(C1)
        n1 = C1.shape[0]
        n2 = C2.shape[0]
        p = ot.unif(n1)
        q = ot.unif(n2)
        M = dist(F1, F2)
        fngw_dist, log = fused_network_gromov_wasserstein2(
            M,
            C1,
            C2,
            A1,
            A2,
            p,
            q,
            dist_fun_C='l2_norm',
            dist_fun_A='square_loss',
            alpha=self.alpha,
            beta=self.beta,
            numItermax=100,
            stopThr=1e-5,
            verbose=False,
            log=True)
        return fngw_dist


    # def gw_distance(self, G1, G2):

    #     n1 = len(G1[1])
    #     n2 = len(G2[1])
    #     p1 = ot.unif(n1)
    #     p2 = ot.unif(n2)
    #     loss_fun = 'square_loss'
    #     d = ot.gromov.gromov_wasserstein2(G1[0], G2[0], p1, p2, loss_fun=loss_fun)

    #     return d


def fngw_distance(G1, G2, alpha, beta):
    C1 = G1[2]
    F1 = G1[1]
    A1 = G1[0]

    C2 = G2[2]
    F2 = G2[1]
    A2 = G2[0]

    #print(C2)
    #print(C1)
    n1 = C1.shape[0]
    n2 = C2.shape[0]
    p = ot.unif(n1)
    q = ot.unif(n2)
    M = dist(F1, F2)
    fngw_dist, log = fused_network_gromov_wasserstein2(
        M,
        C1,
        C2,
        A1,
        A2,
        p,
        q,
        dist_fun_C='l2_norm',
        dist_fun_A='square_loss',
        alpha=alpha,
        beta=beta,
        numItermax=100,
        stopThr=1e-5,
        verbose=False,
        log=True)
    return fngw_dist