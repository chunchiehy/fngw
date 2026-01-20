import numpy as np
import rdkit
from rdkit import Chem
import pickle as pk
from rdkit.Chem import AllChem
import scipy.io
# from scipy.linalg import issymmetric
import os

from rdkit.Chem.rdchem import BondType, BondStereo


data_path = 'data'


def inchi_to_graph(inc, edge_info='mix'):
    #print(inc)

    m = rdkit.Chem.inchi.MolFromInchi(inc)

    # 1) get molecule graph adjacency
    if m is not None:
        C = Chem.GetAdjacencyMatrix(m)
    else:
        return -1


    # 3) Get molecule bonds (edges) features
    n_atoms = C.shape[0]
    bond_types = list(BondType.values.values())
    i_temps = np.eye(len(bond_types))
    bond_embeds = {}
    for (index, bind_type) in enumerate(bond_types):
        #print(bind_type)
        bond_embeds[bind_type] = i_temps[index]

    none_edge_embed = np.zeros(len(bond_types))

    E = np.ones((n_atoms * n_atoms,
                     1)).dot(np.expand_dims(none_edge_embed, axis=0))
    E = E.reshape((n_atoms, n_atoms, -1))

    for b in m.GetBonds():
        E[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = bond_embeds[b.GetBondType()]
        E[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = bond_embeds[b.GetBondType()]
    
    E_type = E
    
    bond_stereo = list(BondStereo.values.values())
    i_temps = np.eye(len(bond_stereo))
    bond_embeds = {}
    for (index, bind_type) in enumerate(bond_stereo):
        #print(bind_type)
        bond_embeds[bind_type] = i_temps[index]
    none_edge_embed = np.zeros(len(bond_stereo))
    E = np.ones((n_atoms * n_atoms,
                     1)).dot(np.expand_dims(none_edge_embed, axis=0))
    E = E.reshape((n_atoms, n_atoms, -1))
    for b in m.GetBonds():
        E[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = bond_embeds[b.GetStereo()]
        E[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = bond_embeds[b.GetStereo()] # fix bug symetric
    
    E_stereo = E

    if edge_info == 'mix':
        E = np.concatenate((E_type, E_stereo), axis=-1)
    elif edge_info == 'type':
        E = E_type
    elif edge_info == 'stereo':
        E = E_stereo
    else:
        raise ValueError

    #print(issymmetric(np.argmax(E, axis=-1)))

    # 2) compute atom representations
    Atoms = m.GetAtoms()
    F = []
    U = {'C': 0, 'N': 1, 'O': 2, 'Cl': 3, 'S': 4, 'F': 5, 'P': 6, 'I': 7, 'Br': 8, 'Se': 9, 'Si': 10, 'H': 11, 'B': 12}

    I = np.eye(len(U))

    for atom in Atoms:
        #print(atom.GetIdx())

        rep = list(I[U[atom.GetSymbol()]])


        Nei = [a.GetSymbol() for a in atom.GetNeighbors()]
        n_attached_H = Nei.count('H')
        rep.append(n_attached_H)

        n_heavy_neigh = len(Nei) - n_attached_H
        rep.append(n_heavy_neigh)

        rep.append(atom.GetFormalCharge())

        rep.append(int(atom.IsInRing()))

        rep.append(int(atom.GetIsAromatic()))

        F.append(rep)
        #print(rep)


    # compute atoms positions
    AllChem.Compute2DCoords(m)
    P = []
    for c in m.GetConformers():
        P.append(c.GetPositions())
    # print(E_type.shape)
    # print(E_stereo.shape)
    # print(E.shape)
    return C, np.array(F), P, E


def build_spectrum_graph_dataset(edge_info='type', file_name='output_graphs.pickle'):

    # # Order output data according to data_GNPS.tct in order to align input kernel and output graph
    
    mat = scipy.io.loadmat(os.path.join(data_path, 'data_GNPS.mat'))

    In = [inch[0][0] for inch in mat['inchi']]
    Mf = [mf[0][0] for mf in mat['mf']]

    Cs, Fs, Ps, Es = [], [], [], []

    for inc in In:
        C, F, P, E = inchi_to_graph(inc, edge_info)
        Cs.append(C)
        Fs.append(F)
        Es.append(E)
        # Ps.append(P)

    # save
    # Y = [Cs, Fs, Ps]
    Y = [Cs, Fs, Es, In, Mf]
    
    with open(os.path.join(data_path, file_name), 'wb') as handle:
            pk.dump(Y, handle)


def load_dataset_kernel_graph(n, file_name='output_graphs.pickle'):
    K = np.loadtxt(os.path.join(data_path, 'input_kernels/PPKr.txt'))
    with open(os.path.join(data_path, file_name), 'rb') as handle:
        Y = pk.load(handle)
    
    # Divide train/test
    K_tr = K[:n, :n]
    K_tr_te = K[:n, n:]
    K_te_te = K[n:, n:]
    Cs, Fs, Es, In, Mf = Y
    Cs_tr = Cs[: n]
    Cs_te = Cs[n:]
    
    Es_tr = Es[: n]
    Es_te = Es[n:]
    
    Fs_tr = Fs[: n]
    Fs_te = Fs[n:]
    In_tr = In[: n]
    In_te = In[n:]
    Mf_tr = Mf[: n]
    Mf_te = Mf[n:]
    Y_tr = [Cs_tr, Fs_tr, Es_tr, In_tr, Mf_tr]
    Y_te = [Cs_te, Fs_te, Es_te, In_te, Mf_te]

    return [K_tr, Y_tr], [K_tr_te, K_te_te, Y_te]


def load_candidate_inchi(mf):
    
    mat = scipy.io.loadmat(os.path.join(data_path, f'candidates/candidate_set_{mf}.mat'))

    In = [inch[0][0] for inch in mat['inchi']]

    return In