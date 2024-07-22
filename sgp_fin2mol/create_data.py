## Code adapted from
# https://github.com/calvin-zcx/moflow/blob/master/data/smile_to_graph.py
# https://github.com/calvin-zcx/moflow/blob/master/data/transform_qm9.py

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

from chainer_chemistry.dataset.parsers import DataFrameParser
from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms, construct_atomic_number_array, construct_discrete_edge_matrix


class OurPreprocessor(MolPreprocessor):
    """Preprocessor

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
        add_Hs (bool): If True, implicit Hs are added.
        kekulize (bool): If True, Kekulizes the molecule.

    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False,
                 kekulize=False):
        super(OurPreprocessor, self).__init__()
        self.add_Hs = add_Hs
        self.kekulize = kekulize

        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048)

    def get_input_features(self, mol):
        """get input features

        Args:
            mol (Mol):

        Returns:

        """
        type_check_num_atoms(mol, self.max_atoms)
        # Get F: node feature
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        # Get E: edge feature tensor
        adj_array = construct_discrete_edge_matrix(mol, out_size=self.out_size)
        # Get fingerprint
        fp = self.mfpgen.GetFingerprint(mol).ToList()

        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False,
                                            canonical=True)

        return atom_array, adj_array, fp, canonical_smiles

    def prepare_smiles_and_mol(self, mol):
        """Prepare `smiles` and `mol` used in following preprocessing.

        This method is called before `get_input_features` is called, by parser
        class.
        This method may be overriden to support custom `smile`/`mol` extraction

        Args:
            mol (mol): mol instance

        Returns (tuple): (`smiles`, `mol`)
        """
        # Note that smiles expression is not unique.
        # we obtain canonical smiles which is unique in `mol`
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False,
                                            canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)
        if self.add_Hs:
            mol = Chem.AddHs(mol)
        if self.kekulize:
            Chem.Kekulize(mol)
        return canonical_smiles, mol


def one_hot(data, out_size=9, num_max_id=5):
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id))
    # data = data[data > 0]
    # 6 is C: Carbon, we adopt 6:C, 7:N, 8:O, 9:F only. the last place (4) is for padding virtual node.
    indices = np.where(data >= 6, data - 6, num_max_id - 1)
    b[np.arange(out_size), indices] = 1
    # print('[DEBUG] data', data, 'b', b)
    return b


def transform_fn(data):
    """
    :param data: ((9,), (4,9,9), (15,))
    :return:
    """
    node, adj, fp, smiles = data   # node (9,), adj (4,9,9), label (15,)
    # convert to one-hot vector
    node = one_hot(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, fp, smiles


def create_sgp_data(dataset, save_dir, set_name):
    X_fp_all = []
    X_smiles_all = []
    Ys = []
    for example in dataset:
        X_fp = example[2]
        X_smiles = example[3]
        F = example[0]
        E = example[1]
        num_atom = 9 - np.sum((np.argmax(F, axis=1) == 4))
        E = E[:, :num_atom, :num_atom]
        A = 1 - E[3]
        X_fp_all.append(X_fp)
        X_smiles_all.append(X_smiles)
        E = np.transpose(E, (1, 2, 0))
        F = F[:num_atom, :4]
        Ys.append({'F': F, 'E': E, 'A': A})
    np.save(os.path.join(
        save_dir, f'X_fingerprint_{set_name}_qm9.npy'), X_fp_all)
    np.save(os.path.join(
        save_dir, f'X_smiles_{set_name}_qm9.npy'), X_smiles_all)
    np.save(os.path.join(save_dir, f'y_{set_name}_qm9.npy'), Ys)


if __name__ == "__main__":
    print('Preprocessing qm9 data.')
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    df_qm9 = pd.read_csv(os.path.join(data_dir, 'qm9.csv'), index_col=0)

    max_atoms = 9
    preprocessor = OurPreprocessor(out_size=max_atoms, kekulize=True)

    parser = DataFrameParser(preprocessor, smiles_col='SMILES1')
    result = parser.parse(df_qm9, return_smiles=True)

    dataset = result['dataset']
    dataset = [transform_fn(data) for data in dataset]

    train_set, test_set = train_test_split(
        dataset, test_size=2000, random_state=42)

    create_sgp_data(train_set, data_dir, set_name='train')
    create_sgp_data(test_set, data_dir, set_name='test')
