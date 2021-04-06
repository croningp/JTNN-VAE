import os
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = ""


import torch
from tqdm import tqdm

from fast_jtnn import JTNNVAE, MolTree, Vocab
from fast_jtnn.datautils import tensorize

cpu = torch.device("cpu")
import numpy as np


class FingerprintCalculator:
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        latent_size: int = 56,
        hidden_size=450,
        depthT=20,
        depthG=20,
    ):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.depthT = depthT
        self.depthG = depthG

        with open(self.vocab_path, "r") as vocab_file:
            self.vocab = Vocab([x.strip() for x in vocab_file])

        self.model = JTNNVAE(
            self.vocab,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            depthT=self.depthT,
            depthG=self.depthG,
        )

        self.model.load_state_dict(
            torch.load(self.model_path, map_location=torch.device("cpu"))
        )

    def __call__(self, smiles_list: List[str]):
        fps = []
        in_vocab = np.ones_like(smiles_list, dtype=bool)

        for chunk in tqdm(range(0, len(smiles_list), 200)):
            tree_batch = [MolTree(s) for s in smiles_list[chunk : chunk + 200]]

            tree_batch, jtenc_holder, mpn_holder = tensorize(
                tree_batch, self.model.vocab, assm=False
            )

            in_vocab[
                [i + chunk for i, tree in enumerate(tree_batch) if tree.out_of_vocab]
            ] = False

            tree_vecs, _, mol_vecs = self.model.encode(jtenc_holder, mpn_holder)
            ts = self.model.T_mean(tree_vecs)
            gs = self.model.G_mean(mol_vecs)

            final = torch.cat([ts, gs], dim=1).data.cpu().numpy()
            fps.append(final)

        fps = np.concatenate(fps, axis=0)
        result = np.zeros((len(smiles_list), self.latent_size))
        result[in_vocab, :] = fps
        return result
