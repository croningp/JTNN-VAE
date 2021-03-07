import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
import torch
from fast_jtnn import JTNNVAE, Vocab, MolTree
from fast_jtnn.datautils import tensorize
from tqdm import tqdm

cpu = torch.device("cpu")
import numpy as np


def fingerprint(
    smiles_list,
    model_path=Path(__file__).absolute.parents[1]
    / "data"
    / "dario"
    / "vae_model"
    / "model.iter-2000",
    vocab_path=Path(__file__).absolute.parents[1] / "data" / "dario" / "vocab3.txt",
):

    latent_size = 56

    vocab = [x.strip("\r\n ") for x in open(vocab_path, "r")]
    vocab = Vocab(vocab)

    model = JTNNVAE(
        vocab, hidden_size=450, latent_size=latent_size, depthT=20, depthG=20
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model = model.to(cpu)

    fps = []
    all_out_of_vocab = []
    fp_dic = {}

    for chunk in tqdm(range(0, len(smiles_list), 200)):
        tree_batch = [MolTree(s) for s in smiles_list[chunk : chunk + 200]]

        tree_batch, jtenc_holder, mpn_holder = tensorize(
            tree_batch, model.vocab, assm=False
        )

        indexes_out_of_vocab = [
            i for i, tree in enumerate(tree_batch) if tree.out_of_vocab
        ]

        tree_vecs, _, mol_vecs = model.encode(jtenc_holder, mpn_holder)
        ts = model.T_mean(tree_vecs)
        gs = model.G_mean(mol_vecs)

        final = torch.cat([ts, gs], dim=1).data.cpu().numpy()
        fps.extend(final)

        out_of_vocab = [x + chunk for x in indexes_out_of_vocab]
        all_out_of_vocab += out_of_vocab

        torch.cuda.empty_cache()

    vect_ix = 0
    for ix, rgt in enumerate(smiles_list):

        # if mol was out of vocab, fp is an all-zeros vector.
        if ix in all_out_of_vocab:
            fp_dic[rgt] = np.zeros(
                latent_size,
            )
            continue

        vect = fps[vect_ix]
        fp_dic[rgt] = vect
        vect_ix += 1

    print(len(fps))

    return fp_dic
