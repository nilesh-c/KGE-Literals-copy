import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import kga.op as op
import kga.util as util
from kga.util import inherit_docstrings

from kga.models.base import Model


@inherit_docstrings
class MTKGNN_MovieLens(Model):
    """
    MT-KGNN: Multi-Task Knowledge Graph Neural Network
    --------------------------------------------------
    Tay, Yi, et al. "Multi-task Neural Network for Non-discrete Attribute Prediction in Knowledge Graphs." CIKM 2017.
    """

    def __init__(self, n_usr, n_mov, n_rat, n_lit_usr, n_lit_mov, k, h_dim, lam, gpu=False):
        """
        MT-KGNN: Multi-Task Knowledge Graph Neural Network
        --------------------------------------------------

        Params:
        -------
            n_usr: int
                Number of users in dataset.

            n_rat: int
                Number of ratings in dataset.

            n_mov: int
                Number of movies in dataset.

            n_lit_usr: int
                Number of user's attributes.

            n_lit_mov: int
                Number of movie's atrributes.

            k: int
                Embedding size.

            h_dim: int
                Size of hidden layer.

            lam: float
                Prior strength of the embeddings. Used to constaint the
                embedding norms inside a (euclidean) unit ball. The prior is
                Gaussian, this param is the precision.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(MTKGNN_MovieLens, self).__init__(gpu)

        # Hyperparams
        self.n_usr = n_usr
        self.n_rat = n_rat
        self.n_mov = n_mov
        self.n_lit_usr = n_lit_usr
        self.n_lit_mov = n_lit_mov
        self.k = k
        self.h_dim = h_dim
        self.lam = lam

        # Embeddings for E, R
        self.emb_usr = nn.Embedding(self.n_usr, self.k)
        self.emb_rat = nn.Embedding(self.n_rat, self.k)
        self.emb_mov = nn.Embedding(self.n_mov, self.k)
        # Embeddings for A
        self.emb_lit_usr = nn.Embedding(self.n_lit_usr, self.k)
        self.emb_lit_mov = nn.Embedding(self.n_lit_mov, self.k)

        # Nets
        # ----
        # ER-MLP
        self.ermlp = nn.Sequential(
            nn.Linear(3*k, h_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_dim, 1),
        )

        # SA-MLP: subject-attribute net
        self.usr_mlp = nn.Sequential(
            nn.Linear(2*k, h_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_dim, 1)
        )

        # OA-MLP: object-attribute net
        self.mov_mlp = nn.Sequential(
            nn.Linear(2*k, h_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_dim, 1)
        )

        self.embeddings = [self.emb_usr, self.emb_rat, self.emb_mov,
                           self.emb_lit_usr, self.emb_lit_mov]
        self.initialize_embeddings()

        # Xavier init for all nets
        for net in [self.ermlp, self.usr_mlp, self.mov_mlp]:
            for p in net.modules():
                if isinstance(p, nn.Linear):
                    in_dim = p.weight.size(0)
                    p.weight.data.normal_(0, 1/np.sqrt(in_dim/2))

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X, usr_attrs=None, mov_attrs=None):
        """
        Params:
        -------
        X: Mx3 (s, p, o) triples.
        usr_attrs: M vector of index of attributes of subjects.
        mov_attrs: M vector of index of attributes of objects.

        Returns:
        --------
        y_er: score for link-pred task.
        y_sa: score for subject literal-pred task.
        y_oa: score for object literal-pred task.
        """
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        s, p, o = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_usr = self.emb_usr(s)
        e_rat = self.emb_rat(p)
        e_mov = self.emb_mov(o)

        # Forward ER-MLP
        phi_er = torch.cat([e_usr, e_rat, e_mov], 1)  # M x 3k
        y_er = self.ermlp(phi_er)

        if usr_attrs is not None or mov_attrs is not None:
            usr_attrs = Variable(torch.from_numpy(usr_attrs))
            usr_attrs = usr_attrs.cuda() if self.gpu else usr_attrs
            mov_attrs = Variable(torch.from_numpy(mov_attrs))
            mov_attrs = mov_attrs.cuda() if self.gpu else mov_attrs

            e_lit_usr = self.emb_lit_usr(usr_attrs)
            e_lit_mov = self.emb_lit_mov(mov_attrs)
            # Forward SA-MLP
            phi_lit_usr = torch.cat([e_usr, e_lit_usr], 1)  # M x 2k
            y_lit_usr = self.usr_mlp(phi_lit_usr)

            # Forward OA-MLP
            phi_lit_mov = torch.cat([e_mov, e_lit_mov], 1)  # M x 2k
            y_lit_mov = self.mov_mlp(phi_lit_mov)

            return y_er.view(-1, 1), y_lit_usr.view(-1, 1), y_lit_mov.view(-1, 1)
        else:
            return y_er.view(-1, 1)


@inherit_docstrings
class MTKGNN_YAGO(Model):
    """
    MT-KGNN: Multi-Task Knowledge Graph Neural Network
    --------------------------------------------------
    Tay, Yi, et al. "Multi-task Neural Network for Non-discrete Attribute Prediction in Knowledge Graphs." CIKM 2017.
    """

    def __init__(self, n_ent, n_rel, n_lit, k, h_dim, gpu=False):
        """
        MT-KGNN: Multi-Task Knowledge Graph Neural Network
        --------------------------------------------------

        Params:
        -------
            n_usr: int
                Number of users in dataset.

            n_rat: int
                Number of ratings in dataset.

            n_mov: int
                Number of movies in dataset.

            n_lit_usr: int
                Number of user's attributes.

            n_lit_mov: int
                Number of movie's atrributes.

            k: int
                Embedding size.

            h_dim: int
                Size of hidden layer.

            p: float
                Dropout rate.

            lam: float
                Prior strength of the embeddings. Used to constaint the
                embedding norms inside a (euclidean) unit ball. The prior is
                Gaussian, this param is the precision.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(MTKGNN_YAGO, self).__init__(gpu)

        # Hyperparams
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.n_lit = n_lit
        self.k = k
        self.h_dim = h_dim

        # Embeddings for E, R
        self.emb_ent = nn.Embedding(self.n_ent, self.k)
        self.emb_rel = nn.Embedding(self.n_rel, self.k)
        # Embeddings for A
        self.emb_lit = nn.Embedding(self.n_lit, self.k)

        # Nets
        # ----
        # ER-MLP
        self.ermlp = nn.Sequential(
            nn.Linear(3*k, h_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_dim, 1),
        )

        # SA-MLP: subject-attribute net
        self.sa_mlp = nn.Sequential(
            nn.Linear(2*k, h_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_dim, 1)
        )

        # OA-MLP: object-attribute net
        self.oa_mlp = nn.Sequential(
            nn.Linear(2*k, h_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_dim, 1)
        )

        self.embeddings = [self.emb_ent, self.emb_rel, self.emb_lit]
        self.initialize_embeddings()

        # Xavier init for all nets
        for net in [self.ermlp, self.sa_mlp, self.oa_mlp]:
            for p in net.modules():
                if isinstance(p, nn.Linear):
                    in_dim = p.weight.size(0)
                    p.weight.data.normal_(0, 1/np.sqrt(in_dim/2))

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X, s_attrs=None, o_attrs=None):
        """
        Params:
        -------
        X: Mx3 (s, p, o) triples.
        s_attrs: M vector of index of attributes of subjects.
        o_attrs: M vector of index of attributes of objects.

        Returns:
        --------
        y_er: score for link-pred task.
        y_sa: score for subject literal-pred task.
        y_oa: score for object literal-pred task.
        """
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        s, p, o = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_s = self.emb_ent(s)
        e_r = self.emb_rel(p)
        e_o = self.emb_ent(o)

        # Forward ER-MLP
        phi_er = torch.cat([e_s, e_r, e_o], 1)  # M x 3k
        y_er = self.ermlp(phi_er)

        if s_attrs is not None or o_attrs is not None:
            s_attrs = Variable(torch.from_numpy(s_attrs))
            s_attrs = s_attrs.cuda() if self.gpu else s_attrs
            o_attrs = Variable(torch.from_numpy(o_attrs))
            o_attrs = o_attrs.cuda() if self.gpu else o_attrs

            e_lit_s = self.emb_lit(s_attrs)
            e_lit_o = self.emb_lit(o_attrs)

            # Forward SA-MLP
            phi_lit_s = torch.cat([e_s, e_lit_s], 1)  # M x 2k
            y_lit_s = self.sa_mlp(phi_lit_s)

            # Forward OA-MLP
            phi_lit_o = torch.cat([e_o, e_lit_o], 1)  # M x 2k
            y_lit_o = self.oa_mlp(phi_lit_o)

            return y_er.view(-1, 1), y_lit_s.view(-1, 1), y_lit_o.view(-1, 1)
        else:
            return y_er.view(-1, 1)

    def predict_all(self, X):
        """
        Let X be a triple (s, p, o), i.e. tensor of 1x3, return two lists:
            - list of (s, p, all_others)
            - list of (all_others, p, o)
        """
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        s, p, o = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_s = self.emb_ent(s)
        e_r = self.emb_rel(p)
        e_o = self.emb_ent(o)

        # Predict o
        e_s_rep = e_s.repeat(self.n_ent, 1)  # n_ent x k
        e_r_rep = e_r.repeat(self.n_ent, 1)  # n_ent x k
        phi_o = torch.cat([e_s_rep, e_r_rep, self.emb_ent.weight], 1)  # n_ent x 3k
        y_o = self.ermlp(phi_o).view(-1)

        # Predict s
        e_o_rep = e_o.repeat(self.n_ent, 1)
        phi_s = torch.cat([self.emb_ent.weight, e_r_rep, e_o_rep], 1)  # n_ent x 3k
        y_s = self.ermlp(phi_s).view(-1)

        return y_s, y_o
