import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import kga.op as op
import kga.util as util
from kga.util import inherit_docstrings
import pdb


class Model(nn.Module):
    """
    Base class of all models
    """

    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu
        self.embeddings = []

    def forward(self, X):
        """
        Given a (mini)batch of triplets X of size M, predict the validity.

        Params:
        -------
        X: int matrix of M x 3, where M is the (mini)batch size
            First row contains index of head entities.
            Second row contains index of relationships.
            Third row contains index of tail entities.

        Returns:
        --------
        y: Mx1 vectors
            Contains the probs result of each M data.
        """
        raise NotImplementedError

    def predict(self, X, sigmoid=False):
        """
        Predict the score of test batch.

        Params:
        -------
        X: int matrix of M x 3, where M is the (mini)batch size
            First row contains index of head entities.
            Second row contains index of relationships.
            Third row contains index of tail entities.

        sigmoid: bool, default: False
            Whether to apply sigmoid at the prediction or not. Useful if the
            predicted result is scores/logits.

        Returns:
        --------
        y_pred: np.array of Mx1
        """
        y_pred = self.forward(X).view(-1, 1)

        if sigmoid:
            y_pred = F.sigmoid(y_pred)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()

    def log_loss(self, y_pred, y_true, average=True):
        """
        Compute log loss (Bernoulli NLL).

        Params:
        -------
        y_pred: vector of size Mx1
            Contains prediction logits.

        y_true: np.array of size Mx1 (binary)
            Contains the true labels.

        average: bool, default: True
            Whether to average the loss or just summing it.

        Returns:
        --------
        loss: float
        """
        if self.gpu:
            y_true = Variable(torch.from_numpy(y_true.astype(np.float32)).cuda())
        else:
            y_true = Variable(torch.from_numpy(y_true.astype(np.float32)))

        nll = F.binary_cross_entropy_with_logits(y_pred, y_true, size_average=average)

        norm_E = torch.norm(self.emb_E.weight, 2, 1)
        norm_R = torch.norm(self.emb_R.weight, 2, 1)

        # Penalize when embeddings norms larger than one
        nlp1 = torch.sum(torch.clamp(norm_E - 1, min=0))
        nlp2 = torch.sum(torch.clamp(norm_R - 1, min=0))

        if average:
            nlp1 /= nlp1.size(0)
            nlp2 /= nlp2.size(0)

        return nll + self.lam*nlp1 + self.lam*nlp2

    def ranking_loss(self, y_pos, y_neg, margin=1, C=1, average=True):
        """
        Compute loss max margin ranking loss.

        Params:
        -------
        y_pos: vector of size Mx1
            Contains scores for positive samples.

        y_neg: np.array of size Mx1 (binary)
            Contains the true labels.

        margin: float, default: 1
            Margin used for the loss.

        C: int, default: 1
            Number of negative samples per positive sample.

        average: bool, default: True
            Whether to average the loss or just summing it.

        Returns:
        --------
        loss: float
        """
        M = y_pos.size(0)

        y_pos = y_pos.view(-1).repeat(C)  # repeat to match y_neg
        y_neg = y_neg.view(-1)

        # target = [-1, -1, ..., -1], i.e. y_neg should be higher than y_pos
        target = -np.ones(M*C, dtype=np.float32)

        if self.gpu:
            target = Variable(torch.from_numpy(target).cuda())
        else:
            target = Variable(torch.from_numpy(target))

        loss = F.margin_ranking_loss(
            y_pos, y_neg, target, margin=margin, size_average=average
        )

        return loss

    def normalize_embeddings(self):
        for e in self.embeddings:
            e.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def initialize_embeddings(self):
        r = 6/np.sqrt(self.k)

        for e in self.embeddings:
            e.weight.data.uniform_(-r, r)

        self.normalize_embeddings()


@inherit_docstrings
class ERLMLP(Model):
    """
    ERL-MLP: Entity-Relation-Literal MLP for generic KG
    ---------------------------------------------------
    """

    def __init__(self, n_ent, n_rel, n_lit, k, h_dim, gpu=False, img_lit=False, txt_lit=False):
        """
        ERL-MLP: Entity-Relation-Literal MLP for generic KG
        ---------------------------------------------------

        Params:
        -------
            n_ent: int
                Number of entities in dataset.

            n_rel: int
                Number of relationships in dataset.

            n_lit: int
                Number of attributes/literals in dataset.

            k: int
                Embedding size for entity and relationship.

            l: int
                Size of projected attributes/literals.

            h_dim: int
                Size of hidden layer.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(ERLMLP, self).__init__(gpu)

        # Hyperparams
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.n_lit = n_lit
        self.k = k
        self.h_dim = h_dim
        self.img_lit = img_lit
        self.txt_lit = txt_lit

        # Nets
        self.emb_ent = nn.Embedding(n_ent, k)
        self.emb_rel = nn.Embedding(n_rel, k)

        self.emb_img = nn.Linear(512, self.k)
        self.emb_txt = nn.Linear(384, self.k)

        # Image embeddings
        if img_lit and txt_lit:
            n_input = 7*k+2*n_lit  # 3k + 2k + 2k
        elif img_lit or txt_lit:
            n_input = 5*k+2*n_lit  # 3k + 2k
        else:
            n_input = 3*k+2*n_lit

        self.mlp = nn.Sequential(
            nn.Linear(n_input, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

        self.embeddings = [self.emb_ent, self.emb_rel]
        self.initialize_embeddings()

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X, X_lit_s, X_lit_o, X_lit_s_img, X_lit_o_img, X_lit_s_txt, X_lit_o_txt):
        """
        Given a (mini)batch of triplets X of size M, predict the validity.

        Params:
        -------
        X: int matrix of M x 3, where M is the (mini)batch size
            First column contains index of head entities.
            Second column contains index of relationships.
            Third column contains index of tail entities.

        X_lit_s: float matrix of M x n_a
            Contains literals for every subjects in X.

        X_lit_o: float matrix of M x n_a
            Contains literals for every objects in X.

        Returns:
        --------
        y: Mx1 vectors
            Contains the probs result of each M data.
        """
        # Decompose X into head, relationship, tail
        s, r, o = X[:, 0], X[:, 1], X[:, 2]

        if self.gpu:
            s = Variable(torch.from_numpy(s).cuda())
            r = Variable(torch.from_numpy(r).cuda())
            o = Variable(torch.from_numpy(o).cuda())
            X_lit_s = Variable(torch.from_numpy(X_lit_s).cuda())
            X_lit_o = Variable(torch.from_numpy(X_lit_o).cuda())
            X_lit_s_img = Variable(torch.from_numpy(X_lit_s_img).cuda())
            X_lit_o_img = Variable(torch.from_numpy(X_lit_o_img).cuda())
            X_lit_s_txt = Variable(torch.from_numpy(X_lit_s_txt).cuda())
            X_lit_o_txt = Variable(torch.from_numpy(X_lit_o_txt).cuda())
        else:
            s = Variable(torch.from_numpy(s))
            r = Variable(torch.from_numpy(r))
            o = Variable(torch.from_numpy(o))
            X_lit_s = Variable(torch.from_numpy(X_lit_s))
            X_lit_o = Variable(torch.from_numpy(X_lit_o))
            X_lit_s_img = Variable(torch.from_numpy(X_lit_s_img))
            X_lit_o_img = Variable(torch.from_numpy(X_lit_o_img))
            X_lit_s_txt = Variable(torch.from_numpy(X_lit_s_txt))
            X_lit_o_txt = Variable(torch.from_numpy(X_lit_o_txt))

        # Project to embedding, each is M x k
        e_s = self.emb_ent(s)
        e_r = self.emb_rel(r)
        e_o = self.emb_ent(o)

        # Forward
        phi = torch.cat([e_s, e_r, e_o, X_lit_s, X_lit_o], 1)

        if self.img_lit and self.txt_lit:
            # Project image features to embedding of M x k
            e_s_img = self.emb_img(X_lit_s_img)
            e_o_img = self.emb_img(X_lit_o_img)
            e_s_txt = self.emb_txt(X_lit_s_txt)
            e_o_txt = self.emb_txt(X_lit_o_txt)
            phi = torch.cat([e_s, e_r, e_o, X_lit_s, X_lit_o, e_s_img, e_o_img, e_s_txt, e_o_txt], 1)
        elif self.img_lit:
            e_s_img = self.emb_img(X_lit_s_img)
            e_o_img = self.emb_img(X_lit_o_img)
            phi = torch.cat([e_s, e_r, e_o, X_lit_s, X_lit_o, e_s_img, e_o_img], 1)
        elif self.txt_lit:
            e_s_txt = self.emb_txt(X_lit_s_txt)
            e_o_txt = self.emb_txt(X_lit_o_txt)
            phi = torch.cat([e_s, e_r, e_o, X_lit_s, X_lit_o, e_s_txt, e_o_txt], 1)
        else:
            phi = torch.cat([e_s, e_r, e_o, X_lit_s, X_lit_o], 1)

        score = self.mlp(phi)

        return score

    def predict(self, X, X_lit_s, X_lit_o, X_lit_s_img, X_lit_o_img, X_lit_s_txt, X_lit_o_txt):
        y_pred = self.forward(X, X_lit_s, X_lit_o, X_lit_s_img, X_lit_o_img,
                              X_lit_s_txt, X_lit_o_txt).view(-1, 1)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()


@inherit_docstrings
class RESCAL(Model):
    """
    RESCAL: bilinear model
    ----------------------
    Nickel, Maximilian, Volker Tresp, and Hans-Peter Kriegel.
    "A three-way model for collective learning on multi-relational data."
    ICML. 2011.
    """

    def __init__(self, n_e, n_r, k, lam, gpu=False):
        """
        RESCAL: bilinear model
        ----------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.

            lam: float
                Prior strength of the embeddings. Used to constaint the
                embedding norms inside a (euclidean) unit ball. The prior is
                Gaussian, this param is the precision.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(RESCAL, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.lam = lam

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k**2)

        self.embeddings = [self.emb_E, self.emb_R]
        self.initialize_embeddings()

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X):
        X = Variable(torch.from_numpy(X))
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs).view(-1, self.k, 1)
        e_ts = self.emb_E(ts).view(-1, self.k, 1)
        W = self.emb_R(ls).view(-1, self.k, self.k)  # M x k x k

        # Forward
        out = torch.bmm(torch.transpose(e_hs, 1, 2), W)  # h^T W
        out = torch.bmm(out, e_ts)  # (h^T W) h
        out = out.view(-1, 1)  # [-1, 1, 1] -> [-1, 1]

        return out

@inherit_docstrings
class DistMult(Model):
    """
    DistMult: diagonal bilinear model
    ---------------------------------
    Yang, Bishan, et al. "Learning multi-relational semantics using
    neural-embedding models." arXiv:1411.4072 (2014).
    """

    def __init__(self, n_e, n_r, k, lam, gpu=False):
        """
        DistMult: diagonal bilinear model
        ---------------------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.

            lam: float
                Prior strength of the embeddings. Used to constaint the
                embedding norms inside a (euclidean) unit ball. The prior is
                Gaussian, this param is the precision.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(DistMult, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.lam = lam

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)

        self.embeddings = [self.emb_E, self.emb_R]
        self.initialize_embeddings()

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X):
        X = Variable(torch.from_numpy(X))
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        W = self.emb_R(ls)

        # Forward
        f = torch.sum(e_hs * W * e_ts, 1)

        return f.view(-1, 1)


@inherit_docstrings
class ERMLP(Model):
    """
    ER-MLP: Entity-Relation MLP
    ---------------------------
    Dong, Xin, et al. "Knowledge vault: A web-scale approach to probabilistic knowledge fusion." KDD, 2014.
    """

    def __init__(self, n_e, n_r, k, h_dim, p, lam, gpu=False):
        """
        ER-MLP: Entity-Relation MLP
        ---------------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

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
        super(ERMLP, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.h_dim = h_dim
        self.p = p
        self.lam = lam

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)

        self.mlp = nn.Sequential(
            nn.Linear(3*k, h_dim),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(h_dim, 1),
        )

        self.embeddings = [self.emb_E, self.emb_R]
        self.initialize_embeddings()

        # Xavier init
        for p in self.mlp.modules():
            if isinstance(p, nn.Linear):
                in_dim = p.weight.size(0)
                p.weight.data.normal_(0, 1/np.sqrt(in_dim/2))

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X):
        X = Variable(torch.from_numpy(X))
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)

        # Forward
        phi = torch.cat([e_hs, e_ts, e_ls], 1)  # M x 3k
        y = self.mlp(phi)

        return y.view(-1, 1)

    def predict_all(self, X):
        """
        Let X be a triple (s, p, o), i.e. tensor of 1x3, return two lists:
            - list of (s, p, all_others)
            - list of (all_others, p, o)
        """
        X = Variable(torch.from_numpy(X))
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        s, p, o = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_s = self.emb_E(s)
        e_r = self.emb_R(p)
        e_o = self.emb_E(o)

        # Predict o
        e_s_rep = e_s.repeat(self.n_e, 1)  # n_ent x k
        e_r_rep = e_r.repeat(self.n_e, 1)  # n_ent x k
        phi_o = torch.cat([e_s_rep, e_r_rep, self.emb_E.weight], 1)  # n_ent x 3k
        y_o = self.mlp(phi_o).view(-1)

        # Predict s
        e_o_rep = e_o.repeat(self.n_e, 1)
        phi_s = torch.cat([self.emb_E.weight, e_r_rep, e_o_rep], 1)  # n_ent x 3k
        y_s = self.mlp(phi_s).view(-1)

        return y_s, y_o


@inherit_docstrings
class TransE(Model):
    """
    TransE embedding model
    ----------------------
    Bordes, Antoine, et al.
    "Translating embeddings for modeling multi-relational data." NIPS. 2013.
    """

    def __init__(self, n_e, n_r, k, gamma, d='l2', gpu=False):
        """
        TransE embedding model
        ----------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.

            gamma: float
                Margin size for TransE's hinge loss.

            d: {'l1', 'l2'}
                Distance measure to be used in the loss.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(TransE, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e  # Num of entities
        self.n_r = n_r  # Num of rels
        self.k = k
        self.gamma = gamma
        self.d = d

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)

        self.embeddings = [self.emb_E, self.emb_R]
        self.initialize_embeddings()

        # Remove relation embeddings from list so that it won't normalized be
        # during training.
        self.embeddings = [self.emb_E]

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X):
        """
        Given a (mini)batch of triplets X of size M, compute the energies.

        Params:
        -------
        X: int matrix of M x 3, where M is the (mini)batch size
            First column contains index of head entities.
            Second column contains index of relationships.
            Third column contains index of tail entities.

        Returns:
        --------
        f: float matrix of M x 1
            Contains energies of each triplets.
        """
        X = Variable(torch.from_numpy(X))
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)

        f = self.energy(e_hs, e_ls, e_ts).view(-1, 1)

        return f

    def energy(self, h, l, t):
        """
        Compute TransE energy

        Params:
        -------
        h: Mxk tensor
            Contains head embeddings.

        l: Mxk tensor
            Contains relation embeddings.

        t: Mxk tensor
            Contains tail embeddings.

        Returns:
        --------
        E: Mx1 tensor
            Energy of each triplets, computed by d(h + l, t) for some func d.
        """
        if self.d == 'l1':
            out = torch.sum(torch.abs(h + l - t), 1)
        else:
            out = torch.sqrt(torch.sum((h + l - t)**2, 1))

        return out


@inherit_docstrings
class NTN(Model):
    """
    NTN: Neural Tensor Machine
    --------------------------
    Socher, Richard, et al. "Reasoning with neural tensor networks for knowledge base completion." NIPS, 2013.
    """

    def __init__(self, n_e, n_r, k, slice, lam, gpu=False):
        """
        NTN: Neural Tensor Machine
        --------------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.

            slice: int
                Number of tensor slices.

            lam: float
                Prior strength of the embeddings. Used to constaint the
                embedding norms inside a (euclidean) unit ball. The prior is
                Gaussian, this param is the precision.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(NTN, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.slice = slice
        self.lam = lam

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k*self.k*self.slice)
        self.V = nn.Embedding(self.n_r, 2*self.k*self.slice)
        self.U = nn.Embedding(self.n_r, self.slice)
        self.b = nn.Embedding(self.n_r, self.slice)

        self.embeddings = [self.emb_E, self.emb_R]
        self.initialize_embeddings()

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X):
        X = Variable(torch.from_numpy(X))
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, broadcasting is a bit convoluted
        e_hs = self.emb_E(hs).view(-1, self.k, 1)
        e_ts = self.emb_E(ts).view(-1, self.k, 1)
        Wr = self.emb_R(ls).view(-1, self.slice, self.k, self.k)
        Vr = self.V(ls).view(-1, self.slice, 2*self.k)
        Ur = self.U(ls).view(-1, 1, self.slice)
        br = self.b(ls).view(-1, self.slice, 1)

        # Forward
        # -------

        M = e_hs.size(0)

        # M x s x 1 x 3
        e_hs_ = e_hs.unsqueeze(1).expand(M, self.slice, self.k, 1).transpose(2, 3)
        # M x s x k x 1
        e_ts_ = e_ts.unsqueeze(1).expand(M, self.slice, self.k, 1)

        # M x s x 1 x 1
        quad = torch.matmul(torch.matmul(e_hs_, Wr), e_ts_)
        quad = quad.view(-1, self.slice)  # M x s

        # Vr: M x s x 2k
        # [e1 e2]: M x 2k x 1
        es = torch.cat([e_hs, e_ts], dim=1)  # M x 2k x 1
        affine = torch.baddbmm(br, Vr, es).view(-1, self.slice)  # M x s

        # Scores
        g = torch.bmm(Ur, F.leaky_relu(quad + affine).view(-1, self.slice, 1))

        return g.view(-1, 1)
