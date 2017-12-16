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
            nn.Linear(h_dim, 1),
        )

        # SA-MLP: subject-attribute net
        self.sa_mlp = nn.Sequential(
            nn.Linear(2*k, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

        # OA-MLP: object-attribute net
        self.oa_mlp = nn.Sequential(
            nn.Linear(2*k, h_dim),
            nn.ReLU(),
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
        # Decompose X into head, relationship, tail
        s, p, o = X[:, 0], X[:, 1], X[:, 2]

        if self.gpu:
            s = Variable(torch.from_numpy(s).cuda())
            p = Variable(torch.from_numpy(p).cuda())
            o = Variable(torch.from_numpy(o).cuda())

            if s_attrs is not None or o_attrs is not None:
                s_attrs = Variable(torch.from_numpy(s_attrs).cuda())
                o_attrs = Variable(torch.from_numpy(o_attrs).cuda())
        else:
            s = Variable(torch.from_numpy(s))
            p = Variable(torch.from_numpy(p))
            o = Variable(torch.from_numpy(o))

            if s_attrs is not None or o_attrs is not None:
                s_attrs = Variable(torch.from_numpy(s_attrs))
                o_attrs = Variable(torch.from_numpy(o_attrs))

        # Project to embedding, each is M x k
        e_s = self.emb_ent(s)
        e_r = self.emb_rel(p)
        e_o = self.emb_ent(o)

        # Forward ER-MLP
        phi_er = torch.cat([e_s, e_r, e_o], 1)  # M x 3k
        y_er = self.ermlp(phi_er)

        if s_attrs is not None or o_attrs is not None:
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

@inherit_docstrings
class RESCAL_literal(Model):
    """
    RESCAL: bilinear model
    ----------------------
    Nickel, Maximilian, Volker Tresp, and Hans-Peter Kriegel.
    "A three-way model for collective learning on multi-relational data."
    ICML. 2011.
    """

    def __init__(self, n_e, n_r, k, lam, n_l=None, n_text=None , gpu=False):
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
        super(RESCAL_literal, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.lam = lam
        if n_l != None:
            self.reprs_subject = nn.Linear(n_l, self.k)
            self.reprs_object = nn.Linear(n_l, self.k)

        if n_text != None:
            self.reprs_text_subject = nn.Linear(n_text, self.k)
            self.reprs_text_object = nn.Linear(n_text, self.k)

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k**2)

        self.embeddings = [self.emb_E, self.emb_R]
        self.initialize_embeddings()

        self.mlp = nn.Sequential(
            nn.Linear(2*k, k),
            nn.ReLU()
        )
        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X, s_lit=None, o_lit=None, text_s=None, text_o=None):
        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        if self.gpu:
            hs = Variable(torch.from_numpy(hs).cuda())
            ls = Variable(torch.from_numpy(ls).cuda())
            ts = Variable(torch.from_numpy(ts).cuda())
            if s_lit != None and o_lit != None:
                s_lit = Variable(torch.from_numpy(s_lit).cuda())
                o_lit = Variable(torch.from_numpy(o_lit).cuda())
            if text_s != None and text_o != None:    
                text_s = Variable(torch.from_numpy(text_s).cuda())
                text_o = Variable(torch.from_numpy(text_o).cuda())
        else:
            hs = Variable(torch.from_numpy(hs))
            ls = Variable(torch.from_numpy(ls))
            ts = Variable(torch.from_numpy(ts))
            if s_lit != None and o_lit != None:
                s_lit = Variable(torch.from_numpy(s_lit))
                o_lit = Variable(torch.from_numpy(o_lit))
            if text_s != None and text_o != None:    
                text_s = Variable(torch.from_numpy(text_s))
                text_o = Variable(torch.from_numpy(text_o))
        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        if s_lit != None and o_lit != None:
            s_rep = self.reprs_subject(s_lit)
            o_rep = self.reprs_object(o_lit)
            e1_rep = torch.cat([e_hs, s_rep], 1)  # M x 2k
            e2_rep = torch.cat([e_ts, o_rep], 1)  # M x 2k            
            e1_rep = self.mlp(e1_rep).view(-1, self.k, 1)   # M x k x 1
            e2_rep = self.mlp(e2_rep).view(-1, self.k, 1)   # M x k x 1
        
        elif text_s != None and text_o != None:    
            text_s_rep = self.reprs_subject(text_s)
            text_o_rep = self.reprs_object(text_o)
            e1_rep = torch.cat([e_hs, s_rep, text_s_rep], 1)  # M x 3k
            e2_rep = torch.cat([e_ts, o_rep, text_o_rep], 1)  # M x 3k
            e1_rep = self.mlp(e1_rep).view(-1, self.k, 1)   # M x k x 1
            e2_rep = self.mlp(e2_rep).view(-1, self.k, 1)   # M x k x 1
        else:
            e1_rep = e_hs.view(-1, self.k, 1)
            e2_rep = e_ts.view(-1, self.k, 1)

        W = self.emb_R(ls).view(-1, self.k, self.k)  # M x k x k
        # Forward
        out = torch.bmm(torch.transpose(e1_rep, 1, 2), W)  # h^T W
        out = torch.bmm(out, e2_rep)  # (h^T W) h
        out = out.view(-1, 1)  # [-1, 1, 1] -> [-1, 1]

        return out

    def predict(self, X, s_lit=None, o_lit=None, text_s=None, text_o=None, sigmoid=True):
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
        y_pred = self.forward(X, s_lit, o_lit, text_s, text_o).view(-1, 1)
        if sigmoid:
            y_pred = F.sigmoid(y_pred)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()

@inherit_docstrings
class DistMult_literal(Model):
    """
    DistMult: diagonal bilinear model
    ---------------------------------
    Yang, Bishan, et al. "Learning multi-relational semantics using
    neural-embedding models." arXiv:1411.4072 (2014).
    """

    def __init__(self, n_e, n_r, n_l, k, lam, gpu=False):
        """
        DistMult: diagonal bilinear model
        ---------------------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            n_l: int
                Number of literal relations in dataset.

            k: int
                Embedding size.

            lam: float
                Prior strength of the embeddings. Used to constaint the
                embedding norms inside a (euclidean) unit ball. The prior is
                Gaussian, this param is the precision.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(DistMult_literal, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.lam = lam
        self.reprs_subject = nn.Linear(n_l, self.k)
        self.reprs_object = nn.Linear(n_l, self.k)

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)

        self.embeddings = [self.emb_E, self.emb_R]
        self.initialize_embeddings()

        self.mlp = nn.Sequential(
            nn.Linear(2*k, k),
            nn.ReLU()
        )

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X, s_lit, o_lit):
        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]
        if self.gpu:
            hs = Variable(torch.from_numpy(hs).cuda())
            ls = Variable(torch.from_numpy(ls).cuda())
            ts = Variable(torch.from_numpy(ts).cuda())
            s_lit = Variable(torch.from_numpy(s_lit).cuda())
            o_lit = Variable(torch.from_numpy(o_lit).cuda())
        else:
            hs = Variable(torch.from_numpy(hs))
            ls = Variable(torch.from_numpy(ls))
            ts = Variable(torch.from_numpy(ts))
            s_lit = Variable(torch.from_numpy(s_lit))
            o_lit = Variable(torch.from_numpy(o_lit))
        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)

        W = self.emb_R(ls)

        s_rep = self.reprs_subject(s_lit)
        o_rep = self.reprs_object(o_lit)

        e1_rep = torch.cat([e_hs, s_rep], 1)  # M x 2k
        e1_rep = self.mlp(e1_rep)   # M x k
        e2_rep = torch.cat([e_ts, o_rep], 1)  # M x 2k
        e2_rep = self.mlp(e2_rep)   # M x k
        # Forward
        f = torch.sum(e1_rep * W * e2_rep, 1)

        return f.view(-1, 1)

    def predict(self, X, s_lit, o_lit, sigmoid=False):
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
        y_pred = self.forward(X, s_lit, o_lit).view(-1, 1)

        if sigmoid:
            y_pred = F.sigmoid(y_pred)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()
