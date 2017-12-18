import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import kga.op as op
import kga.util as util
from kga.util import inherit_docstrings

from kga.models.base import Model
import pdb

@inherit_docstrings
class ERLMLP_MovieLens(Model):
    """
    ERL-MLP: Entity-Relation-Literal MLP for MovieLens
    --------------------------------------------------
    """

    def __init__(self, n_usr, n_mov, n_rat, n_usr_lit, n_mov_lit, k, h_dim, gpu=False, img_lit=False, txt_lit=False):
        """
        ERL-MLP: Entity-Relation-Literal MLP for MovieLens
        --------------------------------------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            n_a: int
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
        super(ERLMLP_MovieLens, self).__init__(gpu)

        # Hyperparams
        self.n_usr = n_usr
        self.n_mov = n_mov
        self.n_rat = n_rat
        self.n_usr_lit = n_usr_lit
        self.n_mov_lit = n_mov_lit
        self.k = k
        self.h_dim = h_dim
        self.img_lit = img_lit
        self.txt_lit = txt_lit

        # Nets
        self.emb_usr = nn.Embedding(n_usr, k)
        self.emb_mov = nn.Embedding(n_mov, k)
        self.emb_rat = nn.Embedding(n_rat, k)

        self.emb_img = nn.Linear(512, self.k)
        self.emb_txt = nn.Linear(384, self.k)

        # Image embeddings
        if img_lit and txt_lit:
            n_input = 5*k+n_usr_lit+n_mov_lit
        elif img_lit or txt_lit:
            n_input = 4*k+n_usr_lit+n_mov_lit
        else:
            n_input = 3*k+n_usr_lit+n_mov_lit

        self.mlp = nn.Sequential(
            nn.Linear(n_input, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

        self.embeddings = [self.emb_usr, self.emb_mov, self.emb_rat]
        self.initialize_embeddings()

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X, X_lit_usr, X_lit_mov, X_lit_img=None, X_lit_txt=None):
        """
        Given a (mini)batch of triplets X of size M, predict the validity.

        Params:
        -------
        X: int matrix of M x 3, where M is the (mini)batch size
            First column contains index of head entities.
            Second column contains index of relationships.
            Third column contains index of tail entities.

        X_lit: float matrix of M x n_a
            Contains all literals/attributes information of all data in batch.
            i-th row correspond to the i-th data in X.

        Returns:
        --------
        y: Mx1 vectors
            Contains the probs result of each M data.
        """
        M = X.shape[0]

        # Decompose X into head, relationship, tail
        s, r, o = X[:, 0], X[:, 1], X[:, 2]

        if self.gpu:
            s = Variable(torch.from_numpy(s).cuda())
            r = Variable(torch.from_numpy(r).cuda())
            o = Variable(torch.from_numpy(o).cuda())
            X_lit_usr = Variable(torch.from_numpy(X_lit_usr).cuda())
            X_lit_mov = Variable(torch.from_numpy(X_lit_mov).cuda())
            X_lit_img = Variable(torch.from_numpy(X_lit_img).cuda())
            X_lit_txt = Variable(torch.from_numpy(X_lit_txt).cuda())
        else:
            s = Variable(torch.from_numpy(s))
            r = Variable(torch.from_numpy(r))
            o = Variable(torch.from_numpy(o))
            X_lit_usr = Variable(torch.from_numpy(X_lit_usr))
            X_lit_mov = Variable(torch.from_numpy(X_lit_mov))
            X_lit_img = Variable(torch.from_numpy(X_lit_img))
            X_lit_txt = Variable(torch.from_numpy(X_lit_txt))

        # Project to embedding, each is M x k
        e_usr = self.emb_usr(s)
        e_rat = self.emb_rat(r)
        e_mov = self.emb_mov(o)

        if self.img_lit and self.txt_lit:
            # Project image features to embedding of M x k
            e_img = self.emb_img(X_lit_img)
            e_txt = self.emb_txt(X_lit_txt)
            phi = torch.cat([e_usr, e_rat, e_mov, X_lit_usr, X_lit_mov, e_img, e_txt], 1)
        elif self.img_lit:
            e_img = self.emb_img(X_lit_img)
            phi = torch.cat([e_usr, e_rat, e_mov, X_lit_usr, X_lit_mov, e_img], 1)
        elif self.txt_lit:
            e_txt = self.emb_txt(X_lit_txt)
            phi = torch.cat([e_usr, e_rat, e_mov, X_lit_usr, X_lit_mov, e_txt], 1)
        else:
            phi = torch.cat([e_usr, e_rat, e_mov, X_lit_usr, X_lit_mov], 1)

        score = self.mlp(phi)

        return score

    def predict(self, X, X_lit_usr, X_lit_mov, X_lit_img=None, X_lit_txt=None):
        y_pred = self.forward(X, X_lit_usr, X_lit_mov, X_lit_img, X_lit_txt).view(-1, 1)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()


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
        self.n_l = n_l
        if n_text != None:
            self.reprs_text_subject = nn.Linear(n_text, self.k)
            self.reprs_text_object = nn.Linear(n_text, self.k)

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k**2)

        self.embeddings = [self.emb_E, self.emb_R]
        self.initialize_embeddings()

        self.mlp = nn.Sequential(
            nn.Linear(self.n_l + k, k),
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
            e1_rep = torch.cat([e_hs, s_lit], 1)  # M x (k + n_l)
            e2_rep = torch.cat([e_ts, o_lit], 1)  # M x (k + n_l)            
            e1_rep = self.mlp(e1_rep).view(-1, self.k, 1)   # M x k x 1
            e2_rep = self.mlp(e2_rep).view(-1, self.k, 1)   # M x k x 1
        
        elif text_s != None and text_o != None:    
            text_s_rep = self.reprs_subject(text_s)
            text_o_rep = self.reprs_object(text_o)
            e1_rep = torch.cat([e_hs, s_lit, text_s_rep], 1)  # M x 3k
            e2_rep = torch.cat([e_ts, o_lit, text_o_rep], 1)  # M x 3k
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
            
@inherit_docstrings
class DistMultDecoupled(Model):
    """
    DistMult: diagonal bilinear model, without subject and object constraint
    ------------------------------------------------------------------------
    Yang, Bishan, et al. "Learning multi-relational semantics using
    neural-embedding models." arXiv:1411.4072 (2014).
    """

    def __init__(self, n_s, n_r, n_o, k, lam, gpu=False):
        """
        DistMult: diagonal bilinear model, without subject and object constraint
        ------------------------------------------------------------------------

        Params:
        -------
            n_s: int
                Number of subjects in dataset.

            n_r: int
                Number of relationships in dataset.

            n_o: int
                Number of objects in dataset.

            k: int
                Embedding size.

            lam: float
                Prior strength of the embeddings. Used to constaint the
                embedding norms inside a (euclidean) unit ball. The prior is
                Gaussian, this param is the precision.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(DistMultDecoupled, self).__init__(gpu)

        # Hyperparams
        self.n_s = n_s
        self.n_r = n_r
        self.n_o = n_o
        self.k = k
        self.lam = lam

        # Nets
        self.emb_S = nn.Embedding(self.n_s, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)
        self.emb_O = nn.Embedding(self.n_o, self.k)

        self.embeddings = [self.emb_S, self.emb_R, self.emb_O]
        self.initialize_embeddings()

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X):
        # Decompose X into head, relationship, tail
        s, r, o = X[:, 0], X[:, 1], X[:, 2]

        if self.gpu:
            s = Variable(torch.from_numpy(s).cuda())
            r = Variable(torch.from_numpy(r).cuda())
            o = Variable(torch.from_numpy(o).cuda())
        else:
            s = Variable(torch.from_numpy(s))
            r = Variable(torch.from_numpy(r))
            o = Variable(torch.from_numpy(o))

        # Project to embedding, each is M x k
        e_s = self.emb_S(s)
        e_o = self.emb_O(o)
        W = self.emb_R(r)

        # Forward
        f = torch.sum(e_s * W * e_o, 1)

        return f.view(-1, 1)