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

    def __init__(self, n_usr, n_mov, n_rat, n_usr_lit, n_mov_lit, k, h_dim, gpu=False, usr_lit=False, mov_lit=False, img_lit=False, txt_lit=False):
        super(ERLMLP_MovieLens, self).__init__(gpu)

        # Hyperparams
        self.n_usr = n_usr
        self.n_mov = n_mov
        self.n_rat = n_rat
        self.n_usr_lit = n_usr_lit
        self.n_mov_lit = n_mov_lit
        self.k = k
        self.h_dim = h_dim
        self.usr_lit = usr_lit
        self.mov_lit = mov_lit
        self.img_lit = img_lit
        self.txt_lit = txt_lit

        # Nets
        self.emb_usr = nn.Embedding(n_usr, k)
        self.emb_mov = nn.Embedding(n_mov, k)
        self.emb_rat = nn.Embedding(n_rat, k)

        self.emb_img = nn.Linear(512, self.k)
        self.emb_txt = nn.Linear(384, self.k)

        # Determine MLP input size
        n_input = 3*k

        if usr_lit:
            n_input += n_usr_lit
        if mov_lit:
            n_input += n_mov_lit
        if img_lit:
            n_input += k
        if txt_lit:
            n_input += k

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
        M = X.shape[0]

        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        s, r, o = X[:, 0], X[:, 1], X[:, 2]

        if self.gpu:
            X_lit_usr = Variable(torch.from_numpy(X_lit_usr).cuda())
            X_lit_mov = Variable(torch.from_numpy(X_lit_mov).cuda())
            X_lit_img = Variable(torch.from_numpy(X_lit_img).cuda())
            X_lit_txt = Variable(torch.from_numpy(X_lit_txt).cuda())
        else:
            X_lit_usr = Variable(torch.from_numpy(X_lit_usr))
            X_lit_mov = Variable(torch.from_numpy(X_lit_mov))
            X_lit_img = Variable(torch.from_numpy(X_lit_img))
            X_lit_txt = Variable(torch.from_numpy(X_lit_txt))

        # Project to embedding, each is M x k
        e_usr = self.emb_usr(s)
        e_rat = self.emb_rat(r)
        e_mov = self.emb_mov(o)

        e_img = self.emb_img(X_lit_img)
        e_txt = self.emb_txt(X_lit_txt)

        phi = torch.cat([e_usr, e_rat, e_mov], 1)

        if self.usr_lit:
            phi = torch.cat([phi, X_lit_usr], 1)
        if self.mov_lit:
            phi = torch.cat([phi, X_lit_mov], 1)
        if self.img_lit:
            phi = torch.cat([phi, e_img], 1)
        if self.txt_lit:
            phi = torch.cat([phi, e_txt], 1)

        score = self.mlp(phi)

        return score

    def predict(self, X, X_lit_usr, X_lit_mov, X_lit_img=None, X_lit_txt=None):
        y_pred = self.forward(X, X_lit_usr, X_lit_mov, X_lit_img, X_lit_txt).view(-1, 1)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()

@inherit_docstrings
class ERMLP_literal(Model):
    """
    ER-MLP: Entity-Relation MLP
    ---------------------------
    Dong, Xin, et al. "Knowledge vault: A web-scale approach to probabilistic knowledge fusion." KDD, 2014.
    """

    def __init__(self, n_e, n_r, k, h_dim, p, lam, n_numeric, n_text, numeric = True, text=True, gpu=False):
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
        super(ERMLP_literal, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.h_dim = h_dim
        self.p = p
        self.lam = lam
        self.n_numeric = n_numeric
        self.n_text = n_text

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)
        # Determine MLP input size
        n_input = 3*k
        if numeric:
            n_input += 2*n_numeric
        if text:
            n_input += 2*n_text

        self.mlp = nn.Sequential(
            nn.Linear(n_input, h_dim),
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

    def forward(self, X, numeric_lit_s, numeric_lit_o, text_lit_s, text_lit_o, numeric=True, text=True):
        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        if self.gpu:
            hs = Variable(torch.from_numpy(hs).cuda())
            ls = Variable(torch.from_numpy(ls).cuda())
            ts = Variable(torch.from_numpy(ts).cuda())
            if numeric:
                numeric_lit_s = Variable(torch.from_numpy(numeric_lit_s).cuda())
                numeric_lit_o = Variable(torch.from_numpy(numeric_lit_o).cuda())
            if text:
                text_lit_s = Variable(torch.from_numpy(text_lit_s).cuda())
                text_lit_o = Variable(torch.from_numpy(text_lit_o).cuda())

        else:
            hs = Variable(torch.from_numpy(hs))
            ls = Variable(torch.from_numpy(ls))
            ts = Variable(torch.from_numpy(ts))
            if numeric:
                numeric_lit_s = Variable(torch.from_numpy(numeric_lit_s))
                numeric_lit_o = Variable(torch.from_numpy(numeric_lit_o))
            if text:
                text_lit_s = Variable(torch.from_numpy(text_lit_s))
                text_lit_o = Variable(torch.from_numpy(text_lit_o))

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)

        # Forward
        if numeric and not text:
            phi = torch.cat([e_hs, numeric_lit_s, e_ts, numeric_lit_o, e_ls], 1)  # M x (3k + numeric)
        elif text and not numeric:
            phi = torch.cat([e_hs, text_lit_s, e_ts, text_lit_o, e_ls], 1)  # M x (3k + text)
        elif numeric and text:
            phi = torch.cat([e_hs, text_lit_s, numeric_lit_s, e_ts, text_lit_o, numeric_lit_o, e_ls], 1)  # M x (3k + text+numeric)
        else:
            phi = torch.cat([e_hs, e_ts, e_ls])
        y = self.mlp(phi)
        return y.view(-1, 1)

@inherit_docstrings
class ERLMLP(Model):

    def __init__(self, n_ent, n_rel, n_lit, k, h_dim, gpu=False, img_lit=False, txt_lit=False):
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
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        s, r, o = X[:, 0], X[:, 1], X[:, 2]

        if self.gpu:
            X_lit_s = Variable(torch.from_numpy(X_lit_s).cuda())
            X_lit_o = Variable(torch.from_numpy(X_lit_o).cuda())
            X_lit_s_img = Variable(torch.from_numpy(X_lit_s_img).cuda())
            X_lit_o_img = Variable(torch.from_numpy(X_lit_o_img).cuda())
            X_lit_s_txt = Variable(torch.from_numpy(X_lit_s_txt).cuda())
            X_lit_o_txt = Variable(torch.from_numpy(X_lit_o_txt).cuda())
        else:
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
class RESCAL_literal(Model):

    def __init__(self, n_e, n_r, k, lam, n_l=None, n_text=None , gpu=False):
        super(RESCAL_literal, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.lam = lam
        self.n_l = n_l
        self.n_text = n_text
        if self.n_text != None:
            self.reprs_text_subject = nn.Linear(self.n_text, self.k)
            self.reprs_text_object = nn.Linear(self.n_text, self.k)

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
            if self.n_l != None:
                s_lit = Variable(torch.from_numpy(s_lit).cuda())
                o_lit = Variable(torch.from_numpy(o_lit).cuda())
            if self.n_text != None:
                text_s = Variable(torch.from_numpy(text_s).cuda())
                text_o = Variable(torch.from_numpy(text_o).cuda())
        else:
            hs = Variable(torch.from_numpy(hs))
            ls = Variable(torch.from_numpy(ls))
            ts = Variable(torch.from_numpy(ts))
            if self.n_l != None:
                s_lit = Variable(torch.from_numpy(s_lit))
                o_lit = Variable(torch.from_numpy(o_lit))
            if self.n_text != None:
                text_s = Variable(torch.from_numpy(text_s))
                text_o = Variable(torch.from_numpy(text_o))
        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        if self.n_l != None:
            e1_rep = torch.cat([e_hs, s_lit], 1)  # M x (k + n_l)
            e2_rep = torch.cat([e_ts, o_lit], 1)  # M x (k + n_l)
            e1_rep = self.mlp(e1_rep).view(-1, self.k, 1)   # M x k x 1
            e2_rep = self.mlp(e2_rep).view(-1, self.k, 1)   # M x k x 1

        elif self.n_text != None:
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
        y_pred = self.forward(X, s_lit, o_lit, text_s, text_o).view(-1, 1)
        if sigmoid:
            y_pred = F.sigmoid(y_pred)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()


@inherit_docstrings
class DistMult_literal(Model):

    def __init__(self, n_e, n_r, n_l, k, lam, gpu=False):
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
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        if self.gpu:
            s_lit = Variable(torch.from_numpy(s_lit).cuda())
            o_lit = Variable(torch.from_numpy(o_lit).cuda())
        else:
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
        y_pred = self.forward(X, s_lit, o_lit).view(-1, 1)

        if sigmoid:
            y_pred = F.sigmoid(y_pred)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()


@inherit_docstrings
class DistMultDecoupled(Model):

    def __init__(self, n_s, n_r, n_o, k, lam, gpu=False):
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
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        s, r, o = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_s = self.emb_S(s)
        e_o = self.emb_O(o)
        W = self.emb_R(r)

        # Forward
        f = torch.sum(e_s * W * e_o, 1)

        return f.view(-1, 1)
