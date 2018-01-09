import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal, xavier_uniform
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
            nn.Dropout(0.2),
            nn.Linear(n_input, h_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
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

        # Project to embedding, each is M x k
        e_usr = self.emb_usr(s)
        e_rat = self.emb_rat(r)
        e_mov = self.emb_mov(o)

        phi = torch.cat([e_usr, e_rat, e_mov], 1)

        if self.usr_lit:
            X_lit_usr = Variable(torch.from_numpy(X_lit_usr))
            X_lit_usr = X_lit_usr.cuda() if self.gpu else X_lit_usr

            phi = torch.cat([phi, X_lit_usr], 1)

        if self.mov_lit:
            X_lit_mov = Variable(torch.from_numpy(X_lit_mov))
            X_lit_mov = X_lit_mov.cuda() if self.gpu else X_lit_mov

            phi = torch.cat([phi, X_lit_mov], 1)

        if self.img_lit:
            X_lit_img = Variable(torch.from_numpy(X_lit_img))
            X_lit_img = X_lit_img.cuda() if self.gpu else X_lit_img
            e_img = self.emb_img(X_lit_img)

            phi = torch.cat([phi, e_img], 1)

        if self.txt_lit:
            X_lit_txt = Variable(torch.from_numpy(X_lit_txt))
            X_lit_txt = X_lit_txt.cuda() if self.gpu else X_lit_txt
            e_txt = self.emb_txt(X_lit_txt)

            phi = torch.cat([phi, e_txt], 1)

        score = self.mlp(phi).view(-1, 1)

        return score

    def predict(self, X, X_lit_usr, X_lit_mov, X_lit_img=None, X_lit_txt=None):
        y_pred = self.forward(X, X_lit_usr, X_lit_mov, X_lit_img, X_lit_txt).view(-1, 1)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()


@inherit_docstrings
class ERLMLP(Model):
    """
    ERL-MLP: Entity-Relation-Literal MLP for generic KG
    ---------------------------------------------------
    """

    def __init__(self, n_ent, n_rel, n_lit, k, h_dim, gpu=False, num_lit=False, img_lit=False, txt_lit=False):
        super(ERLMLP, self).__init__(gpu)

        # Hyperparams
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.n_lit = n_lit
        self.k = k
        self.h_dim = h_dim
        self.num_lit = num_lit
        self.img_lit = img_lit
        self.txt_lit = txt_lit

        # Nets
        self.emb_ent = nn.Embedding(n_ent, k)
        self.emb_rel = nn.Embedding(n_rel, k)

        self.emb_img = nn.Linear(512, self.k)
        self.emb_txt = nn.Linear(384, self.k)

        # Determine MLP input size
        n_input = 3*k

        if num_lit:
            n_input += 2*n_lit
        if img_lit:
            n_input += 2*k
        if txt_lit:
            n_input += 2*k

        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_input, h_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_dim, 1)
        )

        self.embeddings = [self.emb_ent, self.emb_rel]
        self.initialize_embeddings()

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X, X_lit_s, X_lit_o, X_lit_s_img, X_lit_o_img, X_lit_s_txt, X_lit_o_txt):
        M = X.shape[0]

        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        s, r, o = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_s = self.emb_ent(s)
        e_r = self.emb_rel(r)
        e_o = self.emb_ent(o)

        phi = torch.cat([e_s, e_r, e_o], 1)

        if self.num_lit:
            X_lit_s = Variable(torch.from_numpy(X_lit_s))
            X_lit_s = X_lit_s.cuda() if self.gpu else X_lit_s

            X_lit_o = Variable(torch.from_numpy(X_lit_o))
            X_lit_o = X_lit_o.cuda() if self.gpu else X_lit_o

            phi = torch.cat([phi, X_lit_s, X_lit_o], 1)

        if self.img_lit:
            X_img_s = Variable(torch.from_numpy(X_lit_s_img))
            X_img_s = X_img_s.cuda() if self.gpu else X_img_s
            e_img_s = self.emb_img(X_img_s)

            X_img_o = Variable(torch.from_numpy(X_lit_o_img))
            X_img_o = X_img_o.cuda() if self.gpu else X_img_o
            e_img_o = self.emb_img(X_img_o)

            phi = torch.cat([phi, e_img_s, e_img_o], 1)

        if self.txt_lit:
            X_txt_s = Variable(torch.from_numpy(X_lit_s_txt))
            X_txt_s = X_txt_s.cuda() if self.gpu else X_txt_s
            e_txt_s = self.emb_txt(X_txt_s)

            X_txt_o = Variable(torch.from_numpy(X_lit_o_txt))
            X_txt_o = X_txt_o.cuda() if self.gpu else X_txt_o
            e_txt_o = self.emb_txt(X_txt_o)

            phi = torch.cat([phi, e_txt_s, e_txt_o], 1)

        score = self.mlp(phi)

        return score

    def predict(self, X, X_lit_s, X_lit_o, X_lit_s_img, X_lit_o_img, X_lit_s_txt, X_lit_o_txt):
        y_pred = self.forward(X, X_lit_s, X_lit_o, X_lit_s_img, X_lit_o_img,
                              X_lit_s_txt, X_lit_o_txt).view(-1, 1)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()

    def predict_all(self, X, **kwargs):
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
        e_o_rep = e_o.repeat(self.n_ent, 1)  # n_ent x k

        phi_s = torch.cat([self.emb_ent.weight, e_r_rep, e_o_rep], 1)  # n_ent x 3k
        phi_o = torch.cat([e_s_rep, e_r_rep, self.emb_ent.weight], 1)  # n_ent x 3k

        if self.num_lit:
            X_lit = Variable(torch.from_numpy(kwargs['X_lit']))
            X_lit = X_lit.cuda() if self.gpu else X_lit

            X_lit_s_rep = X_lit[s].repeat(self.n_ent, 1)
            X_lit_o_rep = X_lit[o].repeat(self.n_ent, 1)

            phi_s = torch.cat([phi_s, X_lit, X_lit_o_rep], 1)
            phi_o = torch.cat([phi_o, X_lit_s_rep, X_lit], 1)

        if self.img_lit:
            X_img = Variable(torch.from_numpy(kwargs['X_lit_img']))
            X_img = X_img.cuda() if self.gpu else X_img
            e_img = self.emb_img(X_img)

            e_img_s_rep = e_img[s].repeat(self.n_ent, 1)
            e_img_o_rep = e_img[o].repeat(self.n_ent, 1)

            phi_s = torch.cat([phi_s, e_img, e_img_o_rep], 1)
            phi_o = torch.cat([phi_o, e_img_s_rep, e_img], 1)

        if self.txt_lit:
            X_txt = Variable(torch.from_numpy(kwargs['X_lit_txt']))
            X_txt = X_txt.cuda() if self.gpu else X_txt
            e_txt = self.emb_txt(X_txt)

            e_txt_s_rep = e_txt[s].repeat(self.n_ent, 1)
            e_txt_o_rep = e_txt[o].repeat(self.n_ent, 1)

            phi_s = torch.cat([phi_s, e_txt, e_txt_o_rep], 1)
            phi_o = torch.cat([phi_o, e_txt_s_rep, e_txt], 1)

        # Predict
        y_s = self.mlp(phi_s).view(-1)
        y_o = self.mlp(phi_o).view(-1)

        return y_s, y_o


@inherit_docstrings
class DistMult_MovieLens(Model):
    """
    DistMult: diagonal bilinear model, without subject and object constraint
    ------------------------------------------------------------------------
    Yang, Bishan, et al. "Learning multi-relational semantics using
    neural-embedding models." arXiv:1411.4072 (2014).
    """

    def __init__(self, n_s, n_r, n_o, k, lam, gpu=False):
        super(DistMult_MovieLens, self).__init__(gpu)

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


@inherit_docstrings
class ERMLP_literal1(Model):
    """
    ER-MLP: Entity-Relation MLP
    ---------------------------
    Dong, Xin, et al. "Knowledge vault: A web-scale approach to probabilistic knowledge fusion." KDD, 2014.
    """

    def __init__(self, n_e, n_r, k, h_dim, p, lam, n_numeric, n_text, dim_text, numeric=True, text=True, gpu=False):
        super(ERMLP_literal1, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.h_dim = h_dim
        self.p = p
        self.lam = lam
        self.n_numeric = n_numeric
        self.n_text = n_text
        self.dim_text = dim_text
        self.numeric = numeric
        self.text = text

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)

        # Determine MLP input size
        n_input = 3*k

        if numeric:
            n_input += 2*n_numeric
        if text:
            self.attn_weights_s = nn.Parameter(torch.randn(self.n_text, 1))
            self.attn_weights_o = nn.Parameter(torch.randn(self.n_text, 1))
            n_input += 2*dim_text

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

    def forward(self, X, numeric_lit_s, numeric_lit_o, text_lit_s, text_lit_o):
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)

        # Forward
        phi = torch.cat([e_hs, e_ts, e_ls], 1)

        if self.numeric:
            numeric_lit_s = Variable(torch.from_numpy(numeric_lit_s))
            numeric_lit_s = numeric_lit_s.cuda() if self.gpu else numeric_lit_s

            numeric_lit_o = Variable(torch.from_numpy(numeric_lit_o))
            numeric_lit_o = numeric_lit_o.cuda() if self.gpu else numeric_lit_o

            phi = torch.cat([phi, numeric_lit_s, numeric_lit_o], 1)

        if self.text:
            text_lit_s = Variable(torch.from_numpy(text_lit_s))
            text_lit_s = text_lit_s.cuda() if self.gpu else text_lit_s

            text_lit_o = Variable(torch.from_numpy(text_lit_o))
            text_lit_o = text_lit_o.cuda() if self.gpu else text_lit_o

            weighted_text_s = torch.bmm(self.attn_weights_s.t().unsqueeze(0).repeat(len(X), 1, 1), text_lit_s).view(len(X), self.dim_text)
            weighted_text_o = torch.bmm(self.attn_weights_o.t().unsqueeze(0).repeat(len(X), 1, 1), text_lit_o).view(len(X), self.dim_text)

            phi = torch.cat([phi, weighted_text_s, weighted_text_o], 1)

        y = self.mlp(phi)

        return y.view(-1, 1)

    def predict(self, X, numeric_lit_s, numeric_lit_o, text_lit_s, text_lit_o):
        y_pred = self.forward(X, numeric_lit_s, numeric_lit_o, text_lit_s, text_lit_o).view(-1, 1)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()

    def predict_all(self, X, **kwargs):
        # WIP!
        """
        Let X be a triple (s, p, o), i.e. tensor of 1x3, return two lists:
            - list of (s, p, all_others)
            - list of (all_others, p, o)
        Pass all of the (full matrix of) literals into kwargs.
        """
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        s, p, o = X[:, 0], X[:, 1], X[:, 2]

        # Load literals
        if self.numeric:
            num_lit_s = Variable(kwargs['numeric_lit_s'])
            num_lit_s = num_lit_s.cuda() if self.gpu else num_lit_s
            num_lit_o = Variable(kwargs['numeric_lit_o'])
            num_lit_o = num_lit_o.cuda() if self.gpu else num_lit_o

        if self.text:
            txt_lit_s = Variable(kwargs['text_lit_s'])
            txt_lit_s = txt_lit_s.cuda() if self.gpu else txt_lit_s
            txt_lit_o = Variable(kwargs['text_lit_o'])
            txt_lit_o = txt_lit_o.cuda() if self.gpu else txt_lit_o

            weighted_text_s = torch.bmm(self.attn_weights_s.t().unsqueeze(0).repeat(len(X), 1, 1), text_lit_s).view(len(X), self.dim_text)
            weighted_text_o = torch.bmm(self.attn_weights_o.t().unsqueeze(0).repeat(len(X), 1, 1), text_lit_o).view(len(X), self.dim_text)

        # Project to embedding, each is M x k
        e_s = self.emb_ent(s)
        e_r = self.emb_rel(p)
        e_o = self.emb_ent(o)

        # Predict o
        e_s_rep = e_s.repeat(self.n_e, 1)  # n_e x k
        e_r_rep = e_r.repeat(self.n_e, 1)  # n_e x k

        phi_o = torch.cat([e_s_rep, e_r_rep, self.emb_E.weight], 1)

        if self.numeric:
            num_lit_s_rep = num_lit_s[s].repeat(self.n_e, 1)
            phi_o = torch.cat([phi_o, num_lit_s_rep, num_lit_o], 1)

        if self.text:
            txt_lit_s_rep = weighted_text_s[s].repeat(self.n_e, 1)
            phi_o = torch.cat([phi_o, txt_lit_s_rep, weighted_text_o], 1)

        y_o = self.ermlp(phi_o).view(-1)

        # Predict s
        e_o_rep = e_o.repeat(self.n_e, 1)

        phi_s = torch.cat([self.emb_E.weight, e_r_rep, e_o_rep], 1)

        if self.numeric:
            num_lit_o_rep = num_lit_o[o].repeat(self.n_e, 1)
            phi_s = torch.cat([phi_s, num_lit_s, num_lit_o_rep], 1)

        if self.text:
            txt_lit_o_rep = weighted_text_o[o].repeat(self.n_e, 1)
            phi_s = torch.cat([phi_s, weighted_text_s, txt_lit_o_rep], 1)

        y_s = self.ermlp(phi_s).view(-1)

        return y_s, y_o


@inherit_docstrings
class ERMLP_literal2(Model):
    """
    ER-MLP: Entity-Relation MLP
    ---------------------------
    Dong, Xin, et al. "Knowledge vault: A web-scale approach to probabilistic knowledge fusion." KDD, 2014.
    """

    def __init__(self, n_e, n_r, k, h_dim, p, lam, n_numeric, vocab_size, dim_text, pretrained_embeddings, batch_size, text_length, numeric=True, text=True, gpu=False):
        super(ERMLP_literal2, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.h_dim = h_dim
        self.p = p
        self.lam = lam
        self.n_numeric = n_numeric
        self.vocab_size = vocab_size
        self.dim_text = dim_text
        self.numeric = numeric
        self.text = text
        self.batch_size = batch_size
        self.text_length = text_length

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)

        # Determine MLP input size
        n_input = 3*k

        if numeric:
            n_input += 2*n_numeric
        if text:
            n_input += 2*self.k
            self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_text)
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

            self.lstm_s = nn.LSTM(self.dim_text, self.k)
            self.lstm_o = nn.LSTM(self.dim_text, self.k)

            self.hidden = self.init_hidden()

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

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        h = Variable(torch.zeros(1, self.batch_size, self.k))
        h = h.cuda() if self.gpu else h
        c = Variable(torch.zeros(1, self.batch_size, self.k))
        c = c.cuda() if self.gpu else c

        return (h, c)

    def forward(self, X, numeric_lit_s, numeric_lit_o, text_lit_s, text_lit_o):
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)

        phi = torch.cat([e_hs, e_ts, e_ls], 1)

        if self.numeric:
            numeric_lit_s = Variable(torch.from_numpy(numeric_lit_s))
            numeric_lit_s = numeric_lit_s.cuda() if self.gpu else numeric_lit_s

            numeric_lit_o = Variable(torch.from_numpy(numeric_lit_o))
            numeric_lit_o = numeric_lit_o.cuda() if self.gpu else numeric_lit_o

            phi = torch.cat([phi, numeric_lit_s, numeric_lit_o], 1)

        if self.text:
            text_lit_s = Variable(torch.from_numpy(text_lit_s))
            text_lit_s = text_lit_s.cuda() if self.gpu else text_lit_s
            text_lit_o = Variable(torch.from_numpy(text_lit_o))
            text_lit_o = text_lit_o.cuda() if self.gpu else text_lit_o

            embed_lit_s = self.word_embeddings(text_lit_s.t())
            embed_lit_o = self.word_embeddings(text_lit_o.t())

            x_s = embed_lit_s.view(self.text_length, self.batch_size, -1)
            lstm_s_out, self.hidden = self.lstm_s(x_s, self.hidden)
            lstm_s_out = lstm_s_out[-1]

            x_p = embed_lit_o.view(self.text_length, self.batch_size, -1)
            lstm_o_out, self.hidden = self.lstm_o(x_p, self.hidden)
            lstm_o_out = lstm_o_out[-1]

            phi = torch.cat([phi, lstm_s_out, lstm_o_out], 1)

        y = self.mlp(phi)

        return y.view(-1, 1)

    def predict(self, X, numeric_lit_s, numeric_lit_o, text_lit_s, text_lit_o):
        y_pred = self.forward(X, numeric_lit_s, numeric_lit_o, text_lit_s, text_lit_o).view(-1, 1)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()

    def predict_all(self, X, **kwargs):
        # WIP!
        """
        Let X be a triple (s, p, o), i.e. tensor of 1x3, return two lists:
            - list of (s, p, all_others)
            - list of (all_others, p, o)
        Pass all of the (full matrix of) literals into kwargs.
        """
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        s, p, o = X[:, 0], X[:, 1], X[:, 2]

        # Load literals
        if self.numeric:
            num_lit_s = Variable(kwargs['numeric_lit_s'])
            num_lit_s = num_lit_s.cuda() if self.gpu else num_lit_s
            num_lit_o = Variable(kwargs['numeric_lit_o'])
            num_lit_o = num_lit_o.cuda() if self.gpu else num_lit_o

        if self.text:
            txt_lit_s = Variable(kwargs['text_lit_s'])
            txt_lit_s = txt_lit_s.cuda() if self.gpu else txt_lit_s
            txt_lit_o = Variable(kwargs['text_lit_o'])
            txt_lit_o = txt_lit_o.cuda() if self.gpu else txt_lit_o

            embed_lit_s = self.word_embeddings(txt_lit_s.t())
            embed_lit_o = self.word_embeddings(txt_lit_o.t())

            x_s = embed_lit_s.view(self.text_length, len(txt_lit_s), -1)
            lstm_s_out, self.hidden = self.lstm_s(x_s, self.hidden)
            lstm_s_out = lstm_s_out[-1]

            x_p = embed_lit_o.view(self.text_length, len(txt_lit_o), -1)
            lstm_o_out, self.hidden = self.lstm_o(x_p, self.hidden)
            lstm_o_out = lstm_o_out[-1]

        # Project to embedding, each is M x k
        e_s = self.emb_ent(s)
        e_r = self.emb_rel(p)
        e_o = self.emb_ent(o)

        # Predict o
        e_s_rep = e_s.repeat(self.n_e, 1)  # n_e x k
        e_r_rep = e_r.repeat(self.n_e, 1)  # n_e x k

        phi_o = torch.cat([e_s_rep, e_r_rep, self.emb_E.weight], 1)

        if self.numeric:
            num_lit_s_rep = num_lit_s[s].repeat(self.n_e, 1)
            phi_o = torch.cat([phi_o, num_lit_s_rep, num_lit_o], 1)

        if self.text:
            txt_lit_s_rep = lstm_s_out[s].repeat(self.n_e, 1)
            phi_o = torch.cat([phi_o, txt_lit_s_rep, lstm_o_out], 1)

        y_o = self.ermlp(phi_o).view(-1)

        # Predict s
        e_o_rep = e_o.repeat(self.n_e, 1)

        phi_s = torch.cat([self.emb_E.weight, e_r_rep, e_o_rep], 1)

        if self.numeric:
            num_lit_o_rep = num_lit_o[o].repeat(self.n_e, 1)
            phi_s = torch.cat([phi_s, num_lit_s, num_lit_o_rep], 1)

        if self.text:
            txt_lit_o_rep = lstm_o_out[o].repeat(self.n_e, 1)
            phi_s = torch.cat([phi_s, lstm_s_out, txt_lit_o_rep], 1)

        y_s = self.ermlp(phi_s).view(-1)

        return y_s, y_o


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
        super(RESCAL_literal, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.lam = lam
        self.n_l = n_l
        self.n_text = n_text

        if self.n_text is not None:
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
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        if self.n_l is not None:
            s_lit = Variable(torch.from_numpy(s_lit))
            s_lit = s_lit.cuda() if self.gpu else s_lit
            o_lit = Variable(torch.from_numpy(o_lit))
            o_lit = o_lit.cuda() if self.gpu else o_lit
        if self.n_text is not None:
            text_s = Variable(torch.from_numpy(text_s))
            text_s = text_s.cuda() if self.gpu else text_s
            text_o = Variable(torch.from_numpy(text_o))
            text_o = text_o.cuda() if self.gpu else text_o

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)

        if self.n_l is not None:
            e1_rep = torch.cat([e_hs, s_lit], 1)  # M x (k + n_l)
            e2_rep = torch.cat([e_ts, o_lit], 1)  # M x (k + n_l)

            e1_rep = self.mlp(e1_rep).view(-1, self.k, 1)   # M x k x 1
            e2_rep = self.mlp(e2_rep).view(-1, self.k, 1)   # M x k x 1
        elif self.n_text is not None:
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

    def predict_all(self, X):
        raise NotImplementedError()


@inherit_docstrings
class DistMultLiteral(Model):
    """
    DistMult: diagonal bilinear model
    ---------------------------------
    Yang, Bishan, et al. "Learning multi-relational semantics using
    neural-embedding models." arXiv:1411.4072 (2014).
    """

    def __init__(self, n_e, n_r, n_l, k, gpu=False):
        super(DistMultLiteral, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.n_l = n_l
        self.k = k

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)
        self.emb_E_lit = nn.Linear(k+n_l, self.k)

        self.embeddings = [self.emb_E, self.emb_R]
        self.initialize_embeddings()

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X, X_lit_s, X_lit_o):
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        s, p, o = X[:, 0], X[:, 1], X[:, 2]

        X_lit_s = Variable(torch.from_numpy(X_lit_s))
        X_lit_s = X_lit_s.cuda() if self.gpu else X_lit_s
        X_lit_o = Variable(torch.from_numpy(X_lit_o))
        X_lit_o = X_lit_o.cuda() if self.gpu else X_lit_o

        # Project to embedding, each is M x k
        s = self.emb_E_lit(torch.cat([self.emb_E(s), X_lit_s], 1))
        o = self.emb_E_lit(torch.cat([self.emb_E(o), X_lit_o], 1))
        W = self.emb_R(p)

        # Forward
        f = torch.sum(s * W * o, 1)

        return f.view(-1, 1)

    def predict(self, X, s_lit, o_lit, sigmoid=False):
        y_pred = self.forward(X, s_lit, o_lit).view(-1, 1)

        if sigmoid:
            y_pred = F.sigmoid(y_pred)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()

    def predict_all(self, X, **kwargs):
        # Relations
        X = Variable(torch.from_numpy(X)).long()
        X = X.cuda() if self.gpu else X

        s, p, o = X[:, 0], X[:, 1], X[:, 2]

        # Literals
        X_lit = kwargs['X_lit']
        X_lit = Variable(torch.from_numpy(X_lit))
        X_lit = X_lit.cuda() if self.gpu else X_lit

        X_lit_s = X_lit[s]
        X_lit_o = X_lit[o]

        # 1 x k
        s = self.emb_E_lit(torch.cat([self.emb_E(s), X_lit_s], 1))
        o = self.emb_E_lit(torch.cat([self.emb_E(o), X_lit_o], 1))
        W = self.emb_R(p)

        # n_e x k
        all_ents = self.emb_E_lit(torch.cat([self.emb_E.weight, X_lit], 1))

        # <(1xk \odot 1xk), k x n_e> = 1 x n_e
        y_s = torch.mm(W * o, all_ents.t()).view(-1)
        y_o = torch.mm(s * W, all_ents.t()).view(-1)
        return y_s, y_o
