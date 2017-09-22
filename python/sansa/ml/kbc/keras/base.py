import keras
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from keras.optimizers import Adagrad, SGD
import math
import theano.tensor as T
import numpy as np
from numpy.random import shuffle
from collections import defaultdict, Counter
from sansa.ml.kbc.keras.param import Parameter, AdaGrad
import timeit
import pickle

_cutoff = 30

_DEF_NBATCHES = 100
_DEF_POST_EPOCH = []
_DEF_LEARNING_RATE = 0.1
_DEF_SAMPLE_FUN = None
_DEF_MAX_EPOCHS = 1000
_DEF_MARGIN = 1.0


class Config(object):

    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer

    def __getstate__(self):
        return {
            'model': self.model,
            'trainer': self.trainer
        }


class Model(object):
    """
    Base class for all Knowledge Graph models

    Implements basic setup routines for parameters and serialization methods

    Subclasses need to implement:
    - scores(self, ss, ps, os)
    - _gradients(self, xys) for StochasticTrainer
    - _pairwise_gradients(self, pxs, nxs) for PairwiseStochasticTrainer
    """

    def __init__(self, *args, **kwargs):
        #super(Model, self).__init__(*args, **)
        self.params = {}
        self.hyperparams = {}
        self.add_hyperparam('init', kwargs.pop('init', 'nunif'))

    def add_param(self, param_id, shape, post=None, value=None):
        if value is None:
            value = Parameter(shape, self.init, name=param_id, post=post)
        setattr(self, param_id, value)
        self.params[param_id] = value

    def add_hyperparam(self, param_id, value):
        setattr(self, param_id, value)
        self.hyperparams[param_id] = value

    def __getstate__(self):
        return {
            'hyperparams': self.hyperparams,
            'params': self.params
        }

    def __setstate__(self, st):
        self.params = {}
        self.hyperparams = {}
        for pid, p in st['params'].items():
            self.add_param(pid, None, None, value=p)
        for pid, p in st['hyperparams'].items():
            self.add_hyperparam(pid, p)

    def save(self, fname, protocol=pickle.HIGHEST_PROTOCOL):
        with open(fname, 'wb') as fout:
            pickle.dump(self, fout, protocol=protocol)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as fin:
            mdl = pickle.load(fin)
        return mdl

class KerasTrainer(object):
    """
    Keras model trainer
    """

    def __init__(self, *args, **kwargs):
        self.model = args[0]
        self.hyperparams = {}
        self.add_hyperparam('max_epochs', kwargs.pop('max_epochs', _DEF_MAX_EPOCHS))
        self.add_hyperparam('nbatches', kwargs.pop('nbatches', _DEF_NBATCHES))
        self.add_hyperparam('learning_rate', kwargs.pop('learning_rate', _DEF_LEARNING_RATE))

        self.post_epoch = kwargs.pop('post_epoch', _DEF_POST_EPOCH)
        self.samplef = kwargs.pop('samplef', _DEF_SAMPLE_FUN)

    def __getstate__(self):
        return self.hyperparams

    def __setstate__(self, st):
        for pid, p in st['hyperparams']:
            self.add_hyperparam(pid, p)

    def add_hyperparam(self, param_id, value):
        setattr(self, param_id, value)
        self.hyperparams[param_id] = value

    def fit(self, xs, ys):
        self._optim(list(zip(xs, ys)))


    def getModel(self):
        # # Training stuff
        # batch_placeholder = K.placeholder(shape=(3,), name="batch")
        # label_placeholder = K.placeholder(shape=(1,), name="label")

        # Model stuff
        # E = K.variable(self.model.E, name="entity_embeddings")
        # R = K.variable(self.model.R, name="relation_embeddings")
        model = Sequential()
        model.add(HolographicLayerTest(self.model.E.shape[0], self.model.R.shape[0], self.model.E.shape[1], self.model.rparam))
        # model.add(Dense(5, input_dim=(10,)))
        # model.add(Activation('sigmoid'))
        # adagrad = Adagrad(lr=0.001, epsilon=1e-07)
        adagrad = SGD(lr=0.001, decay=1e-06, nesterov=True, momentum=0.5)

        def loss(y_true, y_pred):
            print(y_pred)
            return -K.sum(K.log(K.sigmoid(-y_true * y_pred)))

        print("Compiling new model")
        model.compile(optimizer=adagrad, loss=loss)
        return model

    def _pre_epoch(self):
        self.loss = 0

    def _optim(self, xys):
        idx = np.arange(len(xys))
        # self.batch_size = np.ceil(len(xys) / self.nbatches)

        # batch_idx = np.arange(self.batch_size, len(xys), self.batch_size)

        model = self.getModel()

        for self.epoch in range(1, self.max_epochs + 1):
            # shuffle training examples
            self._pre_epoch()
            shuffle(idx)

            # store epoch for callback
            self.epoch_start = timeit.default_timer()

            # process mini-batches
            # for batch in np.split(idx, batch_idx):
            #     # select indices for current batch
            #     bxys = [xys[z] for z in batch]
            #     self._process_batch(bxys, model)

            self._process_batch(xys, model)

            # check callback function, if false return
            for f in self.post_epoch:
                if not f(self):
                    break

        #
        # print (self.model.E.shape)
        # print (self.model.R.shape)

    def _process_batch(self, xys, model):
        # if enabled, sample additional examples
        if self.samplef is not None:
            xys += self.samplef(xys)

        if hasattr(self.model, '_prepare_batch_step'):
            self.model._prepare_batch_step(xys)

        shuffle(xys)
        # take step for batch
        assert isinstance(model, keras.models.Model)
        xs, ys = [np.array(i) for i in list(zip(*xys))]
        # print(xs, ys)
        # print(xs.shape, ys.shape)

        class LossHistory(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.loss = -1

            def on_batch_end(self, batch, logs={}):
                self.loss = logs.get('loss')

        history = LossHistory()
        # print(Counter(ys))
        # x = K.placeholder(shape=(1,3))
        # func = K.function([x], model(x))
        # for i, j in zip(xs, ys):
        #     print(func([[i]]), j)
        model.fit(xs, ys, batch_size=len(xs)/100, nb_epoch=100, callbacks=[history])
        loss = history.loss
        # loss = model.train_on_batch(xs, ys)
        E, R = model.layers[0].get_weights()
        # print (np.linalg.norm(self.model.E-E, 'fro'))
        # print (np.linalg.norm(self.model.R-R, 'fro'))
        self.model.E, self.model.R = E, R

        # print(loss)

        self.loss += loss
        # print (acc)

class HolographicLayer(Layer):
    def __init__(self, E, R, rparam, input_shape=(3,), **kwargs):
        self.E = E
        self.R = R
        self.rparam = rparam
        kwargs["input_shape"] = input_shape
        super(HolographicLayer, self).__init__(**kwargs)

    def ccorr1d_sc(self, input, filters, image_shape=None, filter_shape=None,
              border_mode='valid', subsample=(1,), filter_flip=True):
        """
        using conv2d with a single input channel
        """
    #     if border_mode not in ('valid', 0, (0,)):
    #         raise RuntimeError("Unsupported border_mode for conv1d_sc: "
    #                            "%s" % border_mode)

        if image_shape is None:
            image_shape_sc = None
        else:
            # (b, c, i0) to (b, 1, c, i0)
            image_shape_sc = (image_shape[0], 1, image_shape[1], image_shape[2])

        if filter_shape is None:
            filter_shape_sc = None
        else:
            filter_shape_sc = (filter_shape[0], 1, filter_shape[1],
                               filter_shape[2])

        input_sc = input.dimshuffle('x', 'x', 0, 'x')
        # We need to flip the channels dimension because it will be convolved over.
        filters_sc = filters.dimshuffle('x', 'x', 0, 'x')[:, :, ::-1, :]

        conved = T.nnet.conv2d(input_sc, filters_sc, image_shape_sc,
                           filter_shape_sc, subsample=(1, subsample[0]),
                           filter_flip=filter_flip, border_mode=border_mode).flatten()
        return conved  # drop the unused dimension

    def build(self, input_shape):
        self.trainable_weights = [self.E, self.R]
        # from keras.regularizers import l2
        # regularizer = l2(self.rparam)
        # regularizer.set_param(K.concatenate([self.E, self.R], axis=0))
        # self.regularizers.append(regularizer)

    def call(self, x, mask=None):
        batch_placeholder = K.cast(x, 'int32')[0]
        s, o, p = [batch_placeholder[i] for i in range(3)]

        s2v = K.gather(self.E, s)
        o2v = K.gather(self.E, o)
        r2v = K.gather(self.R, p)

        # print(K.shape(s2v).eval())
        # print(self.E[[0]].shape.eval())

        def ccorr(a, b):
            return self.ccorr1d_sc(a, b, border_mode='half')

        eta = K.dot(K.transpose(r2v), ccorr(s2v, o2v))
        return eta

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1)

class HolographicLayerTest(Layer):
    def __init__(self, E, R, d, rparam, input_shape=(3,), **kwargs):
        bnd = math.sqrt(6) / math.sqrt(2*E)
        from numpy.random import uniform
        self.init = [K.variable(uniform(size=(E,d), low=-bnd, high=bnd), name="E"),
                     K.variable(uniform(size=(R,d*d), low=-bnd, high=bnd), name="R")]
        self.rparam = rparam
        kwargs["input_shape"] = input_shape
        super(HolographicLayerTest, self).__init__(**kwargs)

    def ccorr1d_sc(self, input, filters, image_shape=None, filter_shape=None,
              border_mode='valid', subsample=(1,), filter_flip=True):
        """
        using conv2d with a single input channel
        """
    #     if border_mode not in ('valid', 0, (0,)):
    #         raise RuntimeError("Unsupported border_mode for conv1d_sc: "
    #                            "%s" % border_mode)

        if image_shape is None:
            image_shape_sc = None
        else:
            # (b, c, i0) to (b, 1, c, i0)
            image_shape_sc = (image_shape[0], 1, image_shape[1], image_shape[2])

        if filter_shape is None:
            filter_shape_sc = None
        else:
            filter_shape_sc = (filter_shape[0], 1, filter_shape[1],
                               filter_shape[2])

        input_sc = input.dimshuffle('x', 'x', 0, 'x')
        # We need to flip the channels dimension because it will be convolved over.
        filters_sc = filters.dimshuffle('x', 'x', 0, 'x')[:, :, ::-1, :]

        conved = T.nnet.conv2d(input_sc, filters_sc, image_shape_sc,
                           filter_shape_sc, subsample=(1, subsample[0]),
                           filter_flip=filter_flip, border_mode=border_mode).flatten()
        return conved  # drop the unused dimension

    def build(self, input_shape):
        self.E, self.R = self.init
        self.trainable_weights = [self.E, self.R]
        # from keras.regularizers import l2
        # regularizer = l2(self.rparam)
        # regularizer.set_param(self.E)
        # self.regularizers.append(regularizer)
        #
        # regularizer = l2(self.rparam)
        # regularizer.set_param(self.R)
        # self.regularizers.append(regularizer)

    def call(self, x, mask=None):
        batch_placeholder = K.cast(x, 'int32')[0]
        s, o, p = [batch_placeholder[i] for i in range(3)]

        s2v = K.gather(self.E, s)
        o2v = K.gather(self.E, o)
        r2v = K.gather(self.R, p)

        def ccorr(a, b):
            return T.outer(a,b).flatten()
            # return self.ccorr1d_sc(a, b, border_mode='half')
        eta = K.dot(r2v, ccorr(s2v, o2v))

        # func = K.function([s2v,o2v,r2v], K.gradients(K.sigmoid(eta), [s2v,o2v,r2v]))
        # print(func([np.random.random(150),np.random.random(150),np.random.random(150)]))

        return eta

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1)

class TheanoGradTest(object):
    def ccorr1d_sc(self, input, filters, image_shape=None, filter_shape=None,
              border_mode='valid', subsample=(1,), filter_flip=True):
        """
        using conv2d with a single input channel
        """
    #     if border_mode not in ('valid', 0, (0,)):
    #         raise RuntimeError("Unsupported border_mode for conv1d_sc: "
    #                            "%s" % border_mode)

        if image_shape is None:
            image_shape_sc = None
        else:
            # (b, c, i0) to (b, 1, c, i0)
            image_shape_sc = (image_shape[0], 1, image_shape[1], image_shape[2])

        if filter_shape is None:
            filter_shape_sc = None
        else:
            filter_shape_sc = (filter_shape[0], 1, filter_shape[1],
                               filter_shape[2])

        input_sc = input.dimshuffle('x', 'x', 0, 'x')
        # We need to flip the channels dimension because it will be convolved over.
        filters_sc = filters.dimshuffle('x', 'x', 0, 'x')[:, :, ::-1, :]

        conved = T.nnet.conv2d(input_sc, filters_sc, image_shape_sc,
                           filter_shape_sc, subsample=(1, subsample[0]),
                           filter_flip=filter_flip, border_mode=border_mode).flatten()
        return conved  # drop the unused dimension


    def call(self):
        E = K.variable(np.random.random((1000,100)), name="entity_embeddings")
        R = K.variable(np.random.random((10,10000)), name="relation_embeddings")
        x = K.placeholder(shape=(1,3), name="spo")
        y = K.placeholder(ndim=0, name="y")
        batch_placeholder = K.cast(x, 'int32')[0]
        # print(batch_placeholder.eval())
        s, o, p = [batch_placeholder[i] for i in range(3)]

        s2v = K.gather(E, s)
        o2v = K.gather(E, o)
        r2v = K.gather(R, p)

        def ccorr(a, b):
            return T.outer(a,b).flatten()
            # return T.arctan(s2v) + T.arctan(o2v)
            # return (s2v.dimshuffle('x', 'x', 0, 'x') + o2v.dimshuffle('x', 'x', 0, 'x')).flatten()
            # return T.nnet.conv2d(a.dimshuffle('x', 'x', 0, 'x'), b.dimshuffle('x', 'x', 0, 'x'), None,
            #                None,
            #                filter_flip=True, border_mode='half')
            # return self.ccorr1d_sc(a, b, border_mode='half')
        eta = K.dot(r2v, ccorr(s2v, o2v))
        # py = 1/(1+K.exp(-eta))
        # l = -K.log(py)
        # from theano import pp, function, printing
        # grad = T.grad(eta, E)
        # print(pp(grad))
        # func = function([x], grad)
        func = K.function([x, y], K.gradients(eta, [s2v, o2v, r2v, E, R]))

        # for i in func.maker.fgraph.outputs:
            # print(pp(i))
        # print (T.grad(py, s2v))
        print (func([[[1,2,3]], -1]))

class StochasticTrainer(object):
    """
    Stochastic gradient descent trainer with scalar loss function.

    Models need to implement

    _gradients(self, xys)

    to be trained with this class.

    """

    def __init__(self, *args, **kwargs):
        self.model = args[0]
        self.hyperparams = {}
        self.add_hyperparam('max_epochs', kwargs.pop('max_epochs', _DEF_MAX_EPOCHS))
        self.add_hyperparam('nbatches', kwargs.pop('nbatches', _DEF_NBATCHES))
        self.add_hyperparam('learning_rate', kwargs.pop('learning_rate', _DEF_LEARNING_RATE))

        self.post_epoch = kwargs.pop('post_epoch', _DEF_POST_EPOCH)
        self.samplef = kwargs.pop('samplef', _DEF_SAMPLE_FUN)
        pu = kwargs.pop('param_update', AdaGrad)
        self._updaters = {
            key: pu(param, self.learning_rate)
            for key, param in self.model.params.items()
        }

    def __getstate__(self):
        return self.hyperparams

    def __setstate__(self, st):
        for pid, p in st['hyperparams']:
            self.add_hyperparam(pid, p)

    def add_hyperparam(self, param_id, value):
        setattr(self, param_id, value)
        self.hyperparams[param_id] = value

    def fit(self, xs, ys):
        self._optim(list(zip(xs, ys)))

    def _pre_epoch(self):
        self.loss = 0

    def _optim(self, xys):
        idx = np.arange(len(xys))
        self.batch_size = np.ceil(len(xys) / self.nbatches)
        batch_idx = np.arange(self.batch_size, len(xys), self.batch_size)

        for self.epoch in range(1, self.max_epochs + 1):
            # shuffle training examples
            self._pre_epoch()
            shuffle(idx)

            # store epoch for callback
            self.epoch_start = timeit.default_timer()

            # process mini-batches
            for batch in np.split(idx, batch_idx):
                # select indices for current batch
                bxys = [xys[z] for z in batch]
                self._process_batch(bxys)

            # check callback function, if false return
            for f in self.post_epoch:
                if not f(self):
                    break

    def _process_batch(self, xys):
        # if enabled, sample additional examples
        if self.samplef is not None:
            xys += self.samplef(xys)

        if hasattr(self.model, '_prepare_batch_step'):
            self.model._prepare_batch_step(xys)

        # take step for batch
        grads = self.model._gradients(xys)
        self.loss += self.model.loss
        self._batch_step(grads)

    def _batch_step(self, grads):
        for paramID in self._updaters.keys():
            self._updaters[paramID](*grads[paramID])


class PairwiseStochasticTrainer(StochasticTrainer):
    """
    Stochastic gradient descent trainer with pairwise ranking loss functions.

    Models need to implement

    _pairwise_gradients(self, pxs, nxs)

    to be trained with this class.

    """


    def __init__(self, *args, **kwargs):
        super(PairwiseStochasticTrainer, self).__init__(*args, **kwargs)
        self.model.add_hyperparam('margin', kwargs.pop('margin', _DEF_MARGIN))

    def fit(self, xs, ys):
        if self.samplef is None:
            pidx = np.where(np.array(ys) == 1)[0]
            nidx = np.where(np.array(ys) != 1)[0]
            pxs = [xs[i] for i in pidx]
            self.nxs = [xs[i] for i in nidx]
            self.pxs = int(len(self.nxs) / len(pxs)) * pxs
            xys = list(range(min(len(pxs), len(self.nxs))))
            self._optim(xys)
        else:
            self._optim(list(zip(xs, ys)))

    def _pre_epoch(self):
        self.nviolations = 0
        if self.samplef is None:
            shuffle(self.pxs)
            shuffle(self.nxs)

    def _process_batch(self, xys):
        pxs = []
        nxs = []

        for xy in xys:
            if self.samplef is not None:
                for nx in self.samplef([xy]):
                    pxs.append(xy)
                    nxs.append(nx)
            else:
                pxs.append((self.pxs[xy], 1))
                nxs.append((self.nxs[xy], 1))

        # take step for batch
        if hasattr(self.model, '_prepare_batch_step'):
            self.model._prepare_batch_step(pxs, nxs)
        grads = self.model._pairwise_gradients(pxs, nxs)

        # update if examples violate margin
        if grads is not None:
            self.nviolations += self.model.nviolations
            self._batch_step(grads)


def sigmoid(fs):
    # compute elementwise gradient for sigmoid
    for i in range(len(fs)):
        if fs[i] > _cutoff:
            fs[i] = 1.0
        elif fs[i] < -_cutoff:
            fs[i] = 0.0
        else:
            fs[i] = 1.0 / (1 + np.exp(-fs[i]))
    return fs[:, np.newaxis]

if __name__ =="__main__":
    TheanoGradTest().call()

