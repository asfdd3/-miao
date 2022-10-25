# import copy
import logging
import os

from absl import app
from absl import flags
# import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bgrl import *
from bgrl.bgrl import *
from bgrl.bgrl import create_sparse
from bgrl.transforms import degree_drop_weights, pr_drop_weights, evc_drop_weights, compute_pr,feature_drop_weights, feature_drop_weights_dense, drop_feature_weighted_2, eigenvector_centrality,drop_edge_weighted, dropout_adj, drop_feature
from torch_geometric.utils import sort_edge_index, degree, add_remaining_self_loops, remove_self_loops, get_laplacian, \
    to_undirected, to_dense_adj, to_networkx
import copy
from torch_geometric.transforms import GDC
# import numpy
# import numpy as np
# import torch
# import numba
# from torch_geometric.transforms import GDC
# from torch_geometric.utils import to_dense_adj, dense_to_sparse
# import scipy.sparse as sp
# import numpy as np
import torch
import os
import torch.nn
from torch_geometric.transforms import GDC
# from scipy.linalg._matfuncs import fractional_matrix_power
# from scipy.linalg._basic import inv
# from torch_geometric.nn import knn_graph


log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', None, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 3, 'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_enum('dataset', 'coauthor-cs',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs', 'cora', 'citeseer', 'pubmed'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 10000, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')

# Augmentations.
# flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1.')
# flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
# flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2.')
# flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')

# Logging and checkpoint.
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10, 'Log information at every log_steps.')

# Evaluation
flags.DEFINE_integer('eval_epochs', 5, 'Evaluate every eval_epochs.')
flags.DEFINE_string('drop_scheme', 'degree', 'method')
flags.DEFINE_float('drop_edge_rate_1', 0.2, 'sbl')
flags.DEFINE_float('drop_edge_rate_2', 0.4, 'sbl')
flags.DEFINE_float('drop_f_rate_1', 0.1, 'sbl')
flags.DEFINE_float('drop_f_rate_2', 0.1, 'sbl')
flags.DEFINE_float('lambd', 1e-3, 'lamd1')
flags.DEFINE_float('theta', 0.4, 'sbl')
flags.DEFINE_float('tau', 0.7, 'sbl')
flags.DEFINE_integer('k', 4, 'nei')
flags.DEFINE_integer('augk', 4, 'nei')
flags.DEFINE_float('alpha', 0.05, 'nei')
flags.DEFINE_integer('avg_degree', 4, 'nei')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))

    # set random seed
    if FLAGS.model_seed is not None:
        log.info('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)

    # create log directory
    os.makedirs(FLAGS.logdir, exist_ok=True)
    with open(os.path.join(FLAGS.logdir, 'config.cfg'), "w") as file:
        file.write(FLAGS.flags_into_string())  # save config file

    # load data
    if FLAGS.dataset != 'wiki-cs':
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
        num_eval_splits = FLAGS.num_eval_splits
    else:
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(FLAGS.dataset_dir)
        num_eval_splits = train_masks.shape[1]

    data = dataset[0]
    zm =dataset[0]# all dataset include one graph
    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    data = data.to(device)  # permanently move in gpy memory
    adj = data.edge_index# permanently move in gpy memory

    # edge_index3 = knngraph(data,data.edge_index, data.x, data.x, FLAGS.augk)
    # data = LocalDegreeProfile()
    data = GDC(diffusion_kwargs={'alpha': FLAGS.alpha, 'method': 'ppr'},
                        sparsification_kwargs={'method':'threshold', 'avg_degree': FLAGS.avg_degree})(data.cpu())
    data= data.to(device)
    log.info('Dataset {}, {}.'.format('GDC                          ', data.edge_index.size(1)))
    #data = GDC(diffusion_kwargs={'t': 3, 'method': 'heat'})(data.clone())


    knn_graph = knngraph(data,FLAGS.augk)
    log.info('Dataset {}, {}.'.format('KNN                           ', knn_graph.size(1)))
    data.edge_index = globlegraph(data.x, data.edge_index, adj,knn_graph)
    log.info('Dataset {}, {}.'.format('add                          ',data.edge_index.size(1)-zm.edge_index.size(1)))

    # data = LocalDegreeProfile()


    # prepare transforms
    # transform_1 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
    # transform_2 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)

    # build networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)   # 512, 256, 128
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = BGRL(encoder, predictor, FLAGS.tau).to(device)
    model = model.to(device)


    # 下面就可以正常使用了

    ...
    ...
    ...

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    # setup tensorboard and make custom layout
    # writer = SummaryWriter(FLAGS.logdir)
    layout = {'accuracy': {'accuracy/test': ['Multiline', [f'accuracy/test_{i}' for i in range(num_eval_splits)]]}}
    # writer.add_custom_scalars(layout)
    # gdc = GDC()
    # # # TypeError: diffusion_matrix_exa
    # data.edge_index = gdc.diffusion_matrix_approx(edge_index=data.edge_index, edge_weight=data.edge_weight,
    #                                             num_nodes = data.num_nodes, normalization='row', method='ppr',
    #                                           alpha=0.5, eps=0.01)
    # adj=to_dense_adj(data.edge_index)
    # adj=adj.cpu().numpy()
    # adj=numpy.squeeze(adj)
    #
    # def compute_ppr(adj, alpha=0.5, self_loop=True):
    #     a = adj
    #     if self_loop:
    #         a = a + np.eye(a.shape[0])  # A^ = A + I_n
    #     d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    #     dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    #     at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    #     return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1
    # adj = compute_ppr(adj=adj)
    #
    # adj = numpy.expand_dims(adj,axis=0)
    # adj = torch.from_numpy(adj).to(device)
    # data.edge_index = dense_to_sparse(adj)
    # edge_index1 = knn_graph(x=data.x,k=1)
    # adj1 = to_dense_adj(edge_index1)
    # # adj1 = adj1.cpu().numpy()
    # adj2 = to_dense_adj(data.edge_index)
    # # adj2 = adj2.cpu().numpy()
    # adj3 = adj1 + adj2dat

    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    global drop_weights
    if FLAGS.drop_scheme == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif FLAGS.drop_scheme == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif FLAGS.drop_scheme == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    if FLAGS.drop_scheme == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if FLAGS.dataset == 'wiki-cs':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif FLAGS.drop_scheme == 'pr':
        node_pr = compute_pr(data.edge_index)
        if FLAGS.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif FLAGS.drop_scheme == 'evc':
        node_evc = eigenvector_centrality(data)
        if FLAGS.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)



    def train(step):
        model.train()





        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()


        def drop_edge1():
            # global drop_weights

            if FLAGS.drop_scheme == 'uniform':
                return dropout_adj(data.edge_index, p=FLAGS.drop_edge_rate_1)[0]
            elif FLAGS.drop_scheme in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(data.edge_index, drop_weights,  p=FLAGS.drop_edge_rate_1,
                                          threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: {FLAGS.drop_scheme}')

        def drop_edge2():
            # global drop_weights

            if FLAGS.drop_scheme == 'uniform':
                return dropout_adj(data.edge_index, p=FLAGS.drop_edge_rate_2)[0]
            elif FLAGS.drop_scheme in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(data.edge_index, drop_weights,  p=FLAGS.drop_edge_rate_2,
                                          threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: {FLAGS.drop_scheme}')
        # edge_index_1 = drop_edge1()
        edge_index_1 = dropout_adj(data.edge_index, p=FLAGS.drop_edge_rate_1)[0]
        edge_index_2 = drop_edge2()
        # x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, FLAGS.drop_f_rate_2)
        if FLAGS.drop_scheme in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(data.x, feature_weights, FLAGS.drop_f_rate_1)
            # x_2 = drop_feature_weighted_2(data.x, feature_weights, 0.1)
        # x1, x2 = transform_1(data), transform_2(data)
        data1 = copy.deepcopy(data)
        data2 = copy.deepcopy(data)
        data1.x = x_1
        data2.x = x_2
        data1.edge_index = edge_index_1
        data2.edge_index = edge_index_2
        q1, y2, ind = model.forward(data1, data2, FLAGS.k)
        q2, y1, ind = model.forward(data2, data1, FLAGS.k)
        # --------------------------------------------
        # z1 = (q1 - q1.mean(0)) / q1.std(0)
        # z2 = (y2 - y2.mean(0)) / y2.std(0)
        # c = torch.mm(z1.T, z2.detach())
        # c1 = torch.mm(z1.T, z1)
        # c2 = torch.mm(z2.T, z2)
        # n = data.num_nodes
        # c = c / n
        # c1 = c1 / n
        # c2 = c2 / n
        # loss_inv = -torch.diagonal(c).sum()
        # iden = torch.tensor(numpy.eye(c.shape[0])).to(device)
        # loss_dec1 = (iden - c1).pow(2).sum()
        # loss_dec2 = (iden - c2).pow(2).sum()
        #
        # loss2 = loss_inv + FLAGS.lambd * (loss_dec1 + loss_dec2)
        # THETA 0.4 TAU 0.7
        # -----------------------------------------------------

        # -----------------------------------------------------------------------------
        # loss1 = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
        # loss2 = model.loss(q1, y2.detach()) + model.loss(q2, y1.detach())
        # loss = FLAGS.theta * loss1 + (1-FLAGS.theta) * loss2
        # --------------------------------------------------------------------------------

        # loss1 = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(),
        #                                                                                   dim=-1).mean()
        loss2 = model.loss(q1, y2.detach(),) + model.loss(q2, y1.detach())
        loss3 = model.loss_fn(q1[ind[0]], y2[ind[1]].detach()).mean()+model.loss_fn(q2[ind[0]], y1[ind[1]].detach()).mean()
        loss = FLAGS.theta * loss2 + (1 - FLAGS.theta) * loss3
        # loss = loss2

        loss.backward()


        # update online network
        optimizer.step()
        # update target network
        model.update_target_network(mm)

        # log scalars
        # writer.add_scalar('params/lr', lr, step)
        # writer.add_scalar('params/mm', mm, step)
        # writer.add_scalar('train/loss', loss1, step)

    def eval(epoch):
        # make temporary copy of encoder
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()
        representations, labels = compute_representations(tmp_encoder, dataset, device)

        if FLAGS.dataset != 'wiki-cs':
            scores = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),FLAGS.dataset,
                                             data_random_seed=FLAGS.data_seed, repeat=FLAGS.num_eval_splits)
        else:
            scores = fit_logistic_regression_preset_splits(representations.cpu().numpy(), labels.cpu().numpy(),
                                                           train_masks, val_masks, test_masks,FLAGS.dataset)

        # for i, score in enumerate(scores):
        #     # writer.add_scalar(f'accuracy/test_{i}', score, epoch)

    f = open("result_" + FLAGS.dataset + ".txt", "a")
    f.write(str("-----------------------------------------") + "\n")

    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        train(epoch-1)
        if epoch % FLAGS.eval_epochs == 0:
            eval(epoch)

    # save encoder weights
    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(FLAGS.logdir, 'online.pt'))


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
