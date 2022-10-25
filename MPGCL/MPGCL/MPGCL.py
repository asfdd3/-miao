import copy

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import  to_scipy_sparse_matrix
# from torch_geometric.transforms import GDC, LocalDegreeProfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_std

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


class BGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """

    def __init__(self, encoder, predictor,tau: float = 0.5,):
        super().__init__()
        # online network
        self.tau = tau
        self.online_encoder = encoder
        self.predictor = predictor

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) \
               + list(self.predictor.parameters())

    # return list(self.online_encoder.parameters()) + list(self.predictor.parameters())
    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, online_x, target_x,k):
        # forward online network
        online_y = self.online_encoder(online_x)

        # prediction
        online_q = self.predictor(online_y)

        # forward target networky
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        knn = posgraph(online_x.edge_index,online_y,target_y,k)

        return online_y, target_y, knn


        ret = ret.mean() if mean else ret.sum()

        return ret

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):

        l1 = self.semi_loss(z1, z2)
        # l2 = self.semi_loss(z2, z2)

        ret = l1
        ret = ret.mean() if mean else ret.sum()

        return ret

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)

        return 2 - 2 * (x * y).sum(dim=-1)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z2))

        return -torch.log(refl_sim.diag() / (refl_sim.sum(1) - refl_sim.diag()))

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


def load_trained_encoder(encoder, ckpt_path, device):
    r"""Utility for loading the trained encoder."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint['model'], strict=True)
    return encoder.to(device)


def compute_representations(net, dataset, device):
    r"""Pre-computes the representations for the entire dataset.

    Returns:
        [torch.Tensor, torch.Tensor]: Representations and labels.
    """
    net.eval()
    reps = []
    labels = []

    for data in dataset:
        # forward
        data = data.to(device)
        with torch.no_grad():
            reps.append(net(data))
            labels.append(data.y)

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    return [reps, labels]
def posgraph(adj, student, teacher, top_k):
    n_data, d = student.shape
    student = F.normalize(student, dim=-1, p=2)
    teacher = F.normalize(teacher, dim=-1, p=2)
    similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
    # similarity += torch.eye(n_data, device=device) * 10

    _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
    tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(device)
    # adj = torch.sparse.FloatTensor(adj, torch.ones_like(adj[0]), [student.shape[0], student.shape[0]])

    knn_neighbor = create_sparse(I_knn)
    # knn_neighbor = abs(knn_neighbor - adj)

    return knn_neighbor._indices()
    # return knn_neighbor


def knngraph(data, top_k):

    # x = LocalDegreeProfile()(data.clone())
    # similarity1 = torch.matmul(x, torch.transpose(x, 1, 0).detach())
    # _, I_knn1 = similarity1.topk(k=5, dim=1, largest=True, sorted=True)
    # s_neighbor = create_sparse(I_knn1).to(device)
    student = F.normalize(data.x, dim=-1, p=2)
    teacher = F.normalize(data.x, dim=-1, p=2)
    similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
    # similarity += torch.eye(n_data, device=device) * 10

    _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)



    knn_neighbor = create_sparse(I_knn.to(device))
    # adj = to_scipy_sparse_matrix(adj)
    # adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)






    return knn_neighbor._indices()


def create_sparse(I):
    similar = I.reshape(-1).tolist()
    index = np.repeat(range(I.shape[0]), I.shape[1])

    assert len(similar) == len(index)
    indices = torch.tensor(np.array([index, similar])).to(device)
    result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

    return result
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
class LocalDegreeProfile(BaseTransform):
    r"""Appends the Local Degree Profile (LDP) from the `"A Simple yet
    Effective Baseline for Non-attribute Graph Classification"
    <https://arxiv.org/abs/1811.03508>`_ paper

    .. math::
        \mathbf{x}_i = \mathbf{x}_i \, \Vert \, (\deg(i), \min(DN(i)),
        \max(DN(i)), \textrm{mean}(DN(i)), \textrm{std}(DN(i)))

    to the node features, where :math:`DN(i) = \{ \deg(j) \mid j \in
    \mathcal{N}(i) \}`.
    """
    def __call__(self, data):
        row, col = data.edge_index
        N = data.num_nodes

        deg = degree(row, N, dtype=torch.float)
        deg_col = deg[col]

        min_deg, _ = scatter_min(deg_col, row, dim_size=N)
        min_deg[min_deg > 10000] = 0
        max_deg, _ = scatter_max(deg_col, row, dim_size=N)
        max_deg[max_deg < -10000] = 0
        mean_deg = scatter_mean(deg_col, row, dim_size=N)
        std_deg = scatter_std(deg_col, row, dim_size=N)

        x = torch.stack([deg, min_deg, max_deg, mean_deg, std_deg], dim=1)
        x = F.normalize(x, dim=-1, p=2)
        # x = F.normalize(x, dim=-1, p=2)

        # if data.x is not None:
        #     data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
        #     data.x = torch.cat([data.x, x], dim=-1)
        # else:
        #     data.x = x

        return x
def globlegraph(x,diff,adj,knn_graph):
    adj = torch.sparse.FloatTensor(adj, torch.ones_like(adj[0]), [x.shape[0], x.shape[0]])
    diff = torch.sparse.FloatTensor(diff, torch.ones_like(diff[0]), [x.shape[0], x.shape[0]])
    kn_ngraph = torch.sparse.FloatTensor(knn_graph, torch.ones_like(knn_graph[0]), [x.shape[0], x.shape[0]])
    # newgraph = diff * kn_ngraph.


    globlegraph= adj.to(device) + (diff.to(device) * kn_ngraph)

    return globlegraph.coalesce()._indices()