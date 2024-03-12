
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.utils.helper import generate_segment_id_from_index
from pgl.utils import op
import pgl.math as math
from utils import generate_segment_id
import warnings
import random


class DenseLayer(nn.Layer):
    def __init__(self, in_dim, out_dim, activation=F.relu, bias=True):
        super(DenseLayer, self).__init__()
        self.activation = activation
        if not bias:
            self.fc = nn.Linear(in_dim, out_dim, bias_attr=False)
        else:
            self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, input_feat):
        return self.activation(self.fc(input_feat))

class EGCL(nn.Layer):

    def __init__(self, input_dim, hidden_dim, hidden_nf, edges_in_d=0, act_fn=nn.Swish(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(EGCL, self).__init__()
        input_edge = input_dim * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        # edge_coords_nf = hidden_dim
        edge_dist_d = hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edge_dist_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_dim))
        # paddle.nn.initializer.XavierNormal(layer.weight, gain=0.001)
        layer = nn.Linear(hidden_nf, 1, bias_attr=False)
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
    def forward(self, atom_g, bond2atom_g, atom_h, coord, edge_feat=None, node_feat=None, dist_h=None):
        edge_index = atom_g.edges
        row, col = edge_index[:, 0], edge_index[:, 1]
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_h = self.edge_model(atom_h[row], atom_h[col], radial, edge_attr=edge_feat)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_h)
        atom_h = self.node_model(atom_h, edge_index, edge_h)
        return atom_h, coord

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = paddle.concat([source, target, radial], axis=1)
        else:
            out = paddle.concat([source, target, radial, edge_attr], axis=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    # def node_model(self, bond2atom_g, atom_h, edge_h, dist_h):
    #     atom_h_l = atom_h
    #     interact_h = self.get_interact(bond2atom_g, atom_h, edge_h, dist_h)
    #     atom_h = atom_h_l + self.beta * interact_h
    #     return atom_h

    def node_model(self, atom_h, edge_index, edge_attr, node_attr=None):
        row, col = edge_index[:, 0], edge_index[:, 1]
        agg_h = math.segment_pool(edge_attr, row, pool_type='sum')
        if node_attr is not None:
            agg_h = paddle.concat([atom_h, agg_h, node_attr], axis=1)
        else:
            agg_h = paddle.concat([atom_h, agg_h], axis=1)
        out = self.node_mlp(agg_h)
        if self.residual:
            out = atom_h + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index[:, 0], edge_index[:, 1]
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg_h = math.segment_pool(trans, row, pool_type='sum')
        elif self.coords_agg == 'mean':
            agg_h = math.segment_pool(trans, row, pool_type='mean')
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg_h
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index[:, 0], edge_index[:, 1]
        coord_diff = coord[row] - coord[col]
        radial = paddle.sum(coord_diff**2, 1).unsqueeze(1)
        if self.normalize:
            norm = paddle.sqrt(radial) + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff


class DistanceEmbedding(nn.Layer):
    def __init__(self, hidden_dim):
        super(DistanceEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.dist_input_layer = nn.Sequential(
            DenseLayer(32, 64),
            nn.Dropout(0.1),
            DenseLayer(64, hidden_dim),
            )

    def forward(self, dist_feat):
        dist_h = RBF(paddle.norm(dist_feat, axis=-1), D_min=0., D_max=6., D_count=32)
        dist_h = self.dist_input_layer(dist_h)
        return dist_h

class GetEdge(nn.Layer):

    def __init__(self, atom_dim, hidden_size=128, activation=F.relu):
        super(GetEdge, self).__init__()
        in_dim = atom_dim * 2 + hidden_size
        self.fc = DenseLayer(in_dim, hidden_size, activation=activation, bias=True)
    def agg_func(self, src_feat, dst_feat, edge_feat):
        src_h = src_feat['h']
        dst_h = dst_feat['h']
        agg_h = paddle.concat([src_h, dst_h, edge_feat['h']], axis=-1)
        return {'h': agg_h}

    def forward(self, g, atom_feat, edge_h):
        message = g.send(self.agg_func, src_feat={'h': atom_feat}, dst_feat={'h': atom_feat}, edge_feat={'h': edge_h})
        bond_feat = message['h']
        bond_feat = self.fc(bond_feat)
        return bond_feat


class DEAL(nn.Layer):

    def __init__(self, bond_dim, atom_dim, hidden_size, num_heads, dropout, merge='mean', activation=F.relu):
        super(DEAL, self).__init__()
        self.merge = merge
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.bond_fc = nn.Linear(bond_dim, num_heads * hidden_size)
        self.atom_fc = nn.Linear(atom_dim, num_heads * hidden_size)
        self.dist_fc = nn.Linear(hidden_size, num_heads * hidden_size)
        self.weight_bond = self.create_parameter(shape=[num_heads, hidden_size])
        self.weight_atom = self.create_parameter(shape=[num_heads, hidden_size])
        self.weight_distande = self.create_parameter(shape=[num_heads, hidden_size])

        self.feat_drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation

    def forward(self, bond2atom_g, atom_h, edge_h, dist_h):
        edge_feat = self.feat_drop(edge_h)
        atom_feat = self.feat_drop(atom_h)
        dist_h = self.feat_drop(dist_h)

        edge_feat = self.bond_fc(edge_feat)
        atom_feat = self.atom_fc(atom_feat)
        dist_h = self.dist_fc(dist_h)

        edge_feat = paddle.reshape(edge_feat, [-1, self.num_heads, self.hidden_size])
        atom_feat = paddle.reshape(atom_feat, [-1, self.num_heads, self.hidden_size])
        dist_h = paddle.reshape(dist_h, [-1, self.num_heads, self.hidden_size])

        interact_h = edge_feat * dist_h
        attn_edge = paddle.sum(edge_feat * self.weight_bond, axis=-1)
        attn_atom = paddle.sum(atom_feat * self.weight_atom, axis=-1)
        attn_dist = paddle.sum(dist_h * self.weight_distande, axis=-1)

        message = bond2atom_g.send(self.attn_send_func,
                     src_feat={"attn": attn_edge, "h": interact_h},  #
                     dst_feat={"attn": attn_atom},
                     edge_feat={"attn": attn_dist})
        agg_h = bond2atom_g.recv(reduce_func=self.attn_recv_func, msg=message)
        if self.activation:
            agg_h = self.activation(agg_h)
        return agg_h

    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        alpha = src_feat["attn"] + dst_feat['attn'] + edge_feat['attn']
        alpha = self.leaky_relu(alpha)
        return {"alpha": alpha, "h": src_feat["h"]}

    def attn_recv_func(self, message):
        alpha = message.reduce_softmax(message["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        alpha = self.attn_drop(alpha)    #
        feature = message["h"]
        feature = paddle.reshape(feature, [-1, self.num_heads, self.hidden_size])
        feature = feature * alpha
        if self.merge == 'cat':
            feature = paddle.reshape(feature, [-1, self.num_heads * self.hidden_size])
        if self.merge == 'mean':
            feature = paddle.mean(feature, axis=1)
        if self.merge == 'sum':
            feature = paddle.sum(feature, axis=1)
        feature = message.reduce(feature, pool_type="sum")
        return feature


class Bond2AtomLayer(nn.Layer):
    """Implementation of Distance-aware Edge->Node Aggregation Layer.
    """

    def __init__(self, bond_dim, atom_dim, hidden_dim, num_heads, dropout, merge='mean', activation=F.relu):
        super(Bond2AtomLayer, self).__init__()
        self.merge = merge
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.src_fc = nn.Linear(bond_dim, num_heads * hidden_dim)
        self.dst_fc = nn.Linear(atom_dim, num_heads * hidden_dim)
        self.edg_fc = nn.Linear(hidden_dim, num_heads * hidden_dim)
        self.weight_src = self.create_parameter(shape=[num_heads, hidden_dim])
        self.weight_dst = self.create_parameter(shape=[num_heads, hidden_dim])
        self.weight_edg = self.create_parameter(shape=[num_heads, hidden_dim])

        self.feat_drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation

    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        alpha = src_feat["attn"] + dst_feat["attn"] + edge_feat['attn']
        alpha = self.leaky_relu(alpha)
        return {"alpha": alpha, "h": src_feat["h"]}

    def attn_recv_func(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        alpha = self.attn_drop(alpha)

        feature = msg["h"]
        feature = paddle.reshape(feature, [-1, self.num_heads, self.hidden_dim])
        feature = feature * alpha
        if self.merge == 'cat':
            feature = paddle.reshape(feature, [-1, self.num_heads * self.hidden_dim])
        if self.merge == 'mean':
            feature = paddle.mean(feature, axis=1)

        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, g, atom_feat, bond_feat, edge_feat):    # edge_feat就是边距离的embedding
        bond_feat = self.feat_drop(bond_feat)
        atom_feat = self.feat_drop(atom_feat)
        edge_feat = self.feat_drop(edge_feat)

        bond_feat = self.src_fc(bond_feat)
        atom_feat = self.dst_fc(atom_feat)
        edge_feat = self.edg_fc(edge_feat)

        bond_feat = paddle.reshape(bond_feat, [-1, self.num_heads, self.hidden_dim])
        atom_feat = paddle.reshape(atom_feat, [-1, self.num_heads, self.hidden_dim])
        edge_feat = paddle.reshape(edge_feat, [-1, self.num_heads, self.hidden_dim])

        attn_src = paddle.sum(bond_feat * self.weight_src, axis=-1)
        attn_dst = paddle.sum(atom_feat * self.weight_dst, axis=-1)
        attn_edg = paddle.sum(edge_feat * self.weight_edg, axis=-1)

        msg = g.send(self.attn_send_func,
                     src_feat={"attn": attn_src, "h": bond_feat},
                     dst_feat={"attn": attn_dst},
                     edge_feat={'attn': attn_edg})
        rst = g.recv(reduce_func=self.attn_recv_func, msg=msg)

        if self.activation:
            rst = self.activation(rst)
        return rst


class DomainAttentionLayer(nn.Layer):
    """Implementation of Angle Domain-speicific Attention Layer.
    """
    def __init__(self, bond_dim, hidden_dim, dropout, activation=None):
        super(DomainAttentionLayer, self).__init__()
        self.attn_fc = nn.Linear(2 * bond_dim, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, 1, bias_attr=False)

        self.feat_drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.activation = activation

    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        h_c = paddle.concat([src_feat['h'], src_feat['h']], axis=-1)
        h_c = self.attn_fc(h_c)
        h_c = self.tanh(h_c)
        h_s = self.attn_out(h_c)
        return {"alpha": h_s, "h": src_feat["h"]}

    def attn_recv_func(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = self.attn_drop(alpha)  # [-1, 1]
        feature = msg["h"]  # [-1, hidden_dim]
        feature = feature * alpha
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, g, bond_feat):
        bond_feat = self.feat_drop(bond_feat)
        msg = g.send(self.attn_send_func,
                     src_feat={"h": bond_feat},
                     dst_feat={"h": bond_feat})
        rst = g.recv(reduce_func=self.attn_recv_func, msg=msg)
        if self.activation:
            rst = self.activation(rst)
        return rst


class AngleI(nn.Layer):

    def __init__(self, bond_dim, hidden_dim, num_angle, dropout, merge='cat', activation=None):
        super(AngleI, self).__init__()
        self.num_angle = num_angle
        self.hidden_dim = hidden_dim
        self.merge = merge
        # self.out_fc = DenseLayer(num_angle * hidden_dim, hidden_dim)
        self.conv_layer = nn.LayerList()
        for _ in range(num_angle):
            conv = DomainAttentionLayer(bond_dim, hidden_dim, dropout, activation=None)
            self.conv_layer.append(conv)
        self.activation = activation

    def forward(self, g_list, bond_feat):
        h_list = []
        for k in range(self.num_angle):
            h = self.conv_layer[k](g_list[k], bond_feat)
            h_list.append(h)

        if self.merge == 'cat':
            feat_h = paddle.concat(h_list, axis=-1)
        if self.merge == 'mean':
            feat_h = paddle.mean(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'sum':
            feat_h = paddle.sum(paddle.stack(h_list, axis=1), axis=1)
        if self.merge == 'max':
            feat_h = paddle.max(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'cat_max':
            feat_h = paddle.stack(h_list, axis=-1)
            feat_max = paddle.max(feat_h, dim=1)[0]
            feat_max = paddle.reshape(feat_max, [-1, 1, self.hidden_dim])
            feat_h = paddle.reshape(feat_h * feat_max, [-1, self.num_angle * self.hidden_dim])

        # feat_h = self.out_fc(feat_h)
        if self.activation:
            feat_h = self.activation(feat_h)
        return feat_h


class LongRangeInteraction(nn.Layer):
    """Implementation of Pairwise Interactive Pooling Layer.
    """

    def __init__(self, edge_dim):
        super(LongRangeInteraction, self).__init__()
        self.edge_dim = edge_dim
        self.num_type = 4 * 9
        self.fc = nn.Linear(edge_dim, 1, bias_attr=False)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, bond_types_batch, type_count_batch, bond_feat, dist_h):
        """
        Input example:
            bond_types_batch: [0,0,2,0,1,2] + [0,0,2,0,1,2] + [2]
            type_count_batch: [[3, 3, 0], [1, 1, 0], [2, 2, 1]] # [num_type, batch_size]
        """
        inter_mat_list = []
        bond_feat = bond_feat * dist_h
        for type_i in range(self.num_type):
            type_i_index = paddle.masked_select(paddle.arange(len(bond_feat)), bond_types_batch == type_i)
            if paddle.sum(type_count_batch[type_i]) == 0:
                inter_mat_list.append(paddle.to_tensor(np.array([0.] * len(type_count_batch[type_i])), dtype='float32'))
                continue
            bond_feat_type_i = paddle.gather(bond_feat, type_i_index)
            graph_bond_index = op.get_index_from_counts(type_count_batch[type_i])
            # graph_bond_id = generate_segment_id_from_index(graph_bond_index)
            graph_bond_id = generate_segment_id(graph_bond_index)
            graph_feat_type_i = math.segment_pool(bond_feat_type_i, graph_bond_id, pool_type='sum')
            mat_flat_type_i = self.fc(graph_feat_type_i).squeeze(1)

            # print(graph_bond_id)
            # print(graph_bond_id.shape, graph_feat_type_i.shape, mat_flat_type_i.shape)
            my_pad = nn.Pad1D(padding=[0, len(type_count_batch[type_i]) - len(mat_flat_type_i)], value=-1e9)
            mat_flat_type_i = my_pad(mat_flat_type_i)
            inter_mat_list.append(mat_flat_type_i)

        inter_mat_batch = paddle.stack(inter_mat_list, axis=1)  # [batch_size, num_type]
        inter_mat_mask = paddle.ones_like(inter_mat_batch) * -1e9
        inter_mat_batch = paddle.where(type_count_batch.transpose([1, 0]) > 0, inter_mat_batch, inter_mat_mask)
        inter_mat_batch = self.softmax(inter_mat_batch)
        return inter_mat_batch


class OutputLayer(nn.Layer):

    def __init__(self, atom_dim):
        super(OutputLayer, self).__init__()
        self.atom_pool = pgl.nn.GraphPool(pool_type='sum')
        self.mlp = nn.LayerList()
        self.drop = nn.Dropout(0.2)   # 在全连接后面都添加drop层
        hidden_dim_list = [128*4, 128*2, 128]
        for hidden_dim in hidden_dim_list:
            self.mlp.append(nn.Sequential(nn.Linear(atom_dim, hidden_dim), nn.ReLU()))
            atom_dim = hidden_dim
        self.output_layer = nn.Linear(atom_dim, 1)

    def forward(self, g, atom_feat):
        graph_feat = self.atom_pool(g, atom_feat)
        for layer in self.mlp:
            graph_feat = layer(graph_feat)

        graph_feat = self.drop(graph_feat)
        output = self.output_layer(graph_feat)
        return output

def RBF(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''

    D_mu = paddle.linspace(D_min, D_max, D_count)
    D_mu = paddle.reshape(D_mu, [1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = paddle.unsqueeze(D, -1)
    out = paddle.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return out









