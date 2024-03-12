import numpy as np
import paddle

from layers import *
import pgl


class EIGN(nn.Layer):
    def __init__(self, args):
        super(EIGN, self).__init__()
        input_dim = args.input_dim
        hidden_dim = args.hidden_dim
        self.hidden_dim = hidden_dim
        self.node_linear = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Dropout(0.1), nn.LeakyReLU(), nn.BatchNorm1D(hidden_dim))
        self.dist_embed = DistanceEmbedding(hidden_dim)
        self.get_edge = nn.LayerList()
        self.get_interact = nn.LayerList()
        self.egcl_layers = nn.LayerList()
        self.beta = self.create_parameter(shape=[1], default_initializer=paddle.nn.initializer.Assign(np.array([1.], dtype='float32')))
        self.num_egcl = args.num_egcl
        self.num_deal = args.num_deal
        for i in range(self.num_egcl):
            self.egcl_layers.append(EGCL(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, hidden_nf=self.hidden_dim, edges_in_d=self.hidden_dim))
        for i in range(self.num_deal):
            self.get_edge.append(GetEdge(atom_dim=hidden_dim))
            self.get_interact.append(DEAL(bond_dim=hidden_dim, atom_dim=hidden_dim, hidden_size=hidden_dim, num_heads=4, dropout=0.2, merge='mean', activation=F.relu))

        self.get_long_interact = LongRangeInteraction(edge_dim=hidden_dim)
        self.predict_layer = OutputLayer(hidden_dim)
    def forward(self, atom_g, bond2atom_g, global_feat, edge_type_l, inter_type_counts):
        atom_feat = atom_g.node_feat['feat']
        dist_feat = atom_g.edge_feat['dist']
        coords = atom_g.node_feat['coords']
        atom_feat = paddle.cast(atom_feat, 'float32')
        dist_feat = paddle.cast(dist_feat, 'float32')

        coords = paddle.cast(coords, 'float32')
        atom_h = self.node_linear(atom_feat)
        dist_h = self.dist_embed(dist_feat)
        for i in range(self.num_egcl):
            atom_h, coords = self.egcl_layers[i](atom_g, bond2atom_g, atom_h, coords, dist_h)
        for i in range(self.num_deal):
            atom_h_l = atom_h
            edge_h = self.get_edge[i](atom_g, atom_h, dist_h)
            interact_h = self.get_interact[i](bond2atom_g, atom_h, edge_h, dist_h)
            atom_h = atom_h_l + self.beta * interact_h

        inter_matrix = self.get_long_interact(edge_type_l, inter_type_counts, edge_h, dist_h)
        affinity = self.predict_layer(atom_g, atom_h)
        return affinity, inter_matrix