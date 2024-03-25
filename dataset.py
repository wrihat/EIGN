"""
Dataset code for protein-ligand complexe interaction graph construction.
"""

import os
import numpy as np
import paddle
import pgl
import pickle
from pgl.utils.data import Dataset
from pgl.utils.data import Dataloader
from scipy.spatial import distance
from scipy.sparse import coo_matrix
from utils import cos_formula
from tqdm import tqdm

prot_atom_ids = [6, 7, 8, 16]
drug_atom_ids = [6, 7, 8, 9, 15, 16, 17, 35, 53]
pair_ids = [(i, j) for i in prot_atom_ids for j in drug_atom_ids]


class InteractGraphDataset(Dataset):
    def __init__(self, data_path, dataset_name, dataset_seed=None, save_file=True):
        self.data_path = data_path
        self.dataset = dataset_name
        self.cut_dist = 5
        self.save_file = save_file
        self.dataset_length = 5000
        self.dataset_step = 3
        self.step_len = 3000   # 数据集分段的长度s
        self.dataset_seed = dataset_seed

        self.a2a_graphs = []
        self.b2a_graphs = []
        self.labels = []
        self.inter_feats_list = []
        self.bond_types_list = []
        self.type_count_list = []

        self.load_data()

    def __len__(self):
        """ Return the number of graphs. """
        return len(self.labels)

    def __getitem__(self, idx):
        """ Return graphs and label. """
        return self.a2a_graphs[idx], self.b2a_graphs[idx], \
               self.inter_feats_list[idx], self.bond_types_list[idx], self.type_count_list[idx], self.labels[idx]

    def has_cache(self):
        """ Check cache file."""
        if self.dataset[-5:] == 'train':
            file_name = os.path.join(self.data_path, "{0}.pkl".format(self.dataset))
            with open(file_name, 'rb') as f:
                data_mols, data_Y = pickle.load(f)
            self.dataset_length = len(data_Y)   # 数据集总长度
            self.step_len = int(self.dataset_length/self.dataset_step)  # 计算数据集分段的长度
            for i in range(self.dataset_step):
                graph_path = f'{self.data_path}/{self.dataset}_{i + 1}_graph.pkl'
                if os.path.exists(graph_path) == False:
                    return False
            return True
        else:
            self.graph_path = f'{self.data_path}/{self.dataset}_graph.pkl'
            return os.path.exists(self.graph_path)

    def save(self):
        """ Save the generated graphs. """
        print('--------------------Saving processed complex data...----------------------------')
        if self.dataset[-5:] == 'train':
            start = 0  # 数据集切片索引起始值
            for i in range(self.dataset_step):
                print(f'Saving processed {i + 1} part complex data...')
                tem = int(start+self.step_len)
                if tem < self.dataset_length:
                    end = int(start+self.step_len)
                else:
                    end = self.dataset_length
                graphs = [self.a2a_graphs[start:end], self.b2a_graphs[start:end]]
                global_feat = [self.inter_feats_list[start:end], self.bond_types_list[start:end], self.type_count_list[start:end]]
                labels = self.labels[start:end]
                start = end
                graph_path = f'{self.data_path}/{self.dataset}_{i + 1}_egcl_graph.pkl'
                with open(graph_path, 'wb') as f:
                    pickle.dump((graphs, global_feat, labels), f)
            print('--------------------Saving processed complex data...  end ----------------------------')
        else:
            graphs = [self.a2a_graphs, self.b2a_graphs]
            global_feat = [self.inter_feats_list, self.bond_types_list, self.type_count_list]
            with open(self.graph_path, 'wb') as f:
                pickle.dump((graphs, global_feat, self.labels), f)

    def load(self):
        """ Load the generated graphs. """
        print('Loading processed complex data...')
        if self.dataset[-5:] == 'train':
            if self.dataset_seed != None:
                print(f'Loading processed {self.dataset_seed} complex data...')
                graph_path = f'{self.data_path}/{self.dataset}_{self.dataset_seed}_graph.pkl'
                with open(graph_path, 'rb') as f:
                    graphs, global_feat, labels = pickle.load(f)
                    for a2a_graph, b2a_graph, inter_feat_list, bond_type_list, type_count_list, label in \
                            zip(graphs[0], graphs[1], global_feat[0], global_feat[1], global_feat[2],
                                labels):
                        self.a2a_graphs.append(a2a_graph)
                        self.b2a_graphs.append(b2a_graph)
                        self.inter_feats_list.append(inter_feat_list)
                        self.bond_types_list.append(bond_type_list)
                        self.type_count_list.append(type_count_list)
                        self.labels.append(label)
            else:
                for i in range(self.dataset_step):
                    print(f'Loading processed {i + 1} complex data...')
                    graph_path = f'{self.data_path}/{self.dataset}_{i + 1}_egcl_graph.pkl'
                    with open(graph_path, 'rb') as f:
                        graphs, global_feat, labels = pickle.load(f)
                        for a2a_graph, b2a_graph, inter_feat_list, bond_type_list, type_count_list, label in  \
                            zip(graphs[0], graphs[1], global_feat[0], global_feat[1], global_feat[2], labels):
                            self.a2a_graphs.append(a2a_graph)
                            self.b2a_graphs.append(b2a_graph)
                            self.inter_feats_list.append(inter_feat_list)
                            self.bond_types_list.append(bond_type_list)
                            self.type_count_list.append(type_count_list)
                            self.labels.append(label)

        else:
            with open(self.graph_path, 'rb') as f:
                graphs, global_feat, labels = pickle.load(f)
                self.a2a_graphs, self.b2a_graphs = graphs
                self.inter_feats_list, self.bond_types_list, self.type_count_list = global_feat
                self.labels = labels

    def build_graph(self, mol):
        # num_atoms_d, coords, features, atoms, long_inter_feats = mol
        num_atoms_d, coords, features, atoms, edges, long_inter_feats = mol  #in  general set
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        long_inter_feats = np.array([long_inter_feats])
        long_inter_feats = long_inter_feats / long_inter_feats.sum()

        ############################
        # build atom to atom graph #
        ############################
        num_atoms = len(coords)
        dist_graph_base = dist_mat.copy()
        dist_feat = dist_graph_base[dist_graph_base < self.cut_dist].reshape(-1, 1)

        dist_graph_base[dist_graph_base >= self.cut_dist] = 0.
        atom_graph = coo_matrix(dist_graph_base)
        a2a_edges = list(zip(atom_graph.row, atom_graph.col))
        edge_type = []
        for (i, j) in a2a_edges:
            if i < num_atoms_d and j >= num_atoms_d:
                edge_type.append(1.)
            elif i>=num_atoms_d and j<num_atoms_d:
                edge_type.append(1.)
            else:
                edge_type.append(0.)
        edge_type = np.array(edge_type).reshape(-1, 1)
        coords = np.array(coords).reshape(-1, 3)
        a2a_graph = pgl.Graph(a2a_edges, num_nodes=num_atoms, node_feat={"feat": features, 'coords': coords},
                              edge_feat={"dist": dist_feat, 'edge_type': edge_type})

        ######################
        # prepare bond nodes #
        ######################
        indices = []
        bond_pair_atom_types = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                a = dist_mat[i, j]
                if a < self.cut_dist:
                    at_i, at_j = atoms[i], atoms[j]
                    if i < num_atoms_d and j >= num_atoms_d and (at_j, at_i) in pair_ids:
                        bond_pair_atom_types += [pair_ids.index((at_j, at_i))]
                    elif i >= num_atoms_d and j < num_atoms_d and (at_i, at_j) in pair_ids:
                        bond_pair_atom_types += [pair_ids.index((at_i, at_j))]
                    else:
                        bond_pair_atom_types += [-1]
                    indices.append([i, j])

        ############################
        # build bond to atom graph #
        ############################
        num_bonds = len(indices)   
        assignment_b2a = np.zeros((num_bonds, num_atoms), dtype=np.int64)  # Maybe need too much memory
        assignment_a2b = np.zeros((num_atoms, num_bonds), dtype=np.int64)  # Maybe need too much memory
        for i, idx in enumerate(indices):
            assignment_b2a[i, idx[1]] = 1
            assignment_a2b[idx[0], i] = 1

        b2a_graph = coo_matrix(assignment_b2a)
        b2a_edges = list(zip(b2a_graph.row, b2a_graph.col))
        b2a_graph = pgl.BiGraph(b2a_edges, src_num_nodes=num_bonds, dst_num_nodes=num_atoms)

        bond_types = bond_pair_atom_types
        type_count = [0 for _ in range(len(pair_ids))]
        for type_i in bond_types:
            if type_i != -1:
                type_count[type_i] += 1

        bond_types = np.array(bond_types)
        type_count = np.array(type_count)

        graphs = a2a_graph, b2a_graph
        global_feat = long_inter_feats, bond_types, type_count
        return graphs, global_feat

    def load_data(self):
        """ Generate complex interaction graphs. """
        if self.has_cache():
            self.load()
        else:
            print('Processing raw protein-ligand complex data...')
            file_name = os.path.join(self.data_path, "{0}.pkl".format(self.dataset))
            with open(file_name, 'rb') as f:
                data_mols, data_Y = pickle.load(f)
            for mol, y in tqdm(zip(data_mols, data_Y)):
                graphs, global_feat = self.build_graph(mol)
                if graphs is None:
                    continue
                self.a2a_graphs.append(graphs[0])
                self.b2a_graphs.append(graphs[1])

                self.inter_feats_list.append(global_feat[0])
                self.bond_types_list.append(global_feat[1])
                self.type_count_list.append(global_feat[2])
                self.labels.append(y)

            self.labels = np.array(self.labels).reshape(-1, 1)
            if self.save_file:
                self.save()


def collate_fn(batch):
    atom2atom_gs, bond2atom_gs, feats, types, counts, labels = map(list, zip(*batch))

    atom2atom_g = pgl.Graph.batch(atom2atom_gs).tensor()
    bond2atom_g = pgl.BiGraph.batch(bond2atom_gs).tensor()
    feats = paddle.concat([paddle.to_tensor(f, dtype='float32') for f in feats])
    types = paddle.concat([paddle.to_tensor(t) for t in types])
    counts = paddle.stack([paddle.to_tensor(c) for c in counts], axis=1)
    labels = paddle.to_tensor(np.array(labels), dtype='float32')

    return atom2atom_g, bond2atom_g, feats, types, counts, labels


if __name__ == "__main__":
    train_data = InteractGraphDataset("./dataset/", "pdbbind2016_general_train")
    test_data = InteractGraphDataset("./dataset/", "pdbbind2016_general_test")
    val_data = InteractGraphDataset("./dataset/", "pdbbind2016_general_val")
    train_dataloader =Dataloader(train_data, batch_size=10, shuffle=True, collate_fn=collate_fn)
    test_dataloader = Dataloader(test_data, batch_size=10, shuffle=True, collate_fn=collate_fn)
    val_dataloader = Dataloader(val_data, batch_size=10, shuffle=True, collate_fn=collate_fn)

