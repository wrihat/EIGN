import os
import pickle
from tqdm import tqdm
import numpy as np
import argparse
import openbabel
from openbabel import pybel
from featurizer import Featurizer
from scipy.spatial import distance_matrix

remove_complex = ['6b30', '6pgx', '5nxy', '1hyz', '1y3g', '6abk', '1hyv', '4u5t', '5nxx', '3qo9', '2v96', '4ehr']


def pocket_atom_num_from_pdb(name, path):
    n = 0
    with open('%s/%s/%s_pocket.pdb' % (path, name, name)) as f:
        for line in f:
            if 'REMARK' in line:
                break
        for line in f:
            cont = line.split()
            # break
            if cont[0] == 'CONECT':
                break
            n += int(cont[-1] != 'H' and cont[0] == 'ATOM')
    return n


def generate_atom_feature(path, name, featurizer, add_surface):
    ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (path, name, name)))
    ligand_coords, ligand_features = featurizer.get_features(path, name, ligand, molcode=1)
    pocket = next(pybel.readfile('pdb', '%s/%s/%s_pocket.pdb' % (path, name, name)))  
    protein = next(pybel.readfile('pdb', '%s/%s/%s_protein.pdb' % (path, name, name)))
    pocket_coords, pocket_features = featurizer.get_features(path, name, pocket, molcode=-1, surface=add_surface)
    node_num = pocket_atom_num_from_pdb(name, path)  
    pocket_coords = pocket_coords[:node_num]
    pocket_features = pocket_features[:node_num]

    try:
        assert (ligand_features[:, :9].sum(1) != 0).all()
    except:
        pass

    lig_atoms, pock_atoms = [], []
    for i, atom in enumerate(ligand):
        if atom.atomicnum > 1:
            lig_atoms.append(atom.atomicnum)
    for i, atom in enumerate(pocket):
        if atom.atomicnum > 1:
            pock_atoms.append(atom.atomicnum)

    pock_atoms = pock_atoms[:node_num]
    assert len(lig_atoms) == len(ligand_features) and len(pock_atoms) == len(pocket_features)

    # long-interaction atomic type pair
    ligand_atom_types = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # ligand atoms
    protein_atom_types = [6, 7, 8, 16]  # protein atoms
    keys = [(i, j) for i in protein_atom_types for j in ligand_atom_types]
    coords_ligand = np.vstack([atom.coords for atom in ligand])
    coords_protein = np.vstack([atom.coords for atom in protein])
    atom_map_ligand = [atom.atomicnum for atom in ligand]
    atom_map_protein = [atom.atomicnum for atom in protein]
    dm = distance_matrix(coords_ligand, coords_protein)
    ligand_index, protein_index = dist_filter(dm, 12)

    fea_dict = {k: 0 for k in keys}
    for x, y in zip(ligand_index, protein_index):
        x, y = atom_map_ligand[x], atom_map_protein[y]
        if x not in ligand_atom_types or y not in protein_atom_types: continue
        fea_dict[(y, x)] += 1

    return {'ligand_coords': ligand_coords, 'ligand_feats': ligand_features, 'ligand_atoms': lig_atoms,
            'pocket_coords': pocket_coords, 'pocket_feats': pocket_features, 'pocket_atoms': pock_atoms,
            'type_pair': list(fea_dict.values())}


def dist_filter(dist_matrix, theta):
    pos = np.where(dist_matrix <= theta)
    ligand_list, pocket_list = pos
    return ligand_list, pocket_list


def load_labels(data_path):
    labels = {}
    with open(data_path) as f:
        for line in f:
            if '#' in line:
                continue
            cont = line.strip().split()
            if len(cont) < 5:
                continue
            complex_id, pk = cont[0], cont[3]
            labels[complex_id] = float(pk)
    return labels


def inter_edges_ligand_pocket(dist_matrix, lig_size, theta=5.):
    pos = np.where(dist_matrix <= theta)
    ligand_list, pocket_list = pos
    node_list = sorted(list(set(pocket_list)))
    node_map = {node_list[i]: i + lig_size for i in range(len(node_list))}
    dist_list = dist_matrix[pos]
    edge_list = [(x, node_map[y]) for x, y in zip(ligand_list, pocket_list)]
    edge_list += [(y, x) for x, y in edge_list]
    dist_list = np.concatenate([dist_list, dist_list])

    return dist_list, edge_list, node_map


def get_complex_atom_feats(ligand_feats, pocket_feats, combine_mode=1):
    if combine_mode == 1:
        ligand_feats = np.hstack([ligand_feats, [[1]] * len(ligand_feats)])
        pocket_feats = np.hstack([pocket_feats, [[-1]] * len(pocket_feats)])
    elif combine_mode == 2:
        ligand_feats = np.hstack([ligand_feats, [[1, 0]] * len(ligand_feats)])
        pocket_feats = np.hstack([pocket_feats, [[0, 1]] * len(pocket_feats)])
    else:
        ligand_size = ligand_feats.shape[1]
        pocket_size = pocket_feats.shape[1]
        ligand_feats = np.hstack([ligand_feats, [[0] * pocket_size] * len(ligand_feats)])
        # temp = np.array([[0.] * ligand_size] * len(pocket_feats))
        # temp = temp.reshape((-1, ligand_size))
        temp = np.zeros((len(pocket_feats), ligand_size))
        pocket_feats = np.hstack([temp, pocket_feats])

    return ligand_feats, pocket_feats


def construct_ligand_pocket_graph_based_on_spatial_context(ligand, pocket, add_feat_mode=3, theta=5):
    ligand_feats, ligand_coords, ligand_atoms_raw = ligand
    pocket_feats, pocket_coords, pocket_atoms_raw = pocket
    # inter-relation between ligand and pocket
    ligand_size = ligand_feats.shape[0]
    complex_distance_matrix = distance_matrix(ligand_coords, pocket_coords)
    inter_dists, inter_edges, node_map = inter_edges_ligand_pocket(complex_distance_matrix, ligand_size, theta=theta)

    # map new pocket graph
    pocket_size = len(node_map)
    pocket_feats_inter = pocket_feats[sorted(node_map.keys())]  # 提取与小分子相连的原子的特征
    pocket_coords = np.array(pocket_coords)
    pocket_coords_inter = pocket_coords[sorted(node_map.keys())]  # 提取与小分子相连的原子的特征

    # construct ligand-pocket graph
    size = ligand_size + pocket_size
    ligand_feats, pocket_feats = get_complex_atom_feats(ligand_feats, pocket_feats_inter, combine_mode=add_feat_mode)
    if len(pocket_feats) > 0:
        if size != max(node_map.values()) + 1:
            return None

    complex_feats = np.vstack([ligand_feats, pocket_feats])
    complex_edges = {'inter_edges': inter_edges}
    lig_atoms_raw = np.array(ligand_atoms_raw)
    pocket_atoms_raw = np.array(pocket_atoms_raw)
    pocket_atoms_raw = pocket_atoms_raw[sorted(node_map.keys())]
    complex_atoms_raw = np.concatenate([lig_atoms_raw, pocket_atoms_raw]) if len(pocket_atoms_raw) > 0 else lig_atoms_raw
    complex_coords = np.vstack([ligand_coords, pocket_coords_inter]) if len(pocket_feats) > 0 else ligand_coords
    complex_coords = np.array(complex_coords)
    complex_feats = np.array(complex_feats)
    if complex_feats.shape[0] != complex_coords.shape[0]:
        return None
    return ligand_size, complex_coords, complex_feats, complex_atoms_raw, complex_edges


def train_valid_split(dataset_size, split_ratio=0.9, seed=0, shuffle=True):
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    split = 1000
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx


def process_dataset(test_path, train_path, dataset_name, output_path, cutoff, add_surface):
    test_set_list = [x for x in os.listdir(test_path) if len(x) == 4]
    train_set_list = [x for x in os.listdir(train_path) if len(x) == 4]
    path = train_path
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    with open(os.path.join(output_path, dataset_name + '_preprocessed.pkl'), 'rb') as f:
        processed_dict = pickle.load(f)
        # pickle.dump(processed_dict, f)

    for k, v in tqdm(processed_dict.items()):
        ligand = (v['ligand_feats'], v['ligand_coords'], v['ligand_atoms'])
        pocket = (v['pocket_feats'], v['pocket_coords'], v['pocket_atoms'])
        graph = construct_ligand_pocket_graph_based_on_spatial_context(ligand, pocket, add_feat_mode=3, theta=cutoff)
        if graph == None:
            continue
        cofeat = v.get('type_pair', None)
        pk = v.get('pk', None)
        if cofeat == None:
            continue
        if pk == None:
            continue
        graph = list(graph) + [cofeat]
        if k in test_set_list:
            test_data.append(graph)
            test_labels.append(pk)
            continue
        train_data.append(graph)
        train_labels.append(pk)

    # split train and valid
    train_indexs, valid_indexs = train_valid_split(len(train_data), split_ratio=0.9, seed=2020, shuffle=True)
    train_graph = [train_data[i] for i in train_indexs]
    train_label = [train_labels[i] for i in train_indexs]
    valid_graph = [train_data[i] for i in valid_indexs]
    valid_label = [train_labels[i] for i in valid_indexs]
    train = (train_graph, train_label)
    valid = (valid_graph, valid_label)
    test = (test_data, test_labels)

    with open(os.path.join(output_path, dataset_name + '_test.pkl'), 'wb') as f:
        pickle.dump(test, f)
    with open(os.path.join(output_path, dataset_name + '_train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(output_path, dataset_name + '_val.pkl'), 'wb') as f:
        pickle.dump(valid, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path_test', type=str, default="./dataset_raw/pdbbind2016/core-set")
    # parser.add_argument('--data_path_train', type=str, default="./dataset_raw/pdbbind2016/refined-set")
    parser.add_argument('--data_path_test', type=str, default="./dataset_raw/pdbbind2016/core-set")
    parser.add_argument('--data_path_train', type=str, default="D:/WorkSpace/postgraduate\datasets_resources/pdbbind2020/v2020-other-PL")
    # parser.add_argument('--data_path_core', type=str, default="/project/huangyang/jiajun/datasets/PDBbind_dataset/core-set")
    # parser.add_argument('--data_path_refined', type=str, default=r"/project/huangyang/jiajun/datasets/PDBbind_dataset/refined-set")
    parser.add_argument('--add_surface', type=bool, default=False)
    parser.add_argument('--output_path', type=str, default='./dataset/')
    parser.add_argument('--dataset_name', type=str, default='pdbbind2020_general')
    parser.add_argument('--cutoff', type=float, default=5.)
    args = parser.parse_args()
    process_dataset(args.data_path_test, args.data_path_train, args.dataset_name, args.output_path, args.cutoff, args.add_surface)








