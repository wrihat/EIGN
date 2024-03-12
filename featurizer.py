# Adapted from https://gitlab.com/cheminfIBB/tfbio/-/blob/master/tfbio/data.py

import pickle
from openbabel import pybel
import os
import sys
import argparse
import numpy as np
from scipy.spatial import distance_matrix

def dist_filter(dist_matrix, theta):
    pos = np.where(dist_matrix <= theta)
    atom_list, surface_list = pos
    return atom_list, surface_list


class Featurizer():
    """
    Calcaulates atomic features for molecules.  
    Features can encode atom type, 
    native pybel properties or
    any property defined with SMARTS patterns
    Attributes:
    FEATURE_NAMES: list of strings  
        Labels for features (in the same order as features)
    NUM_ATOM_CLASSES: int 
        Number of atom codes
    ATOM_CODES: dict  
        Dictionary mapping atomic numbers to codes
    NAMED_PROPS: list of string  
        Names of atomic properties to retrieve from pybel.Atom object
    CALLABLES: list of callables
        Callables used to calculcate custom atomic properties  
    SMARTS: list of SMARTS strings  
        SMARTS patterns defining additional atomic properties
    """

    def __init__(self,
                 atom_codes=None,
                 atom_labels=None,
                 named_properties=None,
                 save_molecule_codes=True,
                 custom_properties=None,
                 smarts_properties=None,
                 smarts_labels=None):
        """Creates Featurizer with specified types of features. Elements of a
        feature vector will be in a following order: atom type encoding
        (defined by atom_codes), Pybel atomic properties (defined by
        named_properties), molecule code (if present), custom atomic properties
        (defined `custom_properties`), and additional properties defined with
        SMARTS (defined with `smarts_properties`).  
        Parameters
        ----------
        atom_codes: dict, optional
            Dictionary mapping atomic numbers to codes. It will be used for
            one-hot encoding therefore if n different types are used, codes
            should be from 0 to n-1. Multiple atoms can have the same code,
            e.g. you can use {6: 0, 7: 1, 8: 1} to encode carbons with [1, 0]
            and nitrogens and oxygens with [0, 1] vectors. If not provided,
            default encoding is used.  
        atom_labels: list of strings, optional
            Labels for atoms codes. It should have the same length as the
            number of used codes, e.g. for `atom_codes={6: 0, 7: 1, 8: 1}` you
            should provide something like ['C', 'O or N']. If not specified
            labels 'atom0', 'atom1' etc are used. If `atom_codes` is not
            specified this argument is ignored.  
        named_properties: list of strings, optional
            Names of atomic properties to retrieve from pybel.Atom object. If
            not specified ['hyb', 'heavyvalence', 'heterovalence',
            'partialcharge'] is used.  
        save_molecule_codes: bool, optional (default True)
            If set to True, there will be an additional feature to save
            molecule code. It is usefeul when saving molecular complex in a
            single array.
        custom_properties: list of callables, optional
            Custom functions to calculate atomic properties. Each element of
            this list should be a callable that takes pybel.Atom object and
            returns a float. If callable has `__name__` property it is used as
            feature label. Otherwise labels 'func<i>' etc are used, where i is
            the index in `custom_properties` list.
        smarts_properties: list of strings, optional
            Additional atomic properties defined with SMARTS patterns. These
            patterns should match a single atom. If not specified, deafult
            patterns are used.  
        smarts_labels: list of strings, optional
            Labels for properties defined with SMARTS. Should have the same
            length as `smarts_properties`. If not specified labels 'smarts0',
            'smarts1' etc are used. If `smarts_properties` is not specified
            this argument is ignored.
        """
        # Remember namse of all features in the correct order
        self.FEATURE_NAMES = []

        if atom_codes is not None:  
            if not isinstance(atom_codes, dict):
                raise TypeError('Atom codes should be dict, got %s instead'
                                % type(atom_codes))
            codes = set(atom_codes.values())  
            for i in range(len(codes)):
                if i not in codes:
                    raise ValueError('Incorrect atom code %s' % i)

            self.NUM_ATOM_CLASSES = len(codes)  
            self.ATOM_CODES = atom_codes  
            if atom_labels is not None:
                if len(atom_labels) != self.NUM_ATOM_CLASSES:
                    raise ValueError('Incorrect number of atom labels: '
                                     '%s instead of %s'
                                     % (len(atom_labels), self.NUM_ATOM_CLASSES))
            else:
                atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)]
            self.FEATURE_NAMES += atom_labels  
        else:
            self.ATOM_CODES = {}  
            self.ATOM_CODES_POCKET = {}
            metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))  

            # List of tuples (atomic_num, class_name) with atom types to encode.
            atom_classes_pocket = [
                (6, 'C'),
                (1, 'H'),
                (8, 'O'),
                (7, 'N'),
                (16, 'S'),
                (34, 'Se'),
            ] 
            atom_classes = [
                (5, 'B'),
                (6, 'C'),
                (7, 'N'),
                (8, 'O'),
                (15, 'P'),
                (16, 'S'),
                (34, 'Se'),
                ([9, 17, 35, 53], 'halogen'),
                (metals, 'metal')
            ]
            for code, (atom, name) in enumerate(atom_classes):
                if type(atom) is list:
                    for a in atom:
                        self.ATOM_CODES[a] = code  
                else:
                    self.ATOM_CODES[atom] = code
                self.FEATURE_NAMES.append(name) 

            self.NUM_ATOM_CLASSES = len(atom_classes)  

        if named_properties is not None:
            if not isinstance(named_properties, (list, tuple, np.ndarray)): 
                raise TypeError('named_properties must be a list')
            allowed_props = [prop for prop in dir(pybel.Atom)
                             if not prop.startswith('__')]  
            for prop_id, prop in enumerate(named_properties):
                if prop not in allowed_props:
                    raise ValueError(
                        'named_properties must be in pybel.Atom attributes,'
                        ' %s was given at position %s' % (prop_id, prop)
                    )
            self.NAMED_PROPS = named_properties
        else:
            # pybel.Atom properties to save
            self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree',
                                'partialcharge']  
        self.FEATURE_NAMES += self.NAMED_PROPS  

        if not isinstance(save_molecule_codes, bool):
            raise TypeError('save_molecule_codes should be bool, got %s '
                            'instead' % type(save_molecule_codes))
        self.save_molecule_codes = save_molecule_codes
        if save_molecule_codes:  
            # Remember if an atom belongs to the ligand or to the protein
            self.FEATURE_NAMES.append('molcode')

        self.CALLABLES = []
        if custom_properties is not None:  
            for i, func in enumerate(custom_properties):
                if not callable(func):
                    raise TypeError('custom_properties should be list of'
                                    ' callables, got %s instead' % type(func))
                name = getattr(func, '__name__', '')  
                if name == '':
                    name = 'func%s' % i
                self.CALLABLES.append(func)
                self.FEATURE_NAMES.append(name)

        if smarts_properties is None:
            # SMARTS definition for other properties  
            self.SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]
            smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                             'ring'] 
        elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
            raise TypeError('smarts_properties must be a list')
        else:
            self.SMARTS = smarts_properties

        if smarts_labels is not None:
            if len(smarts_labels) != len(self.SMARTS):
                raise ValueError('Incorrect number of SMARTS labels: %s'
                                 ' instead of %s'
                                 % (len(smarts_labels), len(self.SMARTS)))
        else:
            smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

        # Compile patterns
        self.compile_smarts()
        self.FEATURE_NAMES += smarts_labels 

    def compile_smarts(self):
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def encode_num(self, atomic_num):  

        if not isinstance(atomic_num, int):
            raise TypeError('Atomic number must be int, %s was given'
                            % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding 

    def find_smarts(self, molecule):

        if not isinstance(molecule, pybel.Molecule):  
            raise TypeError('molecule must be pybel.Molecule object, %s was given'
                            % type(molecule))

        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))

        for (pattern_id, pattern) in enumerate(self.__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                       dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    def get_features(self, path, name, molecule, molcode=None, surface=False):  
        if not isinstance(molecule, pybel.Molecule):  
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))
        if molcode is None:
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True,'
                                 ' you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given'
                            % type(molcode))

        coords = []  
        features = []  
        heavy_atoms = [] 
        if surface == True:
            surface_features = np.load(
                '%s/%s/%s_protein_pred_surface_features_emb.npy' % (path, name, name), allow_pickle=True)  
            point_cloud_coords = np.load('%s/%s/%s_protein_pred_point_coords.npy' % (path, name, name), allow_pickle=True)  # 读取点云坐标
            for i, atom in enumerate(molecule):  
                # ignore hydrogens and dummy atoms (they have atomicnum set to 0)  忽略H原子和
                if atom.atomicnum > 1:  
                    heavy_atoms.append(i)  
                    coords.append(atom.coords)  
                    if molcode == -1: 
                        dist_mat = distance_matrix(np.array([atom.coords]), point_cloud_coords)
                        atom_index, point_index = dist_filter(dist_mat, 3)   
                        if len(point_index) == 0:
                            surface = np.array([[0.]*16])
                        else:
                            surface = surface_features[point_index]
                        mean_surface = np.mean(surface, axis=0)  # 对每一维的表面特征求平均值
                        features.append(np.concatenate((
                                self.encode_num(atom.atomicnum),  # 原子类型编码  1维向量，9个分量
                                [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],  # 获取NAMED_PROPS中需要的pybel.Atom中的属性  4个分量
                                [func(atom) for func in self.CALLABLES],  # 其他处理的属性
                                mean_surface
                            )))
        else:
            for i, atom in enumerate(molecule): 
                # ignore hydrogens and dummy atoms (they have atomicnum set to 0)  
                if atom.atomicnum > 1:  
                    heavy_atoms.append(i)  
                    coords.append(atom.coords) 

                    if molcode == -1:  
                        features.append(np.concatenate((
                                self.encode_num(atom.atomicnum),  
                                [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],  
                                [func(atom) for func in self.CALLABLES], 
                                # mean_surface
                            )))

                    else:
                        features.append(np.concatenate((
                            self.encode_num(atom.atomicnum),  
                            [atom.__getattribute__(prop) for prop in self.NAMED_PROPS], 
                            [func(atom) for func in self.CALLABLES],  # 其他处理的属性
                            # np.array([0.]*16)    # 如果是小分子则将表面特征设为0
                        )))


        if self.save_molecule_codes:
            features = np.hstack((features,  molcode * np.ones((len(features), 1)))) 
        features = np.hstack([features, self.find_smarts(molecule)[heavy_atoms]])  
        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')

        return coords, features

    def to_pickle(self, fname='featurizer.pkl'):
        """Save featurizer in a given file. Featurizer can be restored with
        `from_pickle` method.
        Parameters
        ----------
        fname: str, optional
           Path to file in which featurizer will be saved
        """
        # patterns can't be pickled, we need to temporarily remove them
        patterns = self.__PATTERNS[:]
        del self.__PATTERNS
        try:
            with open(fname, 'wb') as f:  
                pickle.dump(self, f)  
        finally:
            self.__PATTERNS = patterns[:]

    @staticmethod
    def from_pickle(fname):
        """Load picked featurizer from a given file
        Parameters
        ----------
        fname: str, optional
           Path to file with saved featurize
        Returns
        -------
        featurizer: Featurizer object
           Loaded featurizer
        """
        with open(fname, 'rb') as f:
            featurizer = pickle.load(f)
        featurizer.compile_smarts()
        return featurizer












