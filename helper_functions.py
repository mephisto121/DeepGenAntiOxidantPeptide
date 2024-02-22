import argparse
import csv
import math
import os
import pickle
import time
from collections import defaultdict
from multiprocessing.resource_sharer import stop

import numpy as np
import pandas as pd
import tensorflow as tf
from Levenshtein import distance as lev_dist
from rdkit import Chem
from rdkit.Chem import Lipinski, rdchem, AllChem, rdmolops
from rdkit.Chem.AllChem import rdmolfiles
from rdkit.Chem.rdmolfiles import MolFromFASTA, MolFromSmiles, MolToSmiles
from tqdm import tqdm

from dataset import *
from vocab import *
from vocab import Vocab


class Filtration_Generate(Vocab):
    def __init__(self, vocab, seqs):
        super().__init__(vocab)
        self.seqs = seqs
        self.vocab = vocab

    def uniqueness(self, seqs):
        self.unique_seqs = defaultdict(int)
        for s in seqs:
            self.unique_seqs[s] += 1
        return self.unique_seqs, (len(self.unique_seqs)/len(seqs))*100

    def novelty(self, list_:list):
        self.novel_seq = []
        for s in self.unique_seqs:
            if s not in list_:
                self.novel_seq.append(s)
        return self.novel_seq, (len(self.novel_seq)/len(self.seqs))*100

    def scan_in_training(self ,train_data):
        if self.seqs not in train_data:
            return False
        else:
            return True

    def filter_generated_data(self, filter_list,):

        self.unique_seqs, self.perc_uniqueness = self.uniqueness(self.seqs)
        self.notintraining_seqs, self.perc_novelty = self.novelty(list_ = filter_list)
        self.final_seqs = [seq for seq in self.notintraining_seqs if len(seq)>1]
        print(self.perc_uniqueness, self.perc_novelty, len(self.notintraining_seqs))
        return self.final_seqs

    def save_df(self, train_data, folder):
        df_generated = pd.DataFrame(list(self.unique_seqs.keys()), columns =['Sequence']) 
        df_generated["Repetition"] = df_generated["Sequence"].map(lambda x: self.unique_seqs[x])
        df_generated["inTraining"] = df_generated["Sequence"].map(self.scan_in_training(train_data))
        df_generated["Set"] = "generated_tl_antiox"
        df_generated.to_pickle(folder+"pickles/Generated-TL-anticancer.pkl")

class Generate(Filtration_Generate, Vocab):
    """Generate peptide sequences and filter them based on their uniqueness and novelty"""
    def __init__(self, model, vocab):
        super(Generate).__init__()
        self.vocab_ = vocab
        Vocab.__init__(self,vocab=self.vocab_)
        self.model = model

    def sample_token(self, x, T = 1):
        self.T = T
        return tf.random.categorical(x / T, 1)

    def generate_single(self, length, temperature):
        self.T = temperature
        seq = []
        x = tf.ones((1, 1))
        self.model.reset_states()
        for _ in range(length):
            v = self.model.predict(x, verbose=0)
            x = self.sample_token(tf.reshape(v, [1,23]), self.T)
            if x.numpy() == 2 or x.numpy() ==0:
                stop
            else:
                seq.append(int(np.squeeze(x.numpy())))
        return seq
    
    def generate_multi_seqs(self, num_counts, seqs_length, temp):
        self.gen_seqs = []

        for _ in tqdm(range(num_counts)):
            gen_sngl = self.generate_single(seqs_length,temperature=temp)
            self.single_seq = []
            for n in gen_sngl:
                gen_str = Vocab.int2str(self, num = n)
                self.single_seq.append(gen_str)
            self.gen_seqs.append("".join(self.single_seq))
        return self.gen_seqs

    def filter_generated_data(self, list__, seq_list):
        if seq_list == None:

            self.unique_seqs, self.perc_uniqueness = Filtration_Generate.uniqueness(self.gen_seqs)
            self.notintraining_seqs, self.perc_novelty = Filtration_Generate.novelty(self.unique_seqs, list_ = list__)
            self.final_seqs = [seq for seq in self.notintraining_seqs if len(seq)>1]
            print(self.perc_uniqueness, self.perc_novelty, len(self.notintraining_seqs))
            return self.final_seqs
        else:

            self.unique_seqs, self.perc_uniqueness = Filtration_Generate.uniqueness(seq_list)
            self.notintraining_seqs, self.perc_novelty = Filtration_Generate.novelty(self.unique_seqs, list_ = list__)
            self.final_seqs = [seq for seq in self.notintraining_seqs if len(seq)>1]
            print(self.perc_uniqueness, self.perc_novelty, len(self.notintraining_seqs))
            return self.final_seqs
        
    
    def save_gen_data(self, folder, name):
        df = pd.DataFrame(list(self.unique_seqs.keys()), columns="Sequence")
        df["Repetition"] = df["Sequence"].map(lambda x: self.unique_seqs[x])
        df["inTraining"] = df["Sequence"].map(self.scan_in_training())
        df["Set"] = "generated"

        return df.to_pickle(folder + name+"Generated.pkl")

def filter_len(data_list, length):
    list_ = []
    for seq in data_list:
        if len(seq)<=length:
            list_.append(seq)
    return list_


def get_model_name(k, name):
    return 'model_'+name+str(k)+'.tf'

def generate_pep_seq(predicted_perc, th, seq_lst):
    ans = []
    prob_result = []
    for i, p in enumerate(predicted_perc):
        if p>=th:
            ans.append(seq_lst[i])
            prob_result.append(p)
    return ans, prob_result

""" The codes below are from: https://github.com/reymond-group/MLpeptide"""

def find_seqNN(seq, dataframe):
    best_dist = float("inf")
    dists = dataframe["Sequence"].map(lambda seq2 : lev_dist(seq,seq2))
    NNi = np.argmin(dists)
    best_dist = dists.iloc[NNi]
    NN = dataframe["Sequence"].iloc[NNi]
    label = dataframe["FRS"].iloc[NNi]
    return best_dist, NN, label

def calc_neg(seq):
    seq = seq.upper()
    neg = (seq.count('D') + seq.count('E'))
    return neg

def calc_pos(seq):
    seq = seq.upper()
    pos = (seq.count('K') + seq.count('R'))
    return pos

def calc_aa(seq, aa):
    seq = seq.upper()
    aa_f = seq.count(aa)/len(seq) 
    return aa_f

def calc_hac(smiles):
    mol = MolFromSmiles(smiles)
    hac = Lipinski.HeavyAtomCount(mol)
    return hac

def calc_hydr(seq):
    hydr = (seq.count('A') + seq.count('L') + seq.count('I') + seq.count('L') \
            + seq.count('V') + seq.count('M') + seq.count('F') + seq.count('C'))
    return hydr

def hydropatch(seq):
    seq = seq.upper()
    
    hydro = ["A", "L", "I", "V", "M", "F", "C"]
    patch = ""
    patches = []
    for aa in seq:
        if aa in hydro:
            patch+=aa
        else:
            if patch != "":
                patches.append(len(patch))
            patch=""
    if patch != "":
        patches.append(len(patch))    
    return np.array(patches)


def calc_hba(smiles):
    mol = MolFromSmiles(smiles)
    hba = Lipinski.NumHAcceptors(mol)
    return hba

def calc_hbd(smiles):
    mol = MolFromSmiles(smiles)
    hbd = Lipinski.NumHDonors(mol)
    return hbd

def mean(patches):
    if len(patches) == 0:
        return 0
    return round(patches.mean(),2)

d_aminoacids = ["a","c","d","e","f","g","h","i","l","m","n","p","k","q","r","s","t","v","w","y"]
def d_aa(seq):
    for aa in d_aminoacids:
        if aa in seq:
            return True
    return False


"""
Calculates a set of properties from a protein sequence:
    - hydrophobicity (according to a particular scale)
    - mean hydrophobic dipole moment assuming it is an alpha-helix.
    - total charge (at pH 7.4)
    - amino acid composition
    - discimination factor according to Rob Keller (IJMS, 2011)
Essentially the same as HeliQuest (reproduces the same values).
Author:
  Joao Rodrigues
  j.p.g.l.m.rodrigues@gmail.com
"""




#
# Definitions
#
scales = {'Fauchere-Pliska': {'A':  0.31, 'R': -1.01, 'N': -0.60,
                              'D': -0.77, 'C':  1.54, 'Q': -0.22,
                              'E': -0.64, 'G':  0.00, 'H':  0.13,
                              'I':  1.80, 'L':  1.70, 'K': -0.99,
                              'M':  1.23, 'F':  1.79, 'P':  0.72,
                              'S': -0.04, 'T':  0.26, 'W':  2.25,
                              'Y':  0.96, 'V':  1.22},

          'Eisenberg': {'A':  0.25, 'R': -1.80, 'N': -0.64,
                        'D': -0.72, 'C':  0.04, 'Q': -0.69,
                        'E': -0.62, 'G':  0.16, 'H': -0.40,
                        'I':  0.73, 'L':  0.53, 'K': -1.10,
                        'M':  0.26, 'F':  0.61, 'P': -0.07,
                        'S': -0.26, 'T': -0.18, 'W':  0.37,
                        'Y':  0.02, 'V':  0.54},
          }
_supported_scales = list(scales.keys())

aa_charge = {'E': -1, 'D': -1, 'K': 1, 'R': 1}

#
# Functions
#
def assign_hydrophobicity(sequence, scale='Fauchere-Pliska'):  # noqa: E302
    """Assigns a hydrophobicity value to each amino acid in the sequence"""

    hscale = scales.get(scale, None)
    if not hscale:
        raise KeyError('{} is not a supported scale. '.format(scale))

    hvalues = []
    for aa in sequence:
        sc_hydrophobicity = hscale.get(aa, None)
        if sc_hydrophobicity is None:
            raise KeyError('Amino acid not defined in scale: {}'.format(aa))
        hvalues.append(sc_hydrophobicity)

    return hvalues


def calculate_moment(array, angle=100):
    """Calculates the hydrophobic dipole moment from an array of hydrophobicity
    values. Formula defined by Eisenberg, 1982 (Nature). Returns the average
    moment (normalized by sequence length)
    uH = sqrt(sum(Hi cos(i*d))**2 + sum(Hi sin(i*d))**2),
    where i is the amino acid index and d (delta) is an angular value in
    degrees (100 for alpha-helix, 180 for beta-sheet).
    """

    sum_cos, sum_sin = 0.0, 0.0
    for i, hv in enumerate(array):
        rad_inc = ((i*angle)*math.pi)/180.0
        sum_cos += hv * math.cos(rad_inc)
        sum_sin += hv * math.sin(rad_inc)
    if len(array) != 0:
        return math.sqrt(sum_cos**2 + sum_sin**2) / len(array)
    else:
        print(array)
        return 0


def calculate_charge(sequence, charge_dict=aa_charge):
    """Calculates the charge of the peptide sequence at pH 7.4
    """
    sc_charges = [charge_dict.get(aa, 0) for aa in sequence]
    return sum(sc_charges)


def calculate_discrimination(mean_uH, total_charge):
    """Returns a discrimination factor according to Rob Keller (IJMS, 2011)
    A sequence with d>0.68 can be considered a potential lipid-binding region.
    """
    d = 0.944*mean_uH + 0.33*total_charge
    return d


def calculate_composition(sequence):
    """Returns a dictionary with percentages per classes"""

    # Residue character table
    polar_aa = set(('S', 'T', 'N', 'H', 'Q', 'G'))
    speci_aa = set(('P', 'C'))
    apolar_aa = set(('A', 'L', 'V', 'I', 'M'))
    charged_aa = set(('E', 'D', 'K', 'R'))
    aromatic_aa = set(('W', 'Y', 'F'))

    n_p, n_s, n_a, n_ar, n_c = 0, 0, 0, 0, 0
    for aa in sequence:
        if aa in polar_aa:
            n_p += 1
        elif aa in speci_aa:
            n_s += 1
        elif aa in apolar_aa:
            n_a += 1
        elif aa in charged_aa:
            n_c += 1
        elif aa in aromatic_aa:
            n_ar += 1

    return {'polar': n_p, 'special': n_s,
            'apolar': n_a, 'charged': n_c, 'aromatic': n_ar}


def analyze_sequence(name=None, sequence=None, window=18, verbose=False):
    """Runs all the above on a sequence. Pretty prints the results"""



    w = window

    outdata = []  # for csv writing

    # Processing...
    seq_len = len(sequence)
    print('[+] Analysing sequence {} ({} aa.)'.format(name, seq_len))
    print('[+] Using a window of {} aa.'.format(w))
    for seq_range in range(0, seq_len):

        seq_w = sequence[seq_range:seq_range+w]
        if seq_range and len(seq_w) < w:
            break

        # Numerical values
        z = calculate_charge(seq_w)
        seq_h = assign_hydrophobicity(seq_w)
        av_h = sum(seq_h)/len(seq_h)
        av_uH = calculate_moment(seq_h)
        d = calculate_discrimination(av_uH, z)

        # AA composition
        aa_comp = calculate_composition(seq_w)
        n_tot_pol = aa_comp['polar'] + aa_comp['charged']
        n_tot_apol = aa_comp['apolar'] + aa_comp['aromatic'] + aa_comp['special']  # noqa: E501
        n_charged = aa_comp['charged']  # noqa: E501
        n_aromatic = aa_comp['aromatic']  # noqa: E501

        _t = [name, sequence, seq_range+1, w, seq_w, z, av_h, av_uH, d,
              n_tot_pol, n_tot_apol, n_charged, n_aromatic]
        outdata.append(_t)

        if verbose:
            print('  Window {}: {}-{}-{}'.format(seq_range+1, seq_range,
                                                 seq_w, seq_range+w))
            print('    z={:<3d} <H>={:4.3f} <uH>={:4.3f} D={:4.3f}'.format(z, av_h,  # noqa: E501
                                                                           av_uH, d))  # noqa: E501
            print('    Amino acid composition')
            print('      Polar    : {:3d} / {:3.2f}%'.format(n_tot_pol, n_tot_pol*100/w))  # noqa: E501
            print('      Non-Polar: {:3d} / {:3.2f}%'.format(n_tot_apol, n_tot_apol*100/w))  # noqa: E501
            print('      Charged  : {:3d} / {:3.2f}%'.format(n_charged, n_charged*100/w))  # noqa: E501
            print('      Aromatic : {:3d} / {:3.2f}%'.format(n_aromatic, n_aromatic*100/w))  # noqa: E501
            print()

    return outdata


def read_fasta_file(afile):
    """Parses a file with FASTA formatted sequences"""

    if not os.path.isfile(afile):
        raise IOError('File not found/readable: {}'.format(afile))

    sequences = []
    seq_name, cur_seq = None, None
    with open(afile) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('>'):
                if cur_seq:
                    sequences.append((seq_name, ''.join(cur_seq)))
                seq_name = line[1:]
                cur_seq = []
            elif line:
                cur_seq.append(line)
    sequences.append((seq_name, ''.join(cur_seq)))  # last seq

    return sequences

def hydr_moment(seq):
    seq = seq.upper()
    hdr = assign_hydrophobicity(seq,"Eisenberg")
    return calculate_moment(hdr)

def seq_to_smiles(seq):
    mol = MolFromFASTA(seq, flavor=True, sanitize = True)
    smiles = MolToSmiles(mol, isomericSmiles=True)
    return smiles

def AnalyzeComposition(seq_list:list) -> np.ndarray:
    """
    Returns array with counts of each amino acid.
    """
    AA_list = ['A','V', 'I', 'L', 'M', 'F', 'Y', 'W', 'R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P']
    aa_counts = np.zeros(len(AA_list))

    for idx, aa in enumerate(AA_list):
        for seq in seq_list:
            aa_counts[idx] = aa_counts[idx] + seq.count(aa)
    return aa_counts


def AnalyzeCompositionFraction(seq_list:list) -> np.ndarray:
    """
    Returns array with mean fraction of each amino acid and standard deviation.
    """
    AA_list = ['A','V', 'I', 'L', 'M', 'F', 'Y', 'W', 'R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P']
    fracs_aa = np.zeros(len(seq_list))
    fracs_mean = np.zeros(len(AA_list))
    fracs_std = np.zeros(len(AA_list))

    for i, aa in enumerate(AA_list):
        for j, seq in enumerate(seq_list):
            fracs_aa[j] = seq.count(aa)/len(seq)
        fracs_mean[i] = np.mean(fracs_aa)
        fracs_std[i] = np.std(fracs_aa)
    return fracs_mean, fracs_std

def StatisticalAnalysis(values:list) -> str:
    """
    Returns mean and std in the form of mean +- std
    """
    mean = sum(values)/len(values)
    variance = sum([((x - mean) ** 2) for x in values])/len(values)
    std = variance**0.5

    return f'{mean} ± {std}'

def frac_pos_charges(seq:str) -> float: 
    """
    Determine the fraction of positive charges in a sequence. 
    """
    counter_pos = seq.count('K') + seq.count('R')
    return counter_pos/len(seq)

def createxyzfile(seq):
    """ 
    Returns the XYZ file for a peptide sequence
    """
    mol = rdmolops.AddHs(Chem.MolFromFASTA(seq))
    AllChem.EmbedMolecule(mol)
    return rdmolfiles.MolToXYZFile(mol, "data/xyz_files/"+str(seq)+".xyz")