from src.dataloader.splice_graph_data_processor import *
from src.utils.data_utils import *


DATA_FILE_PATH = '/data/qzs23/projects/pathEm/aletsch-results/nnInput/aletsch-9'
NODE_FILE_PATH = os.path.join(DATA_FILE_PATH, 'polyester_refseq.full.node.gt.csv')
EDGE_FILE_PATH = os.path.join(DATA_FILE_PATH, 'polyester_refseq.full.edge.gt.csv')
GROUND_TRUTH_FILE_PATH = os.path.join(DATA_FILE_PATH, 'polyester_refseq.full.path.gt.csv')
PHASING_FILE_PATH = os.path.join(DATA_FILE_PATH, 'polyester_refseq.full.phasing.gt.csv')

samples_string = 'polyester_test1_refseq_1,polyester_test1_refseq_2,polyester_test1_refseq_3,polyester_test1_refseq_4,polyester_test1_refseq_5,polyester_test1_refseq_6,polyester_test1_refseq_7,polyester_test1_refseq_8,polyester_test1_refseq_9,polyester_test1_refseq_10,polyester_test1_refseq_11,polyester_test1_refseq_12,polyester_test1_refseq_13,polyester_test1_refseq_14,polyester_test1_refseq_15,polyester_test1_refseq_16,polyester_test1_refseq_17,polyester_test1_refseq_18,polyester_test1_refseq_19,polyester_test1_refseq_20,polyester_test3_refseq_1,polyester_test3_refseq_2,polyester_test3_refseq_3,polyester_test3_refseq_4,polyester_test3_refseq_5,polyester_test3_refseq_6,polyester_test3_refseq_7,polyester_test3_refseq_8,polyester_test3_refseq_9,polyester_test3_refseq_10,polyester_test3_refseq_11,polyester_test3_refseq_12,polyester_test3_refseq_13,polyester_test3_refseq_14,polyester_test3_refseq_15,polyester_test3_refseq_16,polyester_test3_refseq_17,polyester_test3_refseq_18,polyester_test3_refseq_19,polyester_test3_refseq_20,polyester_test5_refseq_1,polyester_test5_refseq_2,polyester_test5_refseq_3,polyester_test5_refseq_4,polyester_test5_refseq_5,polyester_test5_refseq_6,polyester_test5_refseq_7,polyester_test5_refseq_8,polyester_test5_refseq_9,polyester_test5_refseq_10,polyester_test5_refseq_11,polyester_test5_refseq_12,polyester_test5_refseq_13,polyester_test5_refseq_14,polyester_test5_refseq_15,polyester_test5_refseq_16,polyester_test5_refseq_17,polyester_test5_refseq_18,polyester_test5_refseq_19,polyester_test5_refseq_20,polyester_test7_refseq_1,polyester_test7_refseq_2,polyester_test7_refseq_3,polyester_test7_refseq_4,polyester_test7_refseq_5,polyester_test7_refseq_6,polyester_test7_refseq_7,polyester_test7_refseq_8,polyester_test7_refseq_9,polyester_test7_refseq_10,polyester_test7_refseq_11,polyester_test7_refseq_12,polyester_test7_refseq_13,polyester_test7_refseq_14,polyester_test7_refseq_15,polyester_test7_refseq_16,polyester_test7_refseq_17,polyester_test7_refseq_18,polyester_test7_refseq_19,polyester_test7_refseq_20,polyester_test9_refseq_1,polyester_test9_refseq_2,polyester_test9_refseq_3,polyester_test9_refseq_4,polyester_test9_refseq_5,polyester_test9_refseq_6,polyester_test9_refseq_7,polyester_test9_refseq_8,polyester_test9_refseq_9,polyester_test9_refseq_10,polyester_test9_refseq_11,polyester_test9_refseq_12,polyester_test9_refseq_13,polyester_test9_refseq_14,polyester_test9_refseq_15,polyester_test9_refseq_16,polyester_test9_refseq_17,polyester_test9_refseq_18,polyester_test9_refseq_19,polyester_test9_refseq_20'
samples = samples_string.split(',')


def save_datasets_to_file(dataset, saving_path, NODE_FILE_PATH, EDGE_FILE_PATH, GROUND_TRUTH_FILE_PATH, PHASING_FILE_PATH, samples):
    for sample in samples:
        print(f'Starting with sample: {sample}')

        data_processor = SpliceGraphDataProcessor(
            node_file_path=NODE_FILE_PATH,
            edge_file_path=EDGE_FILE_PATH,
            ground_truth_file_path=GROUND_TRUTH_FILE_PATH,
            phasing_file_path=PHASING_FILE_PATH,
            samples=[sample]
        )

        dataset = SpliceGraphDataset(data_processor)

        saving_path = '/data/aks7832/reinforce-assembly/data/tests'
        save_processed_dataset(dataset, os.path.join(saving_path, f'{sample}.pt'))