#%%
from src.dataloader.splice_graph_data_processor import *
from src.environments.splice_graph_env import *
#%%
from src.utils.data_utils import *
# %%
input_dir = "/data/qzs23/projects/pathEm/aletsch-results/nnInput"
prefix = "polyester_refseq.full"
samples = ['polyester_test1_refseq_1']

processor = SpliceGraphDataProcessor(
        input_dir=input_dir,
        prefix=prefix,
        samples=samples,
        normalize_features=True,
        num_ground_truths=5
    )
# %%
graph_ids = processor.get_all_graph_ids()
print(f"Found {len(graph_ids)} valid graphs")
#%%
dataset = SpliceGraphDataset(processor)
print(f"Dataset size: {len(dataset)}")
# %%
env = SpliceGraphEnv(dataset[0])
# %%
initial_state = env.reset()
# %%
print(env.get_valid_actions())
# %%
new_graph, total_reward, done, info = env.step(9)
# %%
new_graph.x[env.current_node, -2]
# %%
print(env.current_node)
# %%
