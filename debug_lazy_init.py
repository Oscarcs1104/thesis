import torch
from train import parse_args
from model import build_model_from_args
from data import HybridGraphLangDataset, load_graph_dataset

args = parse_args()
args.data_path = 'data/esol.csv'
args.task = 'regression'
args.training_mode = 'decoder'
args.use_decoder = True
args.decoder_vocab_size = 35
args.decoder_pad_idx = 0
args.decoder_start_idx = 1
args.decoder_end_idx = 2
args.graph_backbone = 'gin'
args.language_backbone = 'chemberta'
args.fusion = 'concat'
args.hidden_dim = 256
args.num_layers = 3
args.num_heads = 8
args.dropout = 0.3
args.batch_size = 32
args.lr = 0.001
args.weight_decay = 1e-4
args.seed = 2025
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = build_model_from_args(args)
ckpt = torch.load('checkpoints\\test1\\esol\\best.pt', map_location='cpu', weights_only=False)
state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict'))
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print('missing', missing[:10])
print('unexpected', unexpected[:10])

graphs = load_graph_dataset(args.data_path)
dataset = HybridGraphLangDataset(graphs, None)
sample_graph = dataset[0]
if not hasattr(sample_graph, 'batch') or sample_graph.batch is None:
    sample_graph.batch = torch.zeros(sample_graph.num_nodes, dtype=torch.long)

model = model.to(args.device)
sample_graph = sample_graph.to(args.device)
with torch.no_grad():
    _ = model(sample_graph)

for name, param in model.named_parameters():
    if isinstance(param, torch.nn.parameter.UninitializedParameter):
        print('UNINIT', name)
        break
else:
    print('no uninitialized params in named_parameters')

for name, module in model.named_modules():
    if isinstance(module, torch.nn.modules.lazy.LazyModuleMixin):
        print('LAZY MODULE', name, type(module))
