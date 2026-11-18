"""Exercise dense CUDA math and the PyG CUDA operators MatterGen uses."""
import json
import torch
import torch_geometric
from torch_cluster import radius_graph
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

assert torch.cuda.is_available(), 'CUDA unavailable'
reports = []
for device_id in range(torch.cuda.device_count()):
    device = torch.device(f'cuda:{device_id}')
    x = torch.randn(256, 256, device=device, requires_grad=True)
    (x @ x.T).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    pos = torch.tensor([[0., 0., 0.], [0.5, 0., 0.], [3., 0., 0.]], device=device)
    edges = radius_graph(pos, r=1.0)
    assert edges.shape == (2, 2)
    values = scatter_add(torch.ones(2, device=device), edges[0], dim_size=3)
    torch.testing.assert_close(values, torch.tensor([1., 1., 0.], device=device))
    sparse = SparseTensor(row=edges[0], col=edges[1], value=torch.ones(2, device=device), sparse_sizes=(3, 3))
    torch.testing.assert_close(sparse.matmul(torch.ones(3, 1, device=device)).flatten(), values)
    torch.cuda.synchronize(device)
    reports.append({'device': str(device), 'name': torch.cuda.get_device_name(device), 'capability': torch.cuda.get_device_capability(device), 'dense_backward': 'passed', 'pyg_cuda_ops': 'passed'})
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'torch_geometric': torch_geometric.__version__, 'devices': reports}, indent=2))
