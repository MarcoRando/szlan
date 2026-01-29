import pytest
import torch

# 1. Global reproducible seed
@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

# 2. Device fixture
@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    return torch.device(request.param)

# 3. Common generator parameters
@pytest.fixture
def gen_params(device):
    return dict(d=5, l=5, s=1, seed=12345, device=device, dtype=torch.float64)
