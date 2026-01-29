import pytest

import torch 


import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from szlan.direction_generators.direction_generators import QRDirectionGenerator
from szlan.optimizer.szlan import SZLan, SZLanPhase


def target_function(x):
    return torch.sum(x**2, dim=1)


@pytest.mark.parametrize("params", [
    {"d":5, "l":5, "s":4, "popsize" : 50, "device":"cpu", "dtype":torch.float64, "seed":12345, "h" : 1e-7},
    {"d":100, "l":100, "s":20, "popsize" : 100, "device":"cpu", "dtype":torch.float64, "seed":12345, "h" : 1e-7},
    {"d":8, "l":8, "s":3, "popsize" : 10, "device":"cpu", "dtype":torch.float64, "seed":12345, "h" : 1e-7},
    {"d":10, "l":10, "s":4,"popsize" : 1, "device":"cpu", "dtype":torch.float64, "seed":12345, "h" : 1e-7},
    {"d":500, "l":500, "s":1,"popsize" : 10, "device":"cpu", "dtype":torch.float64, "seed":12345, "h" : 1e-7},
    {"d":50, "l":50, "s":7,"popsize" : 100, "device":"cpu", "dtype":torch.float64, "seed":12345, "h" : 1e-7},
    {"d":100, "l":100, "s":4,"popsize" : 10, "device":"cpu", "dtype":torch.float64, "seed":12345, "h" : 1e-7},
    {"d":100, "l":100, "s":10,"popsize" : 10, "device":"cpu", "dtype":torch.float64, "seed":12345, "h" : 1e-5}
])
def test_szlan_gradient_approx_quality(params):
    d, l, s = params["d"], params["l"], params["s"]
    device, dtype, seed = params["device"], params["dtype"], params["seed"]
    popsize = params["popsize"]
    h = params["h"]
    population = torch.randn((popsize, d), generator=torch.Generator(device=device).manual_seed(seed), device=device, dtype=dtype)
    f_current = target_function(population)
    direction_generator = QRDirectionGenerator(d=d, l=l, s=s, device=device, dtype=dtype)
    
    P = direction_generator()

    f_next = target_function((population[:, None, None, :] + h * P[None, :, :, :]).reshape(-1, d))

    opt = SZLan(population=population, direction_generator=direction_generator, h=h, gamma=0.01, beta=100000.0, seed=seed, device=seed, dtype=dtype)
    opt.P = P
    opt.current_values = f_current
    opt.forward_values = f_next
    g = opt._approx_gradient(h)

    assert torch.allclose(2 * population, g, atol=1e-5)

