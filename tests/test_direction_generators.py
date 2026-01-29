import torch
import pytest
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from szlan.direction_generators.direction_generators import QRDirectionGenerator




def test_qr_generator_output_shape(gen_params):
    qr_gen = QRDirectionGenerator(**gen_params)
    Q = qr_gen()
    s, l, d = gen_params["s"], gen_params["l"], gen_params["d"]
    assert Q.shape == (s, l, d)


def test_qr_generator_orthogonality(gen_params):
    qr_gen = QRDirectionGenerator(**gen_params)
    Q = qr_gen()
    s, l, d = gen_params["s"], gen_params["l"], gen_params["d"]
    assert torch.allclose(torch.matmul(Q, Q.transpose(1,2)), torch.eye(l, device=gen_params["device"], dtype=gen_params["dtype"]))