import unittest

import numpy as np
import torch
from torch.autograd import Variable

import src.train


class TestTrainer(unittest.TestCase):

    def test_prepare_translated_variable(self):
        batch = torch.from_numpy(np.array([
            [997, 1831],
            [51540, 92],
            [3, 26],
            [2770, 21],
            [3, 412],
            [267, 575],
            [42681, 42724],
            [2, 2]
        ], dtype=np.int32))

        translated = np.array([
            [2593, 3, 3, 4864, 3, 680, 3, 2, 0, 0],
            [4311, 237, 60, 37, 1097, 1527, 3, 2, 0, 0]
        ])

        def translation(variable, lengths):
            return Variable(torch.from_numpy(translated.T), requires_grad=False)

        var, lengths = src.train.Trainer.prepare_translated_variable(translation, Variable(batch, requires_grad=False))
        np.testing.assert_array_equal(np.array([10, 10]), lengths)
        np.testing.assert_array_equal(translated.T, var.data)

    def test_add_noise(self):
        res = src.train.Trainer.add_noise(np.array([997, 51540, 3, 2770, 3, 267, 42681, 2]))
        assert res[-1] == 2
        assert len(res) > 0


if __name__ == '__main__':
    unittest.main()
