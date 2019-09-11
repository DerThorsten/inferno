import unittest
import pytest

import torch
from inferno.utils.model_utils import ModelTester
from inferno.extensions.models.unet.unet import UNet




class UNetTest(unittest.TestCase):


    def test_default_2d(self):
        tester = ModelTester((1, 1, 256, 256), (1, 1, 256, 256))
        if torch.cuda.is_available():
            tester.cuda()


        model = UNet(in_channels=1, out_channels=1, 
                     dim=2, initial_features=32, 
                     depth=3, gain=3)