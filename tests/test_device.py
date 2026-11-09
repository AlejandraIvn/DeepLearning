"""Check CPU fallback and exercise real CUDA transfers and AMP when available."""

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.device import select_device


class DeviceTests(unittest.TestCase):
    def test_cpu_fallback_keeps_float32_and_disables_amp(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            device = select_device()
        self.assertEqual(device.device, torch.device("cpu"))
        self.assertFalse(device.pin_memory)
        self.assertFalse(device.make_grad_scaler().is_enabled())
        images, labels = device.move_batch((torch.ones(2, 2), torch.zeros(2)))
        with device.autocast():
            result = images @ images
        self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(labels.is_cuda)
        torch.testing.assert_close(result, torch.full((2, 2), 2.0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA no disponible en este entorno")
    def test_cuda_pinned_batch_and_amp_backward(self) -> None:
        device = select_device()
        self.assertEqual(device.device, torch.device("cuda:0"))
        dataset = TensorDataset(torch.full((2, 3), 0.125), torch.ones(2))
        loader = DataLoader(dataset, batch_size=2, pin_memory=device.pin_memory)
        batch = next(iter(loader))
        self.assertTrue(all(tensor.is_pinned() for tensor in batch))
        images, labels = device.move_batch(batch)
        self.assertTrue(images.is_cuda and labels.is_cuda)

        weight = torch.nn.Parameter(torch.zeros((3, 1), device=device.device))
        optimizer = torch.optim.SGD([weight], lr=0.01)
        scaler = device.make_grad_scaler()
        self.assertTrue(scaler.is_enabled())
        with device.autocast():
            logits = (images @ weight).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        self.assertEqual(logits.dtype, torch.float16)
        self.assertEqual(loss.dtype, torch.float32)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.testing.assert_close(weight, torch.full_like(weight, 0.000625))
