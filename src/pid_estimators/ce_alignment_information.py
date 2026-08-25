# coding: utf-8
"""Device-agnostic BATCH partial information decomposition estimator.

This module adapts ``estimators/ce_alignment_information.py`` from
https://github.com/pliang279/PID (see ``LICENSE.PID``).  The estimator and PID
definitions are unchanged; device selection, deterministic data loading, and
numerically stable affinity exponentiation are added for reproducible
checkpoint analysis.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class MultimodalDataset(Dataset):
    """Dataset returning all modalities followed by the discrete label."""

    def __init__(self, data, labels):
        if not data:
            raise ValueError("At least one modality tensor is required.")
        length = int(labels.shape[0])
        if any(int(modality.shape[0]) != length for modality in data):
            raise ValueError("All modalities and labels must have equal length.")
        self.data = tuple(data)
        self.labels = labels

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, index):
        return tuple(modality[index] for modality in self.data) + (
            self.labels[index],
        )


def _mlp(input_dim, hidden_dim, output_dim, layers, activation):
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
    if activation not in activations:
        raise ValueError("Unsupported activation: {}".format(activation))
    activation_type = activations[activation]
    modules = [nn.Linear(input_dim, hidden_dim), activation_type()]
    for _ in range(layers):
        modules.extend((nn.Linear(hidden_dim, hidden_dim), activation_type()))
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, layers, activation):
        super().__init__()
        self.network = _mlp(
            input_dim, hidden_dim, num_labels, layers, activation
        )

    def forward(self, *inputs):
        return self.network(torch.cat(inputs, dim=-1))


def _sinkhorn_probs(matrix, row_targets, column_targets, tolerance):
    matrix = matrix / matrix.sum(dim=0, keepdim=True).clamp_min(1e-8)
    matrix = matrix * column_targets.unsqueeze(0)
    if torch.allclose(
        matrix.sum(dim=1), row_targets, rtol=0.0, atol=tolerance
    ):
        return matrix, True
    matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1e-8)
    matrix = matrix * row_targets.unsqueeze(1)
    if torch.allclose(
        matrix.sum(dim=0), column_targets, rtol=0.0, atol=tolerance
    ):
        return matrix, True
    return matrix, False


class CEAlignment(nn.Module):
    def __init__(
        self,
        x1_dim,
        x2_dim,
        hidden_dim,
        embed_dim,
        num_labels,
        layers,
        activation,
        sinkhorn_iterations=500,
        sinkhorn_tolerance=0.01,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.embed_dim = int(embed_dim)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_tolerance = float(sinkhorn_tolerance)
        self.mlp1 = _mlp(
            x1_dim,
            hidden_dim,
            embed_dim * num_labels,
            layers,
            activation,
        )
        self.mlp2 = _mlp(
            x2_dim,
            hidden_dim,
            embed_dim * num_labels,
            layers,
            activation,
        )

    def forward(self, x1, x2, x1_probs, x2_probs):
        q_x1 = self.mlp1(x1).unflatten(1, (self.num_labels, -1))
        q_x2 = self.mlp2(x2).unflatten(1, (self.num_labels, -1))
        q_x1 = (q_x1 - q_x1.mean(dim=2, keepdim=True)) / torch.sqrt(
            q_x1.var(dim=2, keepdim=True) + 1e-8
        )
        q_x2 = (q_x2 - q_x2.mean(dim=2, keepdim=True)) / torch.sqrt(
            q_x2.var(dim=2, keepdim=True) + 1e-8
        )
        logits = torch.einsum("ahx,bhx->abh", q_x1, q_x2) / math.sqrt(
            self.embed_dim
        )

        # A constant shift per label leaves Sinkhorn's normalized coupling
        # unchanged while preventing exp overflow.
        logits = logits - logits.amax(dim=(0, 1), keepdim=True).detach()
        affinity = torch.exp(logits)
        normalized = []
        for label_id in range(self.num_labels):
            current = affinity[..., label_id]
            for _ in range(self.sinkhorn_iterations):
                current, converged = _sinkhorn_probs(
                    current,
                    x1_probs[:, label_id],
                    x2_probs[:, label_id],
                    self.sinkhorn_tolerance,
                )
                if converged:
                    break
            normalized.append(current)
        coupling = torch.stack(normalized, dim=-1)
        if not torch.isfinite(coupling).all():
            raise FloatingPointError("BATCH Sinkhorn coupling is not finite.")
        return coupling


class CEAlignmentInformation(nn.Module):
    """Learn the BATCH coupling and calculate ``[R, U1, U2, S]``."""

    def __init__(
        self,
        x1_dim,
        x2_dim,
        hidden_dim,
        embed_dim,
        num_labels,
        layers,
        activation,
        discriminator_1,
        discriminator_2,
        discriminator_12,
        p_y,
        sinkhorn_iterations=500,
        sinkhorn_tolerance=0.01,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.alignment = CEAlignment(
            x1_dim,
            x2_dim,
            hidden_dim,
            embed_dim,
            num_labels,
            layers,
            activation,
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_tolerance=sinkhorn_tolerance,
        )
        self.discriminator_1 = discriminator_1
        self.discriminator_2 = discriminator_2
        self.discriminator_12 = discriminator_12
        self.register_buffer("p_y", p_y)
        for discriminator in (
            self.discriminator_1,
            self.discriminator_2,
            self.discriminator_12,
        ):
            discriminator.eval()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)

    def alignment_parameters(self):
        return self.alignment.parameters()

    def forward(self, x1, x2, labels):
        with torch.no_grad():
            p_y_x1 = F.softmax(self.discriminator_1(x1), dim=-1)
            p_y_x2 = F.softmax(self.discriminator_2(x2), dim=-1)
            p_y_x1x2 = F.softmax(self.discriminator_12(x1, x2), dim=-1)

        coupling = self.alignment(
            x1.flatten(1), x2.flatten(1), p_y_x1, p_y_x2
        )
        q_x2_x1y = coupling / coupling.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mixture = torch.einsum("aby,ay->ab", q_x2_x1y, p_y_x1)
        log_term = torch.log(q_x2_x1y.clamp_min(1e-8)) - torch.log(
            mixture.clamp_min(1e-8)
        ).unsqueeze(-1)

        alignment_objective = torch.mean(
            torch.sum(
                torch.sum(
                    p_y_x1.unsqueeze(1) * q_x2_x1y * log_term,
                    dim=-1,
                ),
                dim=-1,
            )
        )

        log_p_y = torch.log(self.p_y.clamp_min(1e-8))
        mi_y_x1 = torch.mean(
            torch.sum(p_y_x1 * (torch.log(p_y_x1.clamp_min(1e-8)) - log_p_y), dim=-1)
        )
        mi_y_x2 = torch.mean(
            torch.sum(p_y_x2 * (torch.log(p_y_x2.clamp_min(1e-8)) - log_p_y), dim=-1)
        )
        mi_y_x1x2 = torch.mean(
            torch.sum(
                p_y_x1x2
                * (torch.log(p_y_x1x2.clamp_min(1e-8)) - log_p_y),
                dim=-1,
            )
        )
        mi_q_y_x1x2 = p_y_x1.unsqueeze(1) * q_x2_x1y * (
            log_term
            + torch.log(p_y_x1.clamp_min(1e-8)).unsqueeze(1)
            - log_p_y.view(1, 1, -1)
        )
        mi_q_y_x1x2 = torch.mean(mi_q_y_x1x2.sum(dim=(-1, -2)))

        redundancy = mi_y_x1 + mi_y_x2 - mi_q_y_x1x2
        unique_1 = mi_q_y_x1x2 - mi_y_x2
        unique_2 = mi_q_y_x1x2 - mi_y_x1
        synergy = mi_y_x1x2 - mi_q_y_x1x2
        pid = torch.stack((redundancy, unique_1, unique_2, synergy))
        return alignment_objective, pid, coupling


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(dataset, batch_size, shuffle, drop_last, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        generator=generator,
    )


def _train_discriminator(
    model,
    dataset,
    selector,
    device,
    batch_size,
    epochs,
    seed,
):
    loader = _make_loader(dataset, batch_size, True, True, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    final_loss = float("nan")
    for _ in range(epochs):
        for x1, x2, labels in loader:
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)
            labels = labels.long().view(-1).to(device)
            inputs = selector(x1, x2)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(*inputs), labels)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    model.eval()
    return final_loss


def _train_alignment(
    model,
    dataset,
    device,
    batch_size,
    epochs,
    seed,
):
    loader = _make_loader(dataset, batch_size, True, True, seed)
    optimizer = torch.optim.Adam(model.alignment_parameters(), lr=1e-3)
    model.train()
    final_loss = float("nan")
    for _ in range(epochs):
        for x1, x2, labels in loader:
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)
            labels = labels.long().view(-1, 1).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(x1, x2, labels)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    model.eval()
    return final_loss


@torch.no_grad()
def _evaluate_alignment(model, dataset, device, batch_size, seed):
    loader = _make_loader(dataset, batch_size, False, True, seed)
    results = []
    alignments = []
    for x1, x2, labels in loader:
        x1 = x1.float().to(device)
        x2 = x2.float().to(device)
        labels = labels.long().view(-1, 1).to(device)
        _, pid, coupling = model(x1, x2, labels)
        results.append(pid.cpu())
        alignments.append(coupling.cpu())
    if not results:
        raise ValueError(
            "PID test split must contain at least one complete batch."
        )
    return torch.stack(results), alignments


def critic_ce_alignment(
    x1,
    x2,
    labels,
    num_labels,
    train_ds,
    test_ds,
    discrim_epochs=40,
    ce_epochs=10,
    batch_size=256,
    device="cuda:0",
    seed=999,
    sinkhorn_iterations=500,
    sinkhorn_tolerance=0.01,
    verbose=True,
):
    """Train BATCH critics and return per-test-batch PID estimates.

    The returned PID column order is ``[R, U1, U2, S]``.  In the checkpoint
    analysis, ``x1`` is Full and ``x2`` is Masked, so U1 and U2 correspond to
    ``U_full`` and ``U_mask`` respectively.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("BATCH expects x1 and x2 with shape [N, D].")
    if x1.shape[0] != x2.shape[0] or x1.shape[0] != labels.shape[0]:
        raise ValueError("x1, x2, and labels must have equal sample counts.")
    if len(train_ds) < batch_size or len(test_ds) < batch_size:
        raise ValueError("Both PID splits must contain at least one batch.")
    if discrim_epochs < 1 or ce_epochs < 1 or batch_size < 2:
        raise ValueError("Epoch counts must be positive and batch_size >= 2.")

    labels = labels.long().view(-1, 1)
    counts = torch.bincount(labels.view(-1), minlength=num_labels)
    if counts.numel() != num_labels or torch.any(counts == 0):
        raise ValueError("Every PID label must occur at least once.")

    _seed_everything(int(seed))
    discriminator_1 = Discriminator(
        x1.shape[1], 32, num_labels, 3, "relu"
    ).to(device)
    discriminator_2 = Discriminator(
        x2.shape[1], 32, num_labels, 3, "relu"
    ).to(device)
    discriminator_12 = Discriminator(
        x1.shape[1] + x2.shape[1], 32, num_labels, 3, "relu"
    ).to(device)

    discriminator_losses = (
        _train_discriminator(
            discriminator_1,
            train_ds,
            lambda first, second: (first,),
            device,
            batch_size,
            discrim_epochs,
            seed + 11,
        ),
        _train_discriminator(
            discriminator_2,
            train_ds,
            lambda first, second: (second,),
            device,
            batch_size,
            discrim_epochs,
            seed + 12,
        ),
        _train_discriminator(
            discriminator_12,
            train_ds,
            lambda first, second: (first, second),
            device,
            batch_size,
            discrim_epochs,
            seed + 13,
        ),
    )

    p_y = counts.to(device=device, dtype=torch.float32) / labels.shape[0]
    model = CEAlignmentInformation(
        x1.shape[1],
        x2.shape[1],
        hidden_dim=32,
        embed_dim=10,
        num_labels=num_labels,
        layers=3,
        activation="relu",
        discriminator_1=discriminator_1,
        discriminator_2=discriminator_2,
        discriminator_12=discriminator_12,
        p_y=p_y,
        sinkhorn_iterations=sinkhorn_iterations,
        sinkhorn_tolerance=sinkhorn_tolerance,
    ).to(device)
    alignment_loss = _train_alignment(
        model,
        train_ds,
        device,
        batch_size,
        ce_epochs,
        seed + 21,
    )
    results, alignments = _evaluate_alignment(
        model, test_ds, device, batch_size, seed + 22
    )
    if not torch.isfinite(results).all():
        raise FloatingPointError("BATCH returned non-finite PID estimates.")
    if verbose:
        print(
            "BATCH seed={}: discriminator_losses={}, alignment_loss={:.6f}, "
            "eval_batches={}".format(
                seed,
                tuple(round(value, 6) for value in discriminator_losses),
                alignment_loss,
                int(results.shape[0]),
            )
        )
    return results, alignments, (
        model,
        discriminator_1,
        discriminator_2,
        discriminator_12,
        p_y,
    )
