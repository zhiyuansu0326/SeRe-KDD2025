import torch


def update_utilities(network, eta=0.9):
    if network.last_h1 is None or network.last_x is None:
        return
    h1 = network.last_h1.detach()
    if h1.ndim > 1:
        h1 = h1.mean(dim=0)
    w1 = network.fc1.weight.detach()
    w1_contribution = torch.norm(w1, dim=1)
    new_utils1 = h1 * w1_contribution
    network.utils[1] = eta * network.utils[1] + (1 - eta) * new_utils1
    w2 = network.fc2.weight.detach()
    w2_contribution = torch.norm(w2, dim=1)
    h2 = network.fc2(network.last_h1).detach().mean(dim=0)
    new_utils2 = h2 * w2_contribution
    network.utils[2] = eta * network.utils[2] + (1 - eta) * new_utils2


def increment_ages(network):
    for layer in [1, 2]:
        network.ages[layer] += 1


def apply_sere(network, rho, maturity_threshold, device):
    for layer in [1, 2]:
        expected_size = (
            network.fc1.out_features if layer == 1 else network.fc2.in_features
        )
        assert network.utils[layer].shape[0] == expected_size, "Utils count mismatch"
        mature_mask = network.ages[layer] > maturity_threshold
        mature_units = mature_mask.sum().item()
        network.counters[layer] += rho * mature_units
        if network.counters[layer] >= 1 and mature_units > 0:
            mature_indices = torch.nonzero(mature_mask, as_tuple=False).squeeze()
            if mature_indices.ndim == 0:
                mature_indices = mature_indices.unsqueeze(0)
            utilities_mature = network.utils[layer][mature_indices]
            min_idx_in_mature = torch.argmin(utilities_mature).item()
            min_util_idx = mature_indices[min_idx_in_mature].item()
            if layer == 1:
                network.fc1.weight.data[min_util_idx, :] = (
                    torch.randn_like(network.fc1.weight.data[min_util_idx, :]) * 0.1
                )
                if hasattr(network.fc1, "bias") and network.fc1.bias is not None:
                    network.fc1.bias.data[min_util_idx] = 0.0
            elif layer == 2:
                network.fc2.weight.data[:, min_util_idx] = 0.0
            network.utils[layer][min_util_idx] = 0
            network.ages[layer][min_util_idx] = 0
            network.counters[layer] -= 1
