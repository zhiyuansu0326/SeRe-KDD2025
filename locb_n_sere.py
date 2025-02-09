from collections import defaultdict
import numpy as np
import random
import sys
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim


class Cluster:
    def __init__(self, users, S, b, N):
        self.users = set(users)
        self.S = S
        self.b = b
        self.N = N
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)


class Network_u(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_u, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc_phi = nn.Linear(hidden_size, dim)
        self.fc_y = nn.Linear(hidden_size, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.utils = {
            1: torch.zeros(self.fc1.out_features).to(device),
            2: torch.zeros(self.fc_y.out_features).to(device),
        }
        self.ages = {
            1: torch.zeros(self.fc1.out_features).to(device),
            2: torch.zeros(self.fc_y.out_features).to(device),
        }
        self.counters = {1: 0.0, 2: 0.0}
        self.last_x = None
        self.last_h = None

    def forward(self, x):
        self.last_x = x
        h = self.activate(self.fc1(x))
        self.last_h = h
        phi_x = self.fc_phi(h)
        y_pred = self.fc_y(h)
        return phi_x, y_pred

    def update_utilities(self, eta=0.9):
        if self.last_h is None or self.last_x is None:
            return
        h = self.last_h.detach()
        if len(h.shape) > 1:
            h_mean = h.mean(dim=0)
        else:
            h_mean = h
        w1 = self.fc1.weight.detach()
        w1_norm = torch.norm(w1, dim=1)
        new_utils1 = h_mean * w1_norm
        self.utils[1] = eta * self.utils[1] + (1 - eta) * new_utils1

        w2 = self.fc_y.weight.detach()
        w2_norm = torch.norm(w2, dim=1)
        y_pred = self.fc_y(self.last_h).detach().squeeze()
        if y_pred.dim() == 0:
            y_pred = y_pred.unsqueeze(0)
        new_utils2 = y_pred * w2_norm
        self.utils[2] = eta * self.utils[2] + (1 - eta) * new_utils2

    def increment_ages(self):
        for layer in [1, 2]:
            self.ages[layer] += 1


class LOCB_N_SERE:
    def __init__(
        self,
        nu,
        d,
        gamma,
        num_seeds,
        delta,
        detect_cluster,
        hidden_size=100,
        learning_rate=1e-3,
        update_interval=100,
    ):
        self.S = {i: np.eye(d) for i in range(nu)}
        self.b = {i: np.zeros(d) for i in range(nu)}
        self.Sinv = {i: np.eye(d) for i in range(nu)}
        self.theta = {i: np.zeros(d) for i in range(nu)}
        self.users = range(nu)
        self.seeds = np.random.choice(list(self.users), num_seeds, replace=False)
        self.seed_state = {seed: 0 for seed in self.seeds}
        self.clusters = {
            seed: Cluster(users=self.users, S=np.eye(d), b=np.zeros(d), N=1)
            for seed in self.seeds
        }
        self.N = np.zeros(nu)
        self.gamma = gamma
        self.results = []
        self.fin = 0
        self.cluster_inds = {
            i: [seed for seed in self.seeds if i in self.clusters[seed].users]
            for i in self.users
        }
        self.d = d
        self.n = nu
        self.selected_cluster = 0
        self.delta = delta
        self.if_d = detect_cluster

        self.input_dim = d
        self.hidden_size = hidden_size
        self.output_dim = 1
        self.network = Network_u(self.input_dim, hidden_size=self.hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_interval = update_interval
        self.history = []
        self.current_step = 0

        self.rho_min = 0.01
        self.rho_max = 0.1
        self.rho = self.rho_min
        self.maturity_threshold = 100
        self.PH = 0.0
        self.PH_min = 0.0
        self.PH_threshold = 0.5
        self.PH_delta = 0.1
        self.alpha = 0.01

    def _beta(self, N, t):
        return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        ucbs = np.dot(items, theta) + self._beta(N, t) * (
            np.matmul(items, Sinv) * items
        ).sum(axis=1)
        it = np.argmax(ucbs)
        return it

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)
        return Sinv, theta

    def recommend(self, i, items, t):
        cls = self.cluster_inds[i]
        if cls and t < 40000:
            res = [
                self._select_item_ucb(
                    self.clusters[c].S,
                    self.clusters[c].Sinv,
                    self.clusters[c].theta,
                    items,
                    self.clusters[c].N,
                    t,
                )
                for c in cls
            ]
            return max(res)
        return self._select_item_ucb(
            self.S[i], self.Sinv[i], self.theta[i], items, self.N[i], t
        )

    def store_info(self, i, x, y, t):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            phi_x_tensor, y_pred_tensor = self.network(x_tensor)
            phi_x = phi_x_tensor.numpy().flatten()
            pred_val = y_pred_tensor.item()

        self.S[i] += np.outer(phi_x, phi_x)
        self.b[i] += y * phi_x
        self.N[i] += 1
        self.Sinv[i], self.theta[i] = self._update_inverse(
            self.S[i], self.b[i], self.Sinv[i], phi_x, self.N[i]
        )

        for c in self.cluster_inds[i]:
            self.clusters[c].S += np.outer(phi_x, phi_x)
            self.clusters[c].b += y * phi_x
            self.clusters[c].N += 1
            self.clusters[c].Sinv = np.linalg.inv(self.clusters[c].S)
            self.clusters[c].theta = np.matmul(
                self.clusters[c].Sinv, self.clusters[c].b
            )

        self.history.append((i, phi_x, y))
        self.current_step += 1

        self.network.update_utilities()
        self.network.increment_ages()
        self.update_PH(y, pred_val)
        self.apply_sere(self.network)

    def update(self, i, t):
        def _factT(m):
            if self.if_d:
                delta = self.delta / self.n
                nu_val = (
                    np.sqrt(2 * self.d * np.log(1 + t) + 2 * np.log(2 / self.delta)) + 1
                )
                de = np.sqrt(1 + m / 4) * np.power(self.n, 1 / 3)
                return nu_val / de
            return np.sqrt((1 + np.log(1 + m)) / (1 + m))

        if not self.fin:
            for seed in self.seeds:
                if not self.seed_state[seed]:
                    if i in self.clusters[seed].users:
                        diff = self.theta[i] - self.theta[seed]
                        if np.linalg.norm(diff) > _factT(self.N[i]) + _factT(
                            self.N[seed]
                        ):
                            self.clusters[seed].users.remove(i)
                            self.cluster_inds[i].remove(seed)
                            self.clusters[seed].S = (
                                self.clusters[seed].S - self.S[i] + np.eye(self.d)
                            )
                            self.clusters[seed].b = self.clusters[seed].b - self.b[i]
                            self.clusters[seed].N = self.clusters[seed].N - self.N[i]
                    else:
                        diff = self.theta[i] - self.theta[seed]
                        if np.linalg.norm(diff) < _factT(self.N[i]) + _factT(
                            self.N[seed]
                        ):
                            self.clusters[seed].users.add(i)
                            self.cluster_inds[i].append(seed)
                            self.clusters[seed].S = (
                                self.clusters[seed].S + self.S[i] - np.eye(self.d)
                            )
                            self.clusters[seed].b = self.clusters[seed].b + self.b[i]
                            self.clusters[seed].N = self.clusters[seed].N + self.N[i]

                    thre = self.gamma if self.if_d else self.gamma / 4
                    if _factT(self.N[seed]) <= thre:
                        self.seed_state[seed] = 1
                        self.results.append({seed: list(self.clusters[seed].users)})

            if all(state == 1 for state in self.seed_state.values()):
                if self.if_d:
                    np.save("./results/clusters", self.results)
                    print("Clustering finished! Round:", t)
                self.fin = 1

        if self.current_step % self.update_interval == 0 and self.current_step > 0:
            self.train_network()

    def train_network(self, batch_size=32):
        if len(self.history) < batch_size:
            return
        batch = random.sample(self.history, batch_size)
        user_ids, features, rewards = zip(*batch)
        features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32)
        preds = self.network(features_tensor)[1].squeeze()
        loss = self.loss_fn(preds, rewards_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.history = []

    def update_PH(self, reward, pred_reward):
        self.PH += reward - pred_reward - self.PH_delta
        self.PH_min = min(self.PH_min, self.PH)
        if self.PH - self.PH_min > self.PH_threshold:
            self.rho = self.rho_max
        else:
            self.rho = self.rho_min + self.alpha * (self.PH - self.PH_min)

    def apply_sere(self, network):
        for layer in [1, 2]:
            if layer == 1:
                expected_size = network.fc1.out_features
            elif layer == 2:
                expected_size = network.fc_y.out_features
            mature_mask = network.ages[layer] > self.maturity_threshold
            mature_units = mature_mask.sum().item()
            network.counters[layer] += self.rho * mature_units
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
                    network.fc_y.weight.data[min_util_idx, :] = 0.0
                network.utils[layer][min_util_idx] = 0
                network.ages[layer][min_util_idx] = 0
                network.counters[layer] -= 1
