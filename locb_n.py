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
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class LOCB_N:
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
        self.seeds = np.random.choice(self.users, num_seeds)
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
            phi_x = self.network(x_tensor).numpy().flatten()
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

    def update(self, i, t):
        def _factT(m):
            if self.if_d:
                delta = self.delta / self.n
                nu = (
                    np.sqrt(2 * self.d * np.log(1 + t) + 2 * np.log(2 / self.delta)) + 1
                )
                de = np.sqrt(1 + m / 4) * np.power(self.n, 1 / 3)
                return nu / de
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
        preds = self.network(features_tensor).squeeze()
        loss = self.loss_fn(preds, rewards_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.history = []
