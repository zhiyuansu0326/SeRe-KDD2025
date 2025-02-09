import numpy as np
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim


def isInvertible(S):
    return np.linalg.cond(S) < 1 / sys.float_info.epsilon


class Base:
    def __init__(self, d):
        self.d = d

    def _beta(self, N, t):
        return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        return np.argmax(
            np.dot(items, theta)
            + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis=1)
        )

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)
        return Sinv, theta


class LinUCB_IND(Base):
    def __init__(self, nu, d):
        super(LinUCB_IND, self).__init__(d)
        self.S = {i: np.eye(d) for i in range(nu)}
        self.b = {i: np.zeros(d) for i in range(nu)}
        self.Sinv = {i: np.eye(d) for i in range(nu)}
        self.theta = {i: np.zeros(d) for i in range(nu)}

        self.N = np.zeros(nu)

    def recommend(self, i, items, t):
        return self._select_item_ucb(
            self.S[i], self.Sinv[i], self.theta[i], items, self.N[i], t
        )

    def store_info(self, i, x, y, t, r, br):
        self.S[i] += np.outer(x, x)
        self.b[i] += y * x
        self.N[i] += 1

        self.Sinv[i], self.theta[i] = self._update_inverse(
            self.S[i], self.b[i], self.Sinv[i], x, self.N[i]
        )


class Cluster:
    def __init__(self, users, S, b, N, checks):
        self.users = users
        self.S = S
        self.b = b
        self.N = N
        self.checks = checks

        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)
        self.checked = len(self.users) == sum(self.checks.values())

    def update_check(self, i):
        self.checks[i] = True
        self.checked = len(self.users) == sum(self.checks.values())


class Network_u(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_u, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc_phi = nn.Linear(hidden_size, dim) 
        self.fc_y = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.activate(self.fc1(x))
        phi_x = self.fc_phi(h)
        y_pred = self.fc_y(h)
        return phi_x, y_pred


class SCLUB_N(LinUCB_IND):
    def __init__(
        self,
        nu,
        d,
        num_stages=10,
        hidden_size=100,
        learning_rate=1e-3,
        update_interval=100,
    ):
        super(SCLUB_N, self).__init__(nu, d)

        self.clusters = {
            0: Cluster(
                users=[i for i in range(nu)],
                S=np.eye(d),
                b=np.zeros(d),
                N=0,
                checks={i: False for i in range(nu)},
            )
        }
        self.cluster_inds = np.zeros(nu, dtype=int)

        self.num_stages = num_stages
        self.num_clusters = np.ones(20000)

        self.input_dim = d 
        self.hidden_size = hidden_size
        self.output_dim = d
        self.network = Network_u(self.input_dim, hidden_size=self.hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_interval = update_interval
        self.history = []
        self.current_step = 0

    def _init_each_stage(self):
        for c in self.clusters:
            self.clusters[c].checks = {i: False for i in self.clusters[c].users}
            self.clusters[c].checked = False

    def recommend(self, i, items, t):
        cluster = self.clusters[self.cluster_inds[i]]
        return self._select_item_ucb(
            cluster.S, cluster.Sinv, cluster.theta, items, cluster.N, t
        )

    def store_info(self, i, x, y, t, r, br=1):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0) 
            phi_x, y_pred = self.network(x_tensor)
            phi_x = phi_x.squeeze().numpy() 

        self.S[i] += np.outer(phi_x, phi_x)
        self.b[i] += y * phi_x
        self.N[i] += 1

        self.Sinv[i], self.theta[i] = self._update_inverse(
            self.S[i], self.b[i], self.Sinv[i], phi_x, self.N[i]
        )

        c = self.cluster_inds[i]
        self.clusters[c].S += np.outer(phi_x, phi_x)
        self.clusters[c].b += y * phi_x
        self.clusters[c].N += 1
        self.clusters[c].Sinv, self.clusters[c].theta = self._update_inverse(
            self.clusters[c].S,
            self.clusters[c].b,
            self.clusters[c].Sinv,
            phi_x,
            self.clusters[c].N,
        )

        self.history.append((i, x, y))

        self.current_step += 1
        if self.current_step % self.update_interval == 0:
            self.train_network()

    def _factT(self, T):
        return np.sqrt((1 + np.log(1 + T)) / (1 + T))

    def _split_or_merge(self, theta, N1, N2, split=True):
        alpha = 1
        if split:
            return np.linalg.norm(theta) > alpha * (self._factT(N1) + self._factT(N2))
        else:
            return (
                np.linalg.norm(theta) < alpha * (self._factT(N1) + self._factT(N2)) / 2
            )

    def _cluster_avg_freq(self, c, t):
        return self.clusters[c].N / (len(self.clusters[c].users) * t)

    def _split_or_merge_p(self, p1, p2, t, split=True):
        alpha_p = np.sqrt(2)
        if split:
            return np.abs(p1 - p2) > alpha_p * self._factT(t)
        else:
            return np.abs(p1 - p2) < alpha_p * self._factT(t) / 2

    def split(self, i, t):
        c = self.cluster_inds[i]
        cluster = self.clusters[c]

        cluster.update_check(i)

        if self._split_or_merge_p(
            self.N[i] / (t + 1), self._cluster_avg_freq(c, t + 1), t + 1, split=True
        ) or self._split_or_merge(
            self.theta[i] - cluster.theta, self.N[i], cluster.N, split=True
        ):

            def _find_available_index():
                cmax = max(self.clusters)
                for c1 in range(cmax + 1):
                    if c1 not in self.clusters:
                        return c1
                return cmax + 1

            cnew = _find_available_index()
            self.clusters[cnew] = Cluster(
                users=[i], S=self.S[i], b=self.b[i], N=self.N[i], checks={i: True}
            )
            self.cluster_inds[i] = cnew

            cluster.users.remove(i)
            cluster.S = cluster.S - self.S[i] + np.eye(self.d)
            cluster.b = cluster.b - self.b[i]
            cluster.N = cluster.N - self.N[i]
            del cluster.checks[i]

    def merge(self, t):
        cmax = max(self.clusters)

        for c1 in range(cmax + 1):
            if c1 not in self.clusters or not self.clusters[c1].checked:
                continue

            for c2 in range(c1 + 1, cmax + 1):
                if c2 not in self.clusters or not self.clusters[c2].checked:
                    continue

                if self._split_or_merge(
                    self.clusters[c1].theta - self.clusters[c2].theta,
                    self.clusters[c1].N,
                    self.clusters[c2].N,
                    split=False,
                ) and self._split_or_merge_p(
                    self._cluster_avg_freq(c1, t + 1),
                    self._cluster_avg_freq(c2, t + 1),
                    t + 1,
                    split=False,
                ):

                    for i in self.clusters[c2].users:
                        self.cluster_inds[i] = c1

                    self.clusters[c1].users = (
                        self.clusters[c1].users + self.clusters[c2].users
                    )
                    self.clusters[c1].S = (
                        self.clusters[c1].S + self.clusters[c2].S - np.eye(self.d)
                    )
                    self.clusters[c1].b = self.clusters[c1].b + self.clusters[c2].b
                    self.clusters[c1].N = self.clusters[c1].N + self.clusters[c2].N
                    self.clusters[c1].checks = {
                        **self.clusters[c1].checks,
                        **self.clusters[c2].checks,
                    }

                    del self.clusters[c2]

    def train_network(self, batch_size=32):
        if len(self.history) < batch_size:
            return
        batch = random.sample(self.history, batch_size)
        user_ids, features, rewards = zip(*batch)
        features = np.array(features)
        rewards = np.array(rewards)

        features_tensor = torch.tensor(
            features, dtype=torch.float32
        )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(
            1
        )

        phi_x, y_pred = self.network(features_tensor)

        loss = self.loss_fn(y_pred, rewards_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.history = []
