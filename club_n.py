import networkx as nx
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

    def store_info(self, i, x, y, t):
        self.S[i] += np.outer(x, x)
        self.b[i] += y * x
        self.N[i] += 1
        self.Sinv[i], self.theta[i] = self._update_inverse(
            self.S[i], self.b[i], self.Sinv[i], x, self.N[i]
        )


class Cluster:
    def __init__(self, users, S, b, N):
        self.users = users
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

    def forward(self, x):
        h = self.activate(self.fc1(x))
        phi_x = self.fc_phi(h)
        y_pred = self.fc_y(h)
        return phi_x, y_pred


class CLUB_N(LinUCB_IND):
    def __init__(
        self, nu, d, T=10000, hidden_size=100, learning_rate=1e-3, update_interval=100
    ):
        super(CLUB_N, self).__init__(nu, d)
        self.nu = nu
        self.G = nx.complete_graph(nu)
        self.clusters = {0: Cluster(users=range(nu), S=np.eye(d), b=np.zeros(d), N=0)}
        self.cluster_inds = np.zeros(nu, dtype=int)
        self.num_clusters = np.zeros(T)

        self.input_dim = d
        self.hidden_size = hidden_size
        self.output_dim = d
        self.network = Network_u(self.input_dim, hidden_size=self.hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_interval = update_interval
        self.history = []
        self.current_step = 0

    def recommend(self, i, items, t):
        cluster = self.clusters[self.cluster_inds[i]]
        return self._select_item_ucb(
            cluster.S, cluster.Sinv, cluster.theta, items, cluster.N, t
        )

    def store_info(self, i, x, y, t):
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

    def _if_split(self, theta, N1, N2):
        alpha = 1

        def _factT(T):
            return np.sqrt((1 + np.log(1 + T)) / (1 + T))

        return np.linalg.norm(theta) > alpha * (_factT(N1) + _factT(N2))

    def update(self, i, t):
        update_clusters = False
        c = self.cluster_inds[i]

        A = [a for a in self.G.neighbors(i)]
        for j in A:
            if (
                self.N[i]
                and self.N[j]
                and self._if_split(self.theta[i] - self.theta[j], self.N[i], self.N[j])
            ):
                self.G.remove_edge(i, j)
                update_clusters = True

        if update_clusters:
            C = set(nx.node_connected_component(self.G, i))
            if len(C) < len(self.clusters[c].users):
                remain_users = set(self.clusters[c].users)
                self.clusters[c] = Cluster(
                    users=list(C),
                    S=sum([self.S[k] - np.eye(self.d) for k in C]) + np.eye(self.d),
                    b=sum([self.b[k] for k in C]),
                    N=sum([self.N[k] for k in C]),
                )

                remain_users = remain_users - set(C)
                c = max(self.clusters) + 1
                while len(remain_users) > 0:
                    j = np.random.choice(list(remain_users))
                    C = nx.node_connected_component(self.G, j)

                    self.clusters[c] = Cluster(
                        users=list(C),
                        S=sum([self.S[k] - np.eye(self.d) for k in C]) + np.eye(self.d),
                        b=sum([self.b[k] for k in C]),
                        N=sum([self.N[k] for k in C]),
                    )
                    for user in C:
                        self.cluster_inds[user] = c

                    c += 1
                    remain_users = remain_users - set(C)

        self.num_clusters[t] = len(self.clusters)

    def train_network(self, batch_size=32):
        if len(self.history) < batch_size:
            return
        batch = random.sample(self.history, batch_size)
        user_ids, features, rewards = zip(*batch)
        features = np.array(features)
        rewards = np.array(rewards)

        features_tensor = torch.tensor(features, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

        phi_x, y_pred = self.network(features_tensor)
        loss = self.loss_fn(y_pred, rewards_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.history = []
