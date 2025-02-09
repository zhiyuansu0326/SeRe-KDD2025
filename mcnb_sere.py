from packages import *

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class Network_u(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_u, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.utils = {
            1: torch.zeros(self.fc1.out_features).to(device),
            2: torch.zeros(self.fc2.out_features).to(device),
        }
        self.ages = {
            1: torch.zeros(self.fc1.out_features).to(device),
            2: torch.zeros(self.fc2.out_features).to(device),
        }
        self.counters = {1: 0.0, 2: 0.0}
        self.last_x = None
        self.last_h1 = None

    def forward(self, x):
        self.last_x = x
        h1 = self.activate(self.fc1(x))
        self.last_h1 = h1
        out = self.fc2(h1)
        return out

    def update_utilities(self, eta=0.9):
        if self.last_h1 is None or self.last_x is None:
            return
        h1 = self.last_h1.detach()
        if len(h1.shape) > 1:
            h1 = h1.mean(dim=0)
        w1 = self.fc1.weight.detach()
        w1_contribution = torch.norm(w1, dim=1)
        assert h1.shape[0] == self.hidden_size
        assert w1_contribution.shape[0] == self.hidden_size
        new_utils1 = h1 * w1_contribution
        self.utils[1] = eta * self.utils[1] + (1 - eta) * new_utils1
        w2 = self.fc2.weight.detach()
        w2_contribution = torch.norm(w2, dim=1)
        h2 = self.fc2(self.last_h1).detach().mean()
        new_utils2 = h2 * w2_contribution
        self.utils[2] = eta * self.utils[2] + (1 - eta) * new_utils2

    def increment_ages(self):
        for layer in [1, 2]:
            self.ages[layer] += 1


class MCNB_SERE:
    def __init__(self, dim, n, n_arm, gamma, lr=0.01, hidden=100, nu=0.001):
        self.lr = lr
        self.dim = dim
        self.hidden = hidden
        self.t = 0
        self.gamma = gamma
        self.nu = nu
        self.g = []
        self.n = n
        self.u_count = [0] * n
        self.users = range(n)
        self.u_fun = {}
        self.meta_fun = Network_u(self.dim, hidden_size=self.hidden).to(device)
        for i in range(n):
            self.u_fun[i] = Network_u(dim, hidden_size=hidden).to(device)
        self.contexts = defaultdict(list)
        self.rewards = defaultdict(list)
        self.rho_min = 0.01
        self.rho_max = 0.1
        self.rho = self.rho_min
        self.maturity_threshold = 100
        self.PH = 0
        self.PH_min = 0
        self.PH_threshold = 0.5
        self.PH_delta = 0.1
        self.alpha = 0.01

    def update_PH(self, reward, pred_reward):
        self.PH += reward - pred_reward - self.PH_delta
        self.PH_min = min(self.PH_min, self.PH)
        if self.PH - self.PH_min > self.PH_threshold:
            self.rho = self.rho_max
        else:
            self.rho = self.rho_min + self.alpha * (self.PH - self.PH_min)

    def apply_sere(self, network):
        for layer in [1, 2]:
            expected_size = (
                network.fc1.out_features if layer == 1 else network.fc2.in_features
            )
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
                    network.fc2.weight.data[:, min_util_idx] = 0.0
                network.utils[layer][min_util_idx] = 0
                network.ages[layer][min_util_idx] = 0
                network.counters[layer] -= 1

    def update(self, u, context, reward, g):
        context_tensor = torch.from_numpy(context.reshape(1, -1)).float().to(device)
        self.contexts[u].append(context_tensor)
        self.rewards[u].append(reward)
        with torch.no_grad():
            pred_reward = self.u_fun[u](context_tensor).item()
        self.update_PH(reward, pred_reward)
        _ = self.u_fun[u](context_tensor)
        self.u_fun[u].update_utilities()
        self.u_fun[u].increment_ages()
        self.apply_sere(self.u_fun[u])

    def train_meta(self, g, t, train_limit, meta_lr, y=1):
        if y == 0:
            optimizer = optim.SGD(self.meta_fun.parameters(), lr=meta_lr)
        else:
            optimizer = optim.Adam(self.meta_fun.parameters(), lr=meta_lr)
        index = []
        for u in g:
            for j in range(len(self.rewards[u])):
                index.append((u, j))
        length = len(index)
        cnt = 0
        if length > 0:
            tot_loss = 0
            while True:
                batch_loss = 0
                np.random.shuffle(index)
                for idx in index:
                    u = idx[0]
                    arm = idx[1]
                    c = self.contexts[u][arm]
                    r = self.rewards[u][arm]
                    optimizer.zero_grad()
                    loss = (self.meta_fun(c.to(device)) - r) ** 2
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                    tot_loss += loss.item()
                    cnt += 1
                    if cnt >= train_limit:
                        return tot_loss / cnt
                if batch_loss / length <= 1e-3:
                    return batch_loss / length

    def get_group(self, u, context):
        u_pred = self.u_fun[u](context)
        g = set([u])
        for i in self.users:
            diff = abs(self.u_fun[i](context) - u_pred)
            if diff < self.gamma:
                g.add(i)
        g_limit = int(self.n / 5)
        if len(g) > g_limit:
            g = set(np.random.choice(list(g), g_limit))
            g.add(u)
            return g
        else:
            return g

    def select(self, u, context, t):
        self.t = t
        ucb_list = []
        self.u_count[u] += 1
        for c in context:
            c_tensor = torch.from_numpy(c).float().to(device)
            if t % 10 == 0:
                self.g = self.get_group(u, c_tensor)
                self.train_meta(self.g, t, 100, 1e-2, 0)
            res = self.meta_fun(c_tensor)
            self.meta_fun.zero_grad()
            res.backward()
            gra = torch.cat(
                [p.grad.flatten().detach() for p in self.meta_fun.parameters()]
            )
            sigma = torch.sum(self.nu * gra * gra).item()
            ucb = res.item() + np.sqrt(sigma) + np.sqrt(1 / self.u_count[u]) * self.nu
            ucb_list.append(ucb)
        arm = np.argmax(ucb_list)
        return arm, self.g, ucb_list

    def train(self, u, t):
        optimizer = optim.Adam(self.u_fun[u].parameters(), lr=self.lr)
        length = len(self.rewards[u])
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        if length > 0:
            while True:
                batch_loss = 0
                for idx in index:
                    c = self.contexts[u][idx]
                    r = self.rewards[u][idx]
                    optimizer.zero_grad()
                    loss = (self.u_fun[u](c.to(device)) - r) ** 2
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                    tot_loss += loss.item()
                    cnt += 1
                    if cnt >= 500:
                        return tot_loss / cnt
                if batch_loss / length <= 1e-3:
                    return batch_loss / length
        else:
            return 0.0
