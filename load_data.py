from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import torchvision
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from collections import defaultdict


class load_movielens_dif_user:
    def __init__(self, n_users):
        # Fetch data
        self.m = np.load("./movie_2000users_10000items_entry.npy")
        self.U = np.load("./movie_2000users_10000items_features.npy")
        self.I = np.load("./movie_10000items_2000users_features.npy")
        kmeans = KMeans(n_clusters=n_users, random_state=0).fit(self.U)
        self.groups = kmeans.labels_
        self.n_arm = 10
        self.dim = 20
        self.num_user = n_users
        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in self.m:
            if i[2] == 1:
                self.pos_index[self.groups[i[0]]].append((i[0], i[1]))
            else:
                self.neg_index[self.groups[i[0]]].append((i[0], i[1]))

    def step(self):
        u = np.random.choice(range(self.num_user))
        g = self.groups[u]
        arm = np.random.choice(range(10))
        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])
        pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), 9, replace=True)]
        neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), 1, replace=True)]
        X_ind = np.concatenate((pos[:arm], neg, pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            X.append(np.concatenate((self.I[ind[1]], self.U[ind[0]]), axis=None))
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        contexts = norm.pdf(np.array(X), loc=0, scale=0.5)
        return g, contexts, rwd


class load_kuai_dif_user:
    def __init__(self, n_users):
        full_data_path = "./KuaiRec/processd/full_data.csv"
        self.m = pd.read_csv(full_data_path, header=0, sep="\t")
        self.m = self.m[["user_id", "video_id", "label"]].values

        user_feature_path = "./KuaiRec/processd/user_feature_online.csv"
        user_df = pd.read_csv(user_feature_path, header=0, sep="\t")
        self.U = user_df.drop("user_id", axis=1).values

        item_feature_path = "./KuaiRec/processd/item_feature_online.csv"
        item_df = pd.read_csv(item_feature_path, header=0, sep="\t")
        self.I = item_df.drop(["video_id", "period"], axis=1).values

        self.kmeans = KMeans(n_clusters=n_users, random_state=0).fit(self.U)
        self.groups = self.kmeans.labels_

        self.n_arm = 10
        self.dim = self.U.shape[1] + self.I.shape[1]
        self.num_user = n_users

        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        self._initialize_indices()

        self.t = 0

        for entry in self.m:
            user_id, video_id, label = entry
            group = self.groups[user_id]
            if label == 1:
                self.pos_index[group].append((user_id, video_id))
            else:
                self.neg_index[group].append((user_id, video_id))

    def _initialize_indices(self):
        self.pos_index.clear()
        self.neg_index.clear()
        for interaction in self.m:
            user_id, item_id, interaction_type = interaction
            group_id = self.groups[user_id]
            if interaction_type == 1:
                self.pos_index[group_id].append((user_id, item_id))
            else:
                self.neg_index[group_id].append((user_id, item_id))

    def perturb_features(self):
        noise_std = 0.1

        self.U += np.random.normal(loc=0.0, scale=noise_std, size=self.U.shape)
        self.I += np.random.normal(loc=0.0, scale=noise_std, size=self.I.shape)

        self.kmeans.fit(self.U)
        self.groups = self.kmeans.labels_

        self._initialize_indices()

    def step(self):
        self.t += 1
        g = np.random.choice(range(self.num_user))

        arm = np.random.choice(range(self.n_arm))

        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])

        if p_d >= 9:
            pos_indices = np.random.choice(p_d, 9, replace=False)
        else:
            pos_indices = np.random.choice(p_d, 9, replace=True)
        pos = np.array(self.pos_index[g])[pos_indices]

        if n_d >= 1:
            neg = np.array(self.neg_index[g])[np.random.choice(n_d, 1, replace=False)]
        else:
            neg = pos[:1]

        X_ind = np.insert(pos, arm, neg, axis=0)

        X = []
        for ind in X_ind:
            user_feat = self.U[ind[0]]
            item_feat = self.I[ind[1]]
            combined_feat = np.concatenate((item_feat, user_feat), axis=None)
            X.append(combined_feat)
        X = np.array(X)

        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1

        contexts = norm.pdf(X, loc=0, scale=0.5)

        return g, contexts, rwd


class load_yelp_dif_user:
    def __init__(self, n_users):
        full_data_path = "./yelp/processed_yelp/full_data.csv"
        self.m = pd.read_csv(full_data_path, sep="\t")

        user_feat_path = "./yelp/processed_yelp/user_feature_pca.csv"
        item_feat_path = "./yelp/processed_yelp/item_feature_pca.csv"

        user_df = pd.read_csv(user_feat_path, sep="\t")
        user_df = user_df.select_dtypes(include=[np.number])
        user_df = user_df.apply(pd.to_numeric, errors="coerce").fillna(
            0
        )
        user_df = user_df.dropna(axis=1, how="all") 
        self.U = user_df.values

        item_df = pd.read_csv(item_feat_path, sep="\t")
        item_df = item_df.select_dtypes(include=[np.number])
        item_df = item_df.apply(pd.to_numeric, errors="coerce").fillna(
            0
        ) 
        item_df = item_df.dropna(axis=1, how="all")
        self.I = item_df.values

        self.kmeans = KMeans(n_clusters=n_users, random_state=0).fit(self.U)
        self.groups = self.kmeans.labels_

        self.n_arm = 10
        self.dim = self.U.shape[1] + self.I.shape[1]
        self.num_user = n_users

        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        self._initialize_indices()

        self.t = 0

    def _initialize_indices(self):
        self.pos_index.clear()
        self.neg_index.clear()

        for _, row in self.m.iterrows():
            try:
                user_id, item_id, label = (
                    int(row["user_id"]),
                    int(row["item_id"]),
                    int(row["label"]),
                )
                group_id = self.groups[user_id]
                if label == 1:
                    self.pos_index[group_id].append((user_id, item_id))
                else:
                    self.neg_index[group_id].append((user_id, item_id))
            except ValueError:
                continue 

    def perturb_features(self):
        noise_std = 1.0
        self.U += np.random.normal(loc=0.0, scale=noise_std, size=self.U.shape)
        self.I += np.random.normal(loc=0.0, scale=noise_std, size=self.I.shape)

        self.kmeans.fit(self.U)
        self.groups = self.kmeans.labels_

        self._initialize_indices()

    def step(self):
        self.t += 1

        g = np.random.choice(range(self.num_user))
        arm = np.random.choice(range(self.n_arm))

        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])

        if p_d >= 9:
            pos_indices = np.random.choice(p_d, 9, replace=False)
        else:
            pos_indices = np.random.choice(p_d, 9, replace=True)
        pos = np.array(self.pos_index[g])[pos_indices]

        if n_d >= 1:
            neg_idx = np.random.choice(n_d, 1, replace=False)
            neg = np.array(self.neg_index[g])[neg_idx]
        else:
            neg = pos[:1]

        X_ind = np.insert(pos, arm, neg, axis=0)

        X = []
        for u_id, i_id in X_ind:
            try:
                user_feat = self.U[u_id].astype(float)
                item_feat = self.I[i_id].astype(float)
                combined_feat = np.concatenate((item_feat, user_feat), axis=None)
                X.append(combined_feat)
            except ValueError:
                continue
        X = np.array(X)

        contexts = norm.pdf(X, loc=0, scale=0.5)

        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1

        return g, contexts, rwd


class load_dm_dif_user:
    def __init__(self, n_users):
        full_data_path = "./Amazon/processd_dm/full_data.csv"
        self.m = pd.read_csv(full_data_path, sep="\t")
        self.m = self.m[["user_id", "item_id", "label"]].values

        user_feat_path = "./Amazon/processd_dm/user_feature_pca.csv"
        user_df = pd.read_csv(user_feat_path, sep="\t")
        self.U = user_df.drop("user_id", axis=1).values

        item_feat_path = "./Amazon/processd_dm/item_feature_pca.csv"
        item_df = pd.read_csv(item_feat_path, sep="\t")
        self.I = item_df.drop("item_id", axis=1).values

        self.kmeans = KMeans(n_clusters=n_users, random_state=0).fit(self.U)
        self.groups = self.kmeans.labels_

        self.n_arm = 10
        self.dim = self.U.shape[1] + self.I.shape[1]
        self.num_user = n_users

        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        self._initialize_indices()

        self.t = 0

    def _initialize_indices(self):
        self.pos_index.clear()
        self.neg_index.clear()

        for interaction in self.m:
            user_id, item_id, label = interaction
            group_id = self.groups[user_id]
            if label == 1:
                self.pos_index[group_id].append((user_id, item_id))
            else:
                self.neg_index[group_id].append((user_id, item_id))

    def perturb_features(self):
        noise_std = 1.0
        self.U += np.random.normal(loc=0.0, scale=noise_std, size=self.U.shape)
        self.I += np.random.normal(loc=0.0, scale=noise_std, size=self.I.shape)

        self.kmeans.fit(self.U)
        self.groups = self.kmeans.labels_

        self._initialize_indices()

    def step(self):
        self.t += 1
        g = np.random.choice(range(self.num_user))

        arm = np.random.choice(range(self.n_arm))

        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])

        if p_d >= 9:
            pos_indices = np.random.choice(p_d, 9, replace=False)
        else:
            pos_indices = np.random.choice(p_d, 9, replace=True)
        pos = np.array(self.pos_index[g])[pos_indices]

        if n_d >= 1:
            neg_index = np.random.choice(n_d, 1, replace=False)
            neg = np.array(self.neg_index[g])[neg_index]
        else:
            neg = pos[:1]

        X_ind = np.insert(pos, arm, neg, axis=0)

        X = []
        for u_id, i_id in X_ind:
            user_feat = self.U[u_id]
            item_feat = self.I[i_id]
            combined_feat = np.concatenate((item_feat, user_feat), axis=None)
            X.append(combined_feat)
        X = np.array(X)

        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1

        contexts = norm.pdf(X, loc=0, scale=0.5)

        return g, contexts, rwd


class load_vg_dif_user:
    def __init__(self, n_users):
        full_data_path = "./Amazon/processd_vg/full_data.csv"
        self.m = pd.read_csv(full_data_path, sep="\t")
        self.m = self.m[["user_id", "item_id", "label"]].values

        user_feat_path = "./Amazon/processd_vg/user_feature_pca.csv"
        user_df = pd.read_csv(user_feat_path, sep="\t")
        self.U = user_df.drop("user_id", axis=1).values

        item_feat_path = "./Amazon/processd_vg/item_feature_pca.csv"
        item_df = pd.read_csv(item_feat_path, sep="\t")
        self.I = item_df.drop("item_id", axis=1).values

        self.kmeans = KMeans(n_clusters=n_users, random_state=0).fit(self.U)
        self.groups = self.kmeans.labels_

        self.n_arm = 10
        self.dim = self.U.shape[1] + self.I.shape[1]
        self.num_user = n_users

        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        self._initialize_indices()

        self.t = 0

    def _initialize_indices(self):
        self.pos_index.clear()
        self.neg_index.clear()

        for interaction in self.m:
            user_id, item_id, label = interaction
            group_id = self.groups[user_id]
            if label == 1:
                self.pos_index[group_id].append((user_id, item_id))
            else:
                self.neg_index[group_id].append((user_id, item_id))

    def perturb_features(self):
        noise_std = 1.0
        self.U += np.random.normal(loc=0.0, scale=noise_std, size=self.U.shape)
        self.I += np.random.normal(loc=0.0, scale=noise_std, size=self.I.shape)

        self.kmeans.fit(self.U)
        self.groups = self.kmeans.labels_

        self._initialize_indices()

    def step(self):
        self.t += 1
        g = np.random.choice(range(self.num_user))

        arm = np.random.choice(range(self.n_arm))

        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])

        if p_d >= 9:
            pos_indices = np.random.choice(p_d, 9, replace=False)
        else:
            pos_indices = np.random.choice(p_d, 9, replace=True)
        pos = np.array(self.pos_index[g])[pos_indices]

        if n_d >= 1:
            neg_index = np.random.choice(n_d, 1, replace=False)
            neg = np.array(self.neg_index[g])[neg_index]
        else:
            neg = pos[:1]

        X_ind = np.insert(pos, arm, neg, axis=0)

        X = []
        for u_id, i_id in X_ind:
            user_feat = self.U[u_id]
            item_feat = self.I[i_id]
            combined_feat = np.concatenate((item_feat, user_feat), axis=None)
            X.append(combined_feat)
        X = np.array(X)

        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1

        contexts = norm.pdf(X, loc=0, scale=0.5)

        return g, contexts, rwd


class load_facebook_dif_user:
    def __init__(self, n_users):
        full_data_path = "./facebook/processd_facebook/full_data.csv"
        self.m = pd.read_csv(full_data_path, sep="\t")
        self.m = self.m[["user_id", "item_id", "label"]].values

        node_feat_path = "./facebook/processd_facebook/node_feature_pca.csv"
        node_df = pd.read_csv(node_feat_path, sep="\t")
        self.N = node_df.drop("node_id", axis=1).values

        self.U = self.N
        self.I = self.N

        self.kmeans = KMeans(n_clusters=n_users, random_state=0).fit(self.U)
        self.groups = self.kmeans.labels_

        self.n_arm = 10
        self.dim = self.U.shape[1] + self.I.shape[1]
        self.num_user = n_users

        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        self._initialize_indices()

        self.t = 0

    def _initialize_indices(self):
        self.pos_index.clear()
        self.neg_index.clear()

        for interaction in self.m:
            user_id, item_id, label = interaction
            group_id = self.groups[user_id]
            if label == 1:
                self.pos_index[group_id].append((user_id, item_id))
            else:
                self.neg_index[group_id].append((user_id, item_id))

    def perturb_features(self):
        noise_std = 1.0
        self.U += np.random.normal(loc=0.0, scale=noise_std, size=self.U.shape)
        self.I += np.random.normal(loc=0.0, scale=noise_std, size=self.I.shape)
        
        self.kmeans.fit(self.U)
        self.groups = self.kmeans.labels_

        self._initialize_indices()

    def step(self):
        self.t += 1
        g = np.random.choice(range(self.num_user))

        arm = np.random.choice(range(self.n_arm))

        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])

        if p_d >= 9:
            pos_indices = np.random.choice(p_d, 9, replace=False)
        else:
            pos_indices = np.random.choice(p_d, 9, replace=True)
        pos = np.array(self.pos_index[g])[pos_indices]

        if n_d >= 1:
            neg_idx = np.random.choice(n_d, 1, replace=False)
            neg = np.array(self.neg_index[g])[neg_idx]
        else:
            neg = pos[:1]

        X_ind = np.insert(pos, arm, neg, axis=0)

        X = []
        for u_id, i_id in X_ind:
            user_feat = self.U[u_id]
            item_feat = self.I[i_id]
            combined_feat = np.concatenate((item_feat, user_feat), axis=0)
            X.append(combined_feat)
        X = np.array(X)

        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        
        contexts = norm.pdf(X, loc=0, scale=0.5)

        return g, contexts, rwd
