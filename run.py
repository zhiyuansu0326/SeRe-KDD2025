from club_n import CLUB_N
from locb_n import LOCB_N
from sclub_n import SCLUB_N
from mcnb import MCNB
from club_n_sere import CLUB_N_SERE
from locb_n_sere import LOCB_N_SERE
from sclub_n_sere import SCLUB_N_SERE
from mcnb_sere import MCNB_SERE
import argparse
import numpy as np
import sys
import os
import time
from load_data import load_movielens_dif_user

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="movie", type=str)
    parser.add_argument("--method", default="mcnb_sere", type=str)
    parser.add_argument("--n_runs", default=1, type=int)
    parser.add_argument("--n_rounds", default=10000, type=int)
    args = parser.parse_args()
    dataset = args.dataset.lower()
    method = args.method.lower()
    n_runs = args.n_runs
    n_rounds = args.n_rounds
    for run_i in range(n_runs):
        start_time = time.time()
        print(f"\n=== Start Run {run_i+1}/{n_runs} ===")
        if dataset == "movielens":
            b = load_movielens_dif_user(100)
        else:
            print("Dataset not supported.")
            sys.exit()
        if method == "club_n":
            model = CLUB_N(nu=b.num_user, d=b.dim)
        elif method == "locb_n":
            model = LOCB_N(
                nu=b.num_user,
                d=b.dim,
                gamma=0.2,
                num_seeds=20,
                delta=0.1,
                detect_cluster=0,
            )
        elif method == "sclub_n":
            model = SCLUB_N(nu=b.num_user, d=b.dim)
        elif method == "mcnb":
            model = MCNB(
                dim=b.dim, n=b.num_user, n_arm=10, gamma=0.4, lr=0.001, nu=1e-4
            )
        elif method == "club_n_sere":
            model = CLUB_N_SERE(nu=b.num_user, d=b.dim)
        elif method == "locb_n_sere":
            model = LOCB_N_SERE(
                nu=b.num_user,
                d=b.dim,
                gamma=0.2,
                num_seeds=20,
                delta=0.1,
                detect_cluster=0,
            )
        elif method == "sclub_n_sere":
            model = SCLUB_N_SERE(nu=b.num_user, d=b.dim)
        elif method == "mcnb_sere":
            model = MCNB_SERE(
                dim=b.dim, n=b.num_user, n_arm=10, gamma=0.4, lr=0.001, nu=1e-4
            )
        else:
            print("Method not defined.")
            sys.exit()
        print("Dataset: movie, Method:", method)
        regrets = []
        total_regret = 0
        print("Round;    Regret;    Regret/Round")
        for t in range(n_rounds):
            u, context, rwd = b.step()
            if method in ["mcnb", "mcnb_sere"]:
                arm_select, g, ucb = model.select(u, context, t)
            else:
                arm_select = model.recommend(u, context, t)
            r = rwd[arm_select]
            reg = np.max(rwd) - r
            total_regret += reg
            regrets.append(total_regret)
            if method in ["club_n", "locb_n", "club_n_sere", "locb_n_sere"]:
                model.store_info(i=u, x=context[arm_select], y=r, t=t)
                model.update(i=u, t=t)
            elif method in ["sclub_n", "sclub_n_sere"]:
                model.store_info(i=u, x=context[arm_select], y=r, t=t, r=r, br=1.0)
                model.split(u, t)
                model.merge(t)
                model.num_clusters[t] = len(model.clusters)
            elif method in ["mcnb", "mcnb_sere"]:
                model.update(u, context[arm_select], r, g)
                if t < 1000:
                    if t % 10 == 0:
                        model.train_meta(model.users, t, 1000, model.lr)
                        _ = model.train(u, t)
                else:
                    if t % 100 == 0:
                        model.train_meta(model.users, t, 1000, model.lr)
                        _ = model.train(u, t)
            if t % 50 == 0:
                print(f"{t}: {total_regret}, {total_regret/(t+1):.4f}")
        print("Final round:", t, "; total regret:", total_regret)
        end_time = time.time()
        print(f"Run {run_i+1} completed in {end_time - start_time:.2f} seconds.")
