import time
import sys
import numpy as np
from nbodies_grid_numba import NBodySystem

def bench(filename, dt=0.0015, ncells=(15, 15, 1), nsteps=20):
    system = NBodySystem(filename, ncells_per_dir=ncells)

    # warm-up numba
    system.update_positions(dt)

    times = []
    for _ in range(nsteps):
        t0 = time.perf_counter()
        system.update_positions(dt)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    arr = np.array(times)
    print(f"Fichier              : {filename}")
    print(f"Pas de temps         : {dt}")
    print(f"Grille               : {ncells}")
    print(f"Nombre d'iterations  : {nsteps}")
    print(f"Temps moyen calcul   : {arr.mean():.4f} ms")
    print(f"Temps min calcul     : {arr.min():.4f} ms")
    print(f"Temps max calcul     : {arr.max():.4f} ms")

if __name__ == "__main__":
    filename = "../data/galaxy_1000"
    dt = 0.0015
    ncells = (15, 15, 1)
    nsteps = 20

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])
    if len(sys.argv) > 5:
        ncells = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    if len(sys.argv) > 6:
        nsteps = int(sys.argv[6])

    bench(filename, dt, ncells, nsteps)