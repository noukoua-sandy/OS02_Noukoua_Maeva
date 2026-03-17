import sys
import time
import numpy as np
from mpi4py import MPI
from numba import njit, prange

from nbodies_grid_numba import NBodySystem, compute_acceleration

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@njit(parallel=True)
def compute_acceleration_subset(positions, masses,
                                cell_start_indices, body_indices,
                                cell_masses, cell_com_positions,
                                grid_min, grid_max, cell_size, n_cells,
                                start_idx, end_idx):
    n_local = end_idx - start_idx
    a = np.zeros((n_local, 3), dtype=np.float32)

    for iloc in prange(n_local):
        ibody = start_idx + iloc
        pos = positions[ibody]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)

        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0

        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]

                    # cellules lointaines : approximation par masse + centre de masse
                    if (abs(ix - cell_idx[0]) > 2) or (abs(iy - cell_idx[1]) > 2) or (abs(iz - cell_idx[2]) > 2):
                        cell_com = cell_com_positions[morse_idx]
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com - pos
                            distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                            if distance > 1.0e-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                a[iloc, :] += 1.560339e-13 * direction[:] * inv_dist3 * cell_mass
                    else:
                        # cellules proches : calcul exact sur les corps
                        s0 = cell_start_indices[morse_idx]
                        s1 = cell_start_indices[morse_idx + 1]
                        for j in range(s0, s1):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                direction = positions[jbody] - pos
                                distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                                if distance > 1.0e-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    a[iloc, :] += 1.560339e-13 * direction[:] * inv_dist3 * masses[jbody]
    return a

def split_indices(n, p):
    counts = [n // p] * p
    for r in range(n % p):
        counts[r] += 1
    starts = [0]
    for c in counts[:-1]:
        starts.append(starts[-1] + c)
    ends = [starts[r] + counts[r] for r in range(p)]
    return counts, starts, ends

def allgatherv_float32_2d(local_arr, global_arr, counts_bodies):
    counts_flat = np.array([3 * c for c in counts_bodies], dtype=np.int32)
    displs_flat = np.zeros(len(counts_flat), dtype=np.int32)
    displs_flat[1:] = np.cumsum(counts_flat[:-1])

    comm.Allgatherv(
        [local_arr.ravel(), MPI.FLOAT],
        [global_arr.ravel(), counts_flat, displs_flat, MPI.FLOAT]
    )

def run(filename, dt=0.0015, ncells=(15, 15, 1), nsteps=10):
    system = NBodySystem(filename, ncells_per_dir=ncells)

    n_bodies = system.positions.shape[0]
    counts, starts, ends = split_indices(n_bodies, size)
    i0, i1 = starts[rank], ends[rank]
    n_local = i1 - i0

    # Toutes les masses restent identiques, donc pas besoin de les échanger à chaque pas.
    # Chaque processus charge le fichier et possède déjà masses / couleurs / box.
    # On synchronise seulement positions et vitesses.

    # Warm-up numba
    system.grid.update(system.positions, system.masses)
    _ = compute_acceleration_subset(
        system.positions, system.masses,
        system.grid.cell_start_indices, system.grid.body_indices,
        system.grid.cell_masses, system.grid.cell_com_positions,
        system.grid.min_bounds, system.grid.max_bounds,
        system.grid.cell_size, system.grid.n_cells,
        i0, i1
    )

    local_times = []

    for _ in range(nsteps):
        t0 = time.perf_counter()

        # Grille globale sur toutes les positions
        system.grid.update(system.positions, system.masses)

        # a(t) pour mes corps locaux
        a_local = compute_acceleration_subset(
            system.positions, system.masses,
            system.grid.cell_start_indices, system.grid.body_indices,
            system.grid.cell_masses, system.grid.cell_com_positions,
            system.grid.min_bounds, system.grid.max_bounds,
            system.grid.cell_size, system.grid.n_cells,
            i0, i1
        )

        # mise à jour locale des positions
        local_pos = system.positions[i0:i1].copy()
        local_vel = system.velocities[i0:i1].copy()
        local_pos += local_vel * dt + 0.5 * a_local * dt * dt

        # synchronise positions
        allgatherv_float32_2d(local_pos, system.positions, counts)

        # met à jour la grille avec positions(t+dt)
        system.grid.update(system.positions, system.masses)

        # a(t+dt) pour mes corps locaux
        a_new_local = compute_acceleration_subset(
            system.positions, system.masses,
            system.grid.cell_start_indices, system.grid.body_indices,
            system.grid.cell_masses, system.grid.cell_com_positions,
            system.grid.min_bounds, system.grid.max_bounds,
            system.grid.cell_size, system.grid.n_cells,
            i0, i1
        )

        # mise à jour locale des vitesses
        local_vel += 0.5 * (a_local + a_new_local) * dt

        # synchronise vitesses
        allgatherv_float32_2d(local_vel, system.velocities, counts)

        t1 = time.perf_counter()
        local_times.append((t1 - t0) * 1000.0)

    arr = np.array(local_times, dtype=np.float64)
    mean_local = arr.mean()
    min_local = arr.min()
    max_local = arr.max()

    # on prend le max des temps moyens locaux : c'est le temps effectif du pas parallèle
    mean_global = comm.reduce(mean_local, op=MPI.MAX, root=0)
    min_global = comm.reduce(min_local, op=MPI.MAX, root=0)
    max_global = comm.reduce(max_local, op=MPI.MAX, root=0)

    if rank == 0:
        print(f"Fichier              : {filename}")
        print(f"Pas de temps         : {dt}")
        print(f"Grille               : {ncells}")
        print(f"Nombre d'iterations  : {nsteps}")
        print(f"Processus MPI        : {size}")
        print(f"Temps moyen calcul   : {mean_global:.4f} ms")
        print(f"Temps min calcul     : {min_global:.4f} ms")
        print(f"Temps max calcul     : {max_global:.4f} ms")

if __name__ == "__main__":
    filename = "../data/galaxy_1000"
    dt = 0.0015
    ncells = (15, 15, 1)
    nsteps = 10

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])
    if len(sys.argv) > 5:
        ncells = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    if len(sys.argv) > 6:
        nsteps = int(sys.argv[6])

    run(filename, dt, ncells, nsteps)