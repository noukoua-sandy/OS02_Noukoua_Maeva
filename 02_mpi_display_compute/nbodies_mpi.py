from mpi4py import MPI
import numpy as np
import visualizer3d
import sys
import time

from nbodies_grid_numba import NBodySystem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

system = None

def update_positions_mpi(dt):
    global system

    if rank == 1:
        system.update_positions(dt)
        comm.send(system.positions, dest=0)
    elif rank == 0:
        positions = comm.recv(source=1)
        system.positions = positions

    return system.positions

def run_simulation_mpi(filename, dt, ncells, bench_steps=None):
    global system

    system = NBodySystem(filename, ncells_per_dir=ncells)

    if rank == 0:
        pos = system.positions
        col = system.colors
        intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)

        visu = visualizer3d.Visualizer3D(
            pos, col, intensity,
            [[system.box[0][0], system.box[1][0]],
             [system.box[0][1], system.box[1][1]],
             [system.box[0][2], system.box[1][2]]]
        )

        visu.run(updater=update_positions_mpi, dt=dt)

    else:
        if bench_steps is None:
            while True:
                t0 = time.perf_counter()
                system.update_positions(dt)
                t1 = time.perf_counter()

                comm.send(system.positions, dest=0)
                print(f"[Rank 1] Temps calcul : {(t1 - t0)*1000:.4f} ms")

        else:
            times = []
            for _ in range(bench_steps):
                t0 = time.perf_counter()
                system.update_positions(dt)
                t1 = time.perf_counter()

                times.append((t1 - t0) * 1000.0)
                comm.send(system.positions, dest=0)

            arr = np.array(times)
            print(f"[Rank 1] Temps moyen calcul : {arr.mean():.4f} ms")
            print(f"[Rank 1] Temps min : {arr.min():.4f} ms")
            print(f"[Rank 1] Temps max : {arr.max():.4f} ms")
            raise SystemExit(0)

if __name__ == "__main__":
    filename = "../data/galaxy_1000"
    dt = 0.0015
    ncells = (15, 15, 1)
    bench_steps = None

    if "--bench" in sys.argv:
        idx = sys.argv.index("--bench")
        bench_steps = int(sys.argv[idx + 1])
        del sys.argv[idx:idx + 2]

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])
    if len(sys.argv) > 5:
        ncells = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))

    run_simulation_mpi(filename, dt, ncells, bench_steps)