import subprocess
import sys
import time
import re

def bench_render(filename, dt=0.0015, ncells=(15, 15, 1), nsteps=20):
    cmd = [
        sys.executable,
        "nbodies_grid_numba.py",
        filename,
        str(dt),
        str(ncells[0]),
        str(ncells[1]),
        str(ncells[2]),
        "--bench",
        str(nsteps),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()

    stdout = proc.stdout
    stderr = proc.stderr

    total_ms_per_iter = ((t1 - t0) * 1000.0) / nsteps

    m_mean = re.search(r"compute_mean_ms=([0-9.]+)", stdout)
    m_min = re.search(r"compute_min_ms=([0-9.]+)", stdout)
    m_max = re.search(r"compute_max_ms=([0-9.]+)", stdout)

    if m_mean is None:
        print(stdout)
        print(stderr)
        raise RuntimeError("Impossible de récupérer les mesures BENCH_RENDER")

    compute_mean = float(m_mean.group(1))
    compute_min = float(m_min.group(1))
    compute_max = float(m_max.group(1))
    render_est = total_ms_per_iter - compute_mean

    print(f"Fichier                 : {filename}")
    print(f"Pas de temps            : {dt}")
    print(f"Grille                  : {ncells}")
    print(f"Nombre d'iterations     : {nsteps}")
    print(f"Temps total/iteration   : {total_ms_per_iter:.4f} ms")
    print(f"Temps calcul/iteration  : {compute_mean:.4f} ms")
    print(f"Temps affichage estime  : {render_est:.4f} ms")
    print(f"Calcul min              : {compute_min:.4f} ms")
    print(f"Calcul max              : {compute_max:.4f} ms")

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

    bench_render(filename, dt, ncells, nsteps)