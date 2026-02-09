# TP n°3 — Parallélisation du Bucket Sort 

from mpi4py import MPI
import numpy as np


def compute_bounds(global_min: float, global_max: float, p: int) -> np.ndarray:
    #Découpe [global_min, global_max] en p intervalles réguliers -> bornes de taille p+1.
    if p <= 0:
        raise ValueError("p doit être >= 1")
    if global_max == global_min:
        return np.array([global_min] * (p + 1), dtype=np.float64)
    step = (global_max - global_min) / p
    return np.array([global_min + i * step for i in range(p + 1)], dtype=np.float64)


def bucket_index(x: float, bounds: np.ndarray) -> int:
    """Renvoie l'indice i tel que bounds[i] <= x < bounds[i+1], et gère le cas x == max."""
    p = len(bounds) - 1
    # np.searchsorted renvoie la position d'insertion; side='right' gère bien les égalités
    i = int(np.searchsorted(bounds[1:-1], x, side="right"))  # 0..p-1
    if i < 0:
        return 0
    if i >= p:
        return p - 1
    return i


def distributed_bucket_sort(local: np.ndarray, comm: MPI.Comm) -> np.ndarray:
    #Bucket sort distribué : partition -> Alltoallv -> tri local final.
    rank = comm.Get_rank()
    p = comm.Get_size()

    local = np.asarray(local, dtype=np.float64)

    # min/max globaux (gère les rangs vides)
    local_min = float(np.min(local)) if local.size else float("inf")
    local_max = float(np.max(local)) if local.size else float("-inf")
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)

    bounds = compute_bounds(global_min, global_max, p)

    # Répartition locale vers p buckets (destinations)
    send_lists = [[] for _ in range(p)]
    for x in local:
        dest = bucket_index(float(x), bounds)
        send_lists[dest].append(float(x))

    # Convertir en buffers contigus (Alltoallv nécessite counts/displs)
    send_counts = np.array([len(lst) for lst in send_lists], dtype=np.int32)
    send_displs = np.zeros(p, dtype=np.int32)
    send_displs[1:] = np.cumsum(send_counts[:-1])

    if send_counts.sum() > 0:
        send_buf = np.array([v for lst in send_lists for v in lst], dtype=np.float64)
    else:
        send_buf = np.empty(0, dtype=np.float64)

    # Échanger les tailles, puis les données
    recv_counts = np.empty(p, dtype=np.int32)
    comm.Alltoall([send_counts, MPI.INT], [recv_counts, MPI.INT])

    recv_displs = np.zeros(p, dtype=np.int32)
    recv_displs[1:] = np.cumsum(recv_counts[:-1])
    total_recv = int(recv_counts.sum())
    recv_buf = np.empty(total_recv, dtype=np.float64)

    comm.Alltoallv(
        [send_buf, send_counts, send_displs, MPI.DOUBLE],
        [recv_buf, recv_counts, recv_displs, MPI.DOUBLE],
    )

    # Tri local : chaque rang possède maintenant sa tranche globale
    recv_buf.sort()
    return recv_buf


def chunkify(arr: np.ndarray, p: int):
    #Découpe un tableau 1D en p morceaux (presque égaux).
    n = arr.size
    chunks = []
    for r in range(p):
        start = (r * n) // p
        end = ((r + 1) * n) // p
        chunks.append(arr[start:end])
    return chunks


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()

    # --- 1) Le rang 0 génère les données ---
    N = 8 
    if rank == 0:
        rng = np.random.default_rng(12345)
        data = rng.random(N) * 1000.0  # valeurs arbitraires
        chunks = chunkify(data, p)
        print("Tableau initial :", data)

    else:
        chunks = None

    # --- 2) Dispatch (scatter) ---
    local = comm.scatter(chunks, root=0)

    # --- 3) Tri parallèle distribué ---
    t0 = MPI.Wtime()
    local_sorted = distributed_bucket_sort(local, comm)
    t1 = MPI.Wtime()
    local_time = t1 - t0
    max_time = comm.reduce(local_time, op=MPI.MAX, root=0)

    # --- 4) Rassemblement sur le rang 0 ---
    gathered = comm.gather(local_sorted, root=0)

    if rank == 0:
        final = np.concatenate(gathered) if gathered else np.empty(0, dtype=np.float64)
        ok = np.all(final[:-1] <= final[1:]) if final.size > 1 else True
        print(f"[rank 0] N={N}, p={p}, temps (max) = {max_time:.6f}s, tri OK ? {ok}")
        print("Tableau trié final :", final)
        


if __name__ == "__main__":
    main()
