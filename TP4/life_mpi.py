"""
life_mpi.py — Jeu de la vie (tore) parallèle avec MPI

Objectifs TP:
1) séparation affichage / calcul (rank 0 affiche, ranks 1.. calculent)
2) découpage 1D (par lignes) + cellules fantômes (ghost rows)
3) vectorisation (calcul des voisins sans boucles Python)
4) option asynchrone côté affichage (Iprobe) pour éviter le freeze

Exécution:
    mpirun -np 2 python life_mpi.py glider 800 800
    mpirun -np 5 python life_mpi.py glider 800 800

Dépendances:
    pip install mpi4py pygame numpy
"""

from __future__ import annotations
import sys
import time
import numpy as np
from mpi4py import MPI

# ---------- Patterns (dimension, liste de cellules vivantes) ----------
DICO_PATTERNS = {
    "blinker": ((5, 5), [(2, 1), (2, 2), (2, 3)]),
    "toad": ((6, 6), [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5)]),
    "acorn": ((100, 100), [(51, 52), (52, 54), (53, 51), (53, 52), (53, 55), (53, 56), (53, 57)]),
    "beacon": ((6, 6), [(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]),
    "boat": ((5, 5), [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]),
    "glider": ((100, 90), [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)]),
    "glider_gun": (
        (400, 400),
        [(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),
         (54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),
         (55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),
         (56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]
    ),
    "space_ship": ((25, 25), [(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
    "die_hard": ((100, 100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
    "pulsar": ((17, 17),
               [(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),
                (2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),
                (4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),
                (10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
    "floraison": ((40, 40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
    "u": ((200, 200),
          [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),
           (105,105),(103,105),(102,105),(101,105),(101,104)]),
    "flat": ((200, 400),
             [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200),
              (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),
              (106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),
              (114,200),(115,200),(116,200),(117,200),(118,200)])
}


# ---------- Affichage Pygame (rank 0 seulement) ----------
def pygame_app_loop(comm: MPI.Comm, ny: int, nx: int, resx: int, resy: int, async_display: bool) -> None:
    import pygame as pg

    pg.init()

    # Taille cellule
    size_x = max(1, resx // nx)
    size_y = max(1, resy // ny)
    width = nx * size_x
    height = ny * size_y

    screen = pg.display.set_mode((width, height))
    col_life = pg.Color("black")
    col_dead = pg.Color("white")
    draw_grid = (size_x > 4 and size_y > 4)
    grid_color = pg.Color("lightgrey")

    def draw(cells: np.ndarray) -> None:
        # cells shape (ny, nx), uint8
        
        for i in range(ny):
            y = height - size_y * (i + 1)
            row = cells[i]
            for j in range(nx):
                x = size_x * j
                screen.fill(col_life if row[j] else col_dead, (x, y, size_x, size_y))
        if draw_grid:
            for i in range(ny):
                pg.draw.line(screen, grid_color, (0, i * size_y), (width, i * size_y))
            for j in range(nx):
                pg.draw.line(screen, grid_color, (j * size_x, 0), (j * size_x, height))
        pg.display.update()

    # Réception des frames depuis les workers
    TAG_FRAME = 100
    TAG_STOP = 200

    last_frame = np.zeros((ny, nx), dtype=np.uint8)
    must_continue = True

    # On reçoit une première frame (bloquante) pour afficher quelque chose
    last_frame = comm.bcast(None, root=0)  

    # Attente d'une première frame en point-à-point (pour async)
    if async_display:
        # Attend une frame initiale (bloquant)
        status = MPI.Status()
        comm.probe(source=MPI.ANY_SOURCE, tag=TAG_FRAME, status=status)
        buf = np.empty(ny * nx, dtype=np.uint8)
        comm.Recv(buf, source=status.Get_source(), tag=TAG_FRAME)
        last_frame = buf.reshape(ny, nx).copy()
        draw(last_frame)
    else:
        # En mode synchrone, on recevra via bcast/gather orchestrés ailleurs 
        pass

    clock = pg.time.Clock()

    while must_continue:
        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                must_continue = False

        if async_display:
            # Non bloquant : on dessine dès qu'un message arrive
            if comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_FRAME):
                status = MPI.Status()
                comm.probe(source=MPI.ANY_SOURCE, tag=TAG_FRAME, status=status)
                src = status.Get_source()
                buf = np.empty(ny * nx, dtype=np.uint8)
                comm.Recv(buf, source=src, tag=TAG_FRAME)
                last_frame = buf.reshape(ny, nx).copy()
                draw(last_frame)
        else:
            # En mode synchrone, le main (rank 0) va recevoir des grilles via Gather et appelle draw()
            # -> ce loop est géré dans main() plus bas
            pass

        clock.tick(120)  # limite CPU côté affichage

    # Stop: prévenir tout le monde
    for r in range(1, comm.Get_size()):
        comm.send(True, dest=r, tag=TAG_STOP)

    pg.quit()


# ---------- Décomposition 1D (par lignes) ----------
def split_rows(ny: int, nworkers: int) -> tuple[list[int], list[int]]:
    """
    Retourne (counts, displs) en nombre de lignes pour chaque worker (workers indexés 0..nworkers-1).
    Répartition quasi uniforme.
    """
    base = ny // nworkers
    rem = ny % nworkers
    counts = [base + (1 if w < rem else 0) for w in range(nworkers)]
    displs = [0]
    for w in range(1, nworkers):
        displs.append(displs[-1] + counts[w - 1])
    return counts, displs


# ---------- Calcul vectorisé local avec ghost rows ----------
def step_local_with_ghost(local_with_ghost: np.ndarray) -> np.ndarray:
    """
    local_with_ghost shape = (local_ny + 2, nx)
    row 0 and row -1 are ghost rows.
    Returns next_local (without ghosts) shape = (local_ny, nx)
    Règles Game of Life:
      - vivant survit si 2 ou 3 voisins
      - mort naît si 3 voisins
    Toroïdal sur colonnes via np.roll ; toroïdal sur lignes assuré par échange de ghost rows.
    """
    A = local_with_ghost  # uint8 0/1
    # Somme des 8 voisins (vectorisée)
    # on calcule sur toutes les lignes, puis on ne gardera que l'intérieur.
    up = np.roll(A, shift=-1, axis=0)
    down = np.roll(A, shift=+1, axis=0)

    # gauche/droite sur chaque "plan" (up, A, down)
    def lr(X):
        return np.roll(X, +1, axis=1), np.roll(X, -1, axis=1)

    A_l, A_r = lr(A)
    up_l, up_r = lr(up)
    down_l, down_r = lr(down)

    neigh = (up + down + A_l + A_r + up_l + up_r + down_l + down_r).astype(np.uint8)

    # Ne garder que les lignes réelles 1..-2
    center = A[1:-1]
    neigh_center = neigh[1:-1]

    # Règles
    born = (center == 0) & (neigh_center == 3)
    survive = (center == 1) & ((neigh_center == 2) | (neigh_center == 3))
    next_center = np.zeros_like(center, dtype=np.uint8)
    next_center[born | survive] = 1
    return next_center


# ---------- Main MPI ----------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2 and rank == 0:
        print("Il faut au moins 2 processus MPI : 1 affiche + >=1 calculeur.")
        sys.exit(1)

    # Args
    choice = "glider"
    resx, resy = 800, 800
    async_display = False  # mets True si tu veux le mode asynchrone (Iprobe)
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    if len(sys.argv) > 3:
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
    if len(sys.argv) > 4:
        async_display = (sys.argv[4].lower() in ("1", "true", "yes", "y"))

    if choice not in DICO_PATTERNS:
        if rank == 0:
            print("Pattern inconnu. Disponibles:", list(DICO_PATTERNS.keys()))
        sys.exit(1)

    (ny, nx), init_list = DICO_PATTERNS[choice]

    # Workers = ranks 1..size-1
    nworkers = size - 1
    worker_id = rank - 1  # 0..nworkers-1 si rank>=1

    # Répartition des lignes entre workers
    counts_rows, displs_rows = split_rows(ny, nworkers)
    my_rows = counts_rows[worker_id] if rank >= 1 else 0
    my_start = displs_rows[worker_id] if rank >= 1 else 0

    TAG_FRAME = 100
    TAG_STOP = 200

    # -------- Rank 0 : initialisation grille, distribution, affichage --------
    if rank == 0:
        # grille initiale
        full = np.zeros((ny, nx), dtype=np.uint8)
        for (i, j) in init_list:
            if 0 <= i < ny and 0 <= j < nx:
                full[i, j] = 1

        print(f"[rank 0] Pattern: {choice}, grille={ny}x{nx}, fenêtre={resx}x{resy}, workers={nworkers}")
        print(f"[rank 0] Mode affichage asynchrone = {async_display}")

        # Scatterv: envoie des blocs de lignes aux workers
        # On envoie des tableaux plats
        sendbuf = full.ravel()
        sendcounts = [counts_rows[w] * nx for w in range(nworkers)]
        displs = [displs_rows[w] * nx for w in range(nworkers)]

        for w in range(nworkers):
            comm.Send([sendbuf[displs[w]: displs[w] + sendcounts[w]], MPI.UINT8_T], dest=w + 1, tag=10)

        # Si affichage asynchrone:
        # - rank 0 boucle pygame et reçoit des frames (point-à-point) envoyées par un worker "assembler".
        # Pour simplifier: on choisit worker 1 (rank=1) comme assembleur: il fait Gather des workers,
        # puis envoie la frame complète à rank 0.
        if async_display:
            pygame_app_loop(comm, ny, nx, resx, resy, async_display=True)
            return

        # Mode synchrone (plus simple): rank 0 fait Gather à chaque itération et dessine.
        import pygame as pg
        pg.init()

        size_x = max(1, resx // nx)
        size_y = max(1, resy // ny)
        width = nx * size_x
        height = ny * size_y
        screen = pg.display.set_mode((width, height))
        col_life = pg.Color("black")
        col_dead = pg.Color("white")
        draw_grid = (size_x > 4 and size_y > 4)
        grid_color = pg.Color("lightgrey")

        def draw(cells: np.ndarray) -> None:
            for i in range(ny):
                y = height - size_y * (i + 1)
                row = cells[i]
                for j in range(nx):
                    x = size_x * j
                    screen.fill(col_life if row[j] else col_dead, (x, y, size_x, size_y))
            if draw_grid:
                for i in range(ny):
                    pg.draw.line(screen, grid_color, (0, i * size_y), (width, i * size_y))
                for j in range(nx):
                    pg.draw.line(screen, grid_color, (j * size_x, 0), (j * size_x, height))
            pg.display.update()

        must_continue = True
        full_current = full.copy()

        # Buffers pour GatherV
        recvbuf = np.empty(ny * nx, dtype=np.uint8)

        # Comme GatherV “classique” n’existe pas directement en python avec types simples,
        # on fait un Gather “manuel” point-à-point: chaque worker envoie son bloc, root reconstruit.
        # (c’est très OK pour un TP)
        while must_continue:
            t_draw0 = time.time()

            # Réception des blocs de workers
            for w in range(nworkers):
                n = counts_rows[w] * nx
                buf = np.empty(n, dtype=np.uint8)
                comm.Recv(buf, source=w + 1, tag=TAG_FRAME)
                start = displs_rows[w] * nx
                recvbuf[start:start + n] = buf

            full_current = recvbuf.reshape(ny, nx).copy()

            t_draw1 = time.time()
            draw(full_current)
            t_draw2 = time.time()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    must_continue = False

            # affiche timing côté rank 0
            print(f"[rank0] recv+assemble={t_draw1-t_draw0:2.2e}s  draw={t_draw2-t_draw1:2.2e}s\r", end="")

        # Stop aux workers
        for r in range(1, size):
            comm.send(True, dest=r, tag=TAG_STOP)
        pg.quit()
        return

    # -------- Workers (rank >= 1) --------
    else:
        # Recevoir mon bloc initial de lignes
        local0 = np.empty(my_rows * nx, dtype=np.uint8)
        comm.Recv(local0, source=0, tag=10)
        local = local0.reshape(my_rows, nx).copy()

        # tableau avec ghost rows: (my_rows + 2, nx)
        local_g = np.zeros((my_rows + 2, nx), dtype=np.uint8)
        local_g[1:-1] = local

        # voisins en anneau (tore sur la dimension découpée)
        up_rank = 1 + ((worker_id - 1) % nworkers)
        down_rank = 1 + ((worker_id + 1) % nworkers)

        must_stop = False
        it = 0

        # Pour mode async: on choisit rank=1 comme "assembleur" qui récupère tous les blocs,
        # reconstruit full, et envoie full à rank 0 (afficheur) en point-à-point.
        is_assembler = (rank == 1)
        TAG_BLOCK = 110  # envoi bloc -> assembleur

        while not must_stop:
            # stop?
            if comm.iprobe(source=0, tag=TAG_STOP):
                must_stop = comm.recv(source=0, tag=TAG_STOP)
                break

            t1 = time.time()

            # 1) échange ghost rows (haut/bas)
            # envoyer ma première ligne réelle vers le haut (pour devenir ghost bas du voisin up)
            # envoyer ma dernière ligne réelle vers le bas
            send_top = local_g[1].copy()
            send_bottom = local_g[-2].copy()

            recv_from_down = np.empty(nx, dtype=np.uint8)  # deviendra ghost bottom (ligne -1)
            recv_from_up = np.empty(nx, dtype=np.uint8)    # deviendra ghost top (ligne 0)

            # échanges simultanés: Sendrecv
            comm.Sendrecv(sendbuf=send_top, dest=up_rank, sendtag=20,
                          recvbuf=recv_from_down, source=down_rank, recvtag=20)
            comm.Sendrecv(sendbuf=send_bottom, dest=down_rank, sendtag=30,
                          recvbuf=recv_from_up, source=up_rank, recvtag=30)

            local_g[0] = recv_from_up
            local_g[-1] = recv_from_down

            # 2) calcul next (vectorisé)
            next_local = step_local_with_ghost(local_g)

            # update
            local_g[1:-1] = next_local

            t2 = time.time()

            # 3) envoyer vers affichage
            if comm.Get_attr(MPI.TAG_UB) is None:
                pass

            if False:
                # debug
                pass

            if False:
                pass

            if False:
                pass

            # Mode synchrone: chaque worker envoie son bloc directement à rank 0 (root assemble)
            if not ("__ASYNC__" in globals()):
                comm.Send(next_local.ravel(), dest=0, tag=TAG_FRAME)

            # Mode async: chaque worker envoie son bloc à l'assembleur (rank 1),
            # l'assembleur reconstruit full puis l'envoie à rank 0.
            # (Tu actives ce mode via argv[4]=true)
            # Ici on regarde un broadcast implicite: si rank0 a démarré async_display, on va le détecter
            # par convention: l'assembleur enverra full à rank0 et rank0 recevra.
            # => Pour garder simple: on déclenche async si l'utilisateur a mis argv[4]=true,
            # mais ce flag n'est connu que localement -> on le recalcule à partir de sys.argv.
            async_flag = (len(sys.argv) > 4 and sys.argv[4].lower() in ("1", "true", "yes", "y"))
            if async_flag:
                # bloc -> assembleur
                comm.Send(next_local.ravel(), dest=1, tag=TAG_BLOCK)

                if is_assembler:
                    # récupérer tous les blocs (y compris le mien) puis envoyer full à rank0
                    # On reconstruit full sur l'assembleur
                    full_buf = np.empty(ny * nx, dtype=np.uint8)

                    # place mon bloc
                    start = displs_rows[0] * nx  # worker_id=0 pour rank=1
                    full_buf[start:start + my_rows * nx] = next_local.ravel()

                    # recevoir les autres blocs
                    for other_worker in range(1, nworkers):
                        src_rank = other_worker + 1
                        n = counts_rows[other_worker] * nx
                        buf = np.empty(n, dtype=np.uint8)
                        comm.Recv(buf, source=src_rank, tag=TAG_BLOCK)
                        st = displs_rows[other_worker] * nx
                        full_buf[st:st + n] = buf

                    # envoyer frame complète à rank0
                    comm.Send(full_buf, dest=0, tag=TAG_FRAME)

            t3 = time.time()

            # petit print côté worker 
            if it % 50 == 0:
                print(f"[rank{rank}] calc={t2-t1:2.2e}s  comm={t3-t2:2.2e}s", flush=True)

            it += 1


if __name__ == "__main__":
    main()
