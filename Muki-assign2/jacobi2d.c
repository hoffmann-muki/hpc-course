#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/*
 * 2D Jacobi Heat Diffusion with MPI
 * Uses 2D domain decomposition with periodic boundary conditions.
 *
 * Usage: mpirun -np <P> ./jacobi2d <input_file> <timesteps> <size_X> <size_Y>
 *
 * The 2D grid of size_X x size_Y is decomposed across a px x py process grid.
 * Each process owns a local block of (size_X/px) x (size_Y/py) elements
 * plus ghost/halo rows and columns for neighbor communication.
 */

/* Factor nprocs into px * py for 2D decomposition, trying to keep them as
 * close to each other as possible (square-ish decomposition). */
static void compute_proc_grid(int nprocs, int *px, int *py)
{
    int p = (int)sqrt((double)nprocs);
    while (p > 0) {
        if (nprocs % p == 0) {
            *px = p;
            *py = nprocs / p;
            return;
        }
        p--;
    }
    *px = 1;
    *py = nprocs;
}

int main(int argc, char **argv)
{
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 5) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <input_file> <timesteps> <size_X> <size_Y>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    int timesteps = atoi(argv[2]);
    int arg_X = atoi(argv[3]);    /* size_X from command line (columns, X = horizontal) */
    int arg_Y = atoi(argv[4]);    /* size_Y from command line (rows,    Y = vertical)   */
    int size_X = arg_Y;           /* internal rows */
    int size_Y = arg_X;           /* internal cols */

    /* Compute 2D process grid */
    int px, py;
    compute_proc_grid(nprocs, &px, &py);

    /* My coordinates in the process grid */
    int my_row = rank / py;   /* row index in process grid */
    int my_col = rank % py;   /* col index in process grid */

    /* Local block dimensions (without ghost cells) */
    int local_X = size_X / px;
    int local_Y = size_Y / py;

    /* Global starting indices for this process */
    int start_i = my_row * local_X;
    int start_j = my_col * local_Y;

    /* Allocate local arrays with ghost cells: (local_X+2) x (local_Y+2)
     * Index [1..local_X][1..local_Y] is the real data.
     * Index 0 and local_X+1 are ghost rows; similarly for columns. */
    int lx = local_X + 2;
    int ly = local_Y + 2;
    double *A    = (double *)calloc(lx * ly, sizeof(double));
    double *Anew = (double *)calloc(lx * ly, sizeof(double));
    if (!A || !Anew) {
        fprintf(stderr, "Rank %d: memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Macro for 2D indexing into local array (with ghost cells) */
    #define IDX(i, j) ((i) * ly + (j))

    /* --- Read input file on rank 0 and distribute --- */
    double *global = NULL;
    if (rank == 0) {
        global = (double *)malloc(size_X * size_Y * sizeof(double));
        if (!global) {
            fprintf(stderr, "Rank 0: failed to allocate global array\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        FILE *fp = fopen(input_file, "r");
        if (!fp) {
            fprintf(stderr, "Cannot open input file: %s\n", input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < size_X; i++) {
            for (int j = 0; j < size_Y; j++) {
                if (j < size_Y - 1)
                    fscanf(fp, "%lf,", &global[i * size_Y + j]);
                else
                    fscanf(fp, "%lf", &global[i * size_Y + j]);
            }
        }
        fclose(fp);
    }

    /* Scatter data: rank 0 sends each process its local block */
    if (rank == 0) {
        /* Copy rank 0's own block */
        for (int i = 0; i < local_X; i++)
            for (int j = 0; j < local_Y; j++)
                A[IDX(i + 1, j + 1)] = global[(start_i + i) * size_Y + (start_j + j)];

        /* Send to other ranks */
        for (int r = 1; r < nprocs; r++) {
            int r_row = r / py;
            int r_col = r % py;
            int r_si = r_row * local_X;
            int r_sj = r_col * local_Y;
            double *buf = (double *)malloc(local_X * local_Y * sizeof(double));
            for (int i = 0; i < local_X; i++)
                for (int j = 0; j < local_Y; j++)
                    buf[i * local_Y + j] = global[(r_si + i) * size_Y + (r_sj + j)];
            MPI_Send(buf, local_X * local_Y, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            free(buf);
        }
        free(global);
    } else {
        double *buf = (double *)malloc(local_X * local_Y * sizeof(double));
        MPI_Recv(buf, local_X * local_Y, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < local_X; i++)
            for (int j = 0; j < local_Y; j++)
                A[IDX(i + 1, j + 1)] = buf[i * local_Y + j];
        free(buf);
    }

    /* Determine neighbor ranks with periodic (wrap-around) boundaries */
    int north = ((my_row - 1 + px) % px) * py + my_col;
    int south = ((my_row + 1) % px) * py + my_col;
    int west  = my_row * py + ((my_col - 1 + py) % py);
    int east  = my_row * py + ((my_col + 1) % py);

    /* Allocate send/recv buffers for row and column exchanges */
    double *send_north = (double *)malloc(local_Y * sizeof(double));
    double *send_south = (double *)malloc(local_Y * sizeof(double));
    double *recv_north = (double *)malloc(local_Y * sizeof(double));
    double *recv_south = (double *)malloc(local_Y * sizeof(double));
    double *send_west  = (double *)malloc(local_X * sizeof(double));
    double *send_east  = (double *)malloc(local_X * sizeof(double));
    double *recv_west  = (double *)malloc(local_X * sizeof(double));
    double *recv_east  = (double *)malloc(local_X * sizeof(double));

    /* ---- Main computation loop ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int t = 0; t < timesteps; t++) {

        /* --- Exchange ghost cells with neighbors --- */

        /* Pack north row (row 1) and south row (row local_X) */
        for (int j = 0; j < local_Y; j++) {
            send_north[j] = A[IDX(1, j + 1)];
            send_south[j] = A[IDX(local_X, j + 1)];
        }

        /* Pack west column (col 1) and east column (col local_Y) */
        for (int i = 0; i < local_X; i++) {
            send_west[i] = A[IDX(i + 1, 1)];
            send_east[i] = A[IDX(i + 1, local_Y)];
        }

        MPI_Request reqs[8];

        /* North-South exchange (row ghosts) */
        MPI_Isend(send_north, local_Y, MPI_DOUBLE, north, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_south, local_Y, MPI_DOUBLE, south, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(send_south, local_Y, MPI_DOUBLE, south, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(recv_north, local_Y, MPI_DOUBLE, north, 1, MPI_COMM_WORLD, &reqs[3]);

        /* West-East exchange (column ghosts) */
        MPI_Isend(send_west, local_X, MPI_DOUBLE, west, 2, MPI_COMM_WORLD, &reqs[4]);
        MPI_Irecv(recv_east, local_X, MPI_DOUBLE, east, 2, MPI_COMM_WORLD, &reqs[5]);
        MPI_Isend(send_east, local_X, MPI_DOUBLE, east, 3, MPI_COMM_WORLD, &reqs[6]);
        MPI_Irecv(recv_west, local_X, MPI_DOUBLE, west, 3, MPI_COMM_WORLD, &reqs[7]);

        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

        /* Unpack received ghost cells */
        for (int j = 0; j < local_Y; j++) {
            A[IDX(0, j + 1)]           = recv_north[j];  /* north ghost row */
            A[IDX(local_X + 1, j + 1)] = recv_south[j];  /* south ghost row */
        }
        for (int i = 0; i < local_X; i++) {
            A[IDX(i + 1, 0)]           = recv_west[i];   /* west ghost col */
            A[IDX(i + 1, local_Y + 1)] = recv_east[i];   /* east ghost col */
        }

        /* --- Compute Jacobi update --- */
        for (int i = 1; i <= local_X; i++) {
            for (int j = 1; j <= local_Y; j++) {
                Anew[IDX(i, j)] = 0.2 * (A[IDX(i, j)]
                                        + A[IDX(i - 1, j)]
                                        + A[IDX(i + 1, j)]
                                        + A[IDX(i, j - 1)]
                                        + A[IDX(i, j + 1)]);
            }
        }

        /* Swap pointers */
        double *tmp = A;
        A = Anew;
        Anew = tmp;
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    /* --- Compute Min, Avg, Max time using MPI reductions --- */
    double min_time, max_time, sum_time;
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_time = sum_time / nprocs;
        printf("TIME: Min: %.3f s Avg: %.3f s Max: %.3f s\n", min_time, avg_time, max_time);
    }

    /* --- Gather results to rank 0 and write output --- */
    if (rank == 0) {
        global = (double *)malloc(size_X * size_Y * sizeof(double));

        /* Copy rank 0's local data */
        for (int i = 0; i < local_X; i++)
            for (int j = 0; j < local_Y; j++)
                global[(start_i + i) * size_Y + (start_j + j)] = A[IDX(i + 1, j + 1)];

        /* Receive from other ranks */
        for (int r = 1; r < nprocs; r++) {
            int r_row = r / py;
            int r_col = r % py;
            int r_si = r_row * local_X;
            int r_sj = r_col * local_Y;
            double *buf = (double *)malloc(local_X * local_Y * sizeof(double));
            MPI_Recv(buf, local_X * local_Y, MPI_DOUBLE, r, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < local_X; i++)
                for (int j = 0; j < local_Y; j++)
                    global[(r_si + i) * size_Y + (r_sj + j)] = buf[i * local_Y + j];
            free(buf);
        }

        /* Write output file */
        char outname[256];
        snprintf(outname, sizeof(outname), "%dx%d.%d-output.csv", arg_X, arg_Y, nprocs);
        FILE *fp = fopen(outname, "w");
        if (!fp) {
            fprintf(stderr, "Cannot open output file: %s\n", outname);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < size_X; i++) {
            for (int j = 0; j < size_Y; j++) {
                if (j > 0) fprintf(fp, ",");
                fprintf(fp, "%.3f", global[i * size_Y + j]);
            }
            fprintf(fp, "\r\n");
        }
        fclose(fp);
        free(global);
    } else {
        /* Send local data to rank 0 */
        double *buf = (double *)malloc(local_X * local_Y * sizeof(double));
        for (int i = 0; i < local_X; i++)
            for (int j = 0; j < local_Y; j++)
                buf[i * local_Y + j] = A[IDX(i + 1, j + 1)];
        MPI_Send(buf, local_X * local_Y, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
        free(buf);
    }

    /* Cleanup */
    free(A);
    free(Anew);
    free(send_north); free(send_south);
    free(recv_north); free(recv_south);
    free(send_west);  free(send_east);
    free(recv_west);  free(recv_east);

    MPI_Finalize();
    return 0;
}
