/* bench_parasail.py's C driver: runs one parasail function over a batch of pairs with
 * real threads, no Python between the pairs.
 *
 * The Python bindings cost ~10 us per call before a cell is computed (ctypes, result
 * object, profile build) and their threads share the GIL, so through them a 22x50 pair
 * measures the binding and a 12-thread run measures the lock.  This driver links the
 * wheel's own libparasail.so, so the kernels are the ones a Python user runs.
 *
 *   driver <funcname> <matrix> <open> <extend> <threads> <reps> <min_time> <cigar> <pairs>
 *
 * funcname: a parasail function name (parasail_nw_striped_16 ...), resolved by
 *           parasail_lookup_function, so one binary serves every arm.
 * matrix:   "blosum62" or "dna:<match>:<mismatch>" (parasail_matrix_create over ACGT).
 * cigar:    1 = also walk the traceback (parasail_result_get_cigar) on every pair, the
 *           equivalent of nwgrad's keep_paths; the function must be a *_trace_* one.
 * pairs:    a text file, one pair per line, A and B separated by a tab.
 *
 * Threads are spawned per repetition (nwgrad's batch spawns its workers per call too),
 * pairs handed out by an atomic counter in chunks of 16.  Prints one JSON line:
 * {"secs": best time of one pass over the batch, "saturated": any pair saturated,
 *  "scores": [...]}.  Scores are what the comparison checks against nwgrad's.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "parasail.h"
#include "parasail/matrices/blosum62.h"

typedef struct { char *a, *b; int la, lb; } pair_t;

static pair_t *pairs; static int npairs;
static parasail_function_t *fn; static const parasail_matrix_t *matrix;
static int gopen, gext, do_cigar;
static atomic_int next_pair; static atomic_int saturated;
static int *scores;

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *worker(void *arg) {
    (void)arg;
    for (;;) {
        int i0 = atomic_fetch_add(&next_pair, 16);
        if (i0 >= npairs) return NULL;
        int i1 = i0 + 16 < npairs ? i0 + 16 : npairs;
        for (int i = i0; i < i1; ++i) {
            pair_t *p = &pairs[i];
            parasail_result_t *r = fn(p->a, p->la, p->b, p->lb, gopen, gext, matrix);
            scores[i] = parasail_result_get_score(r);
            if (parasail_result_is_saturated(r)) atomic_store(&saturated, 1);
            if (do_cigar) {
                parasail_cigar_t *c = parasail_result_get_cigar(r, p->a, p->la, p->b, p->lb, matrix);
                parasail_cigar_free(c);
            }
            parasail_result_free(r);
        }
    }
}

static void once(int threads) {
    atomic_store(&next_pair, 0);
    if (threads <= 1) { worker(NULL); return; }
    pthread_t *th = malloc(sizeof(pthread_t) * threads);
    for (int t = 0; t < threads; ++t) pthread_create(&th[t], NULL, worker, NULL);
    for (int t = 0; t < threads; ++t) pthread_join(th[t], NULL);
    free(th);
}

int main(int argc, char **argv) {
    if (argc != 10) { fprintf(stderr, "usage: %s funcname matrix open extend threads reps min_time cigar pairs\n", argv[0]); return 2; }
    fn = parasail_lookup_function(argv[1]);
    if (!fn) { fprintf(stderr, "unknown parasail function %s\n", argv[1]); return 2; }
    parasail_matrix_t *owned = NULL;
    if (strcmp(argv[2], "blosum62") == 0) matrix = &parasail_blosum62;
    else if (strncmp(argv[2], "dna:", 4) == 0) {
        int match, mismatch;
        if (sscanf(argv[2] + 4, "%d:%d", &match, &mismatch) != 2) { fprintf(stderr, "bad matrix %s\n", argv[2]); return 2; }
        owned = parasail_matrix_create("ACGT", match, mismatch); matrix = owned;
    } else { fprintf(stderr, "bad matrix %s\n", argv[2]); return 2; }
    gopen = atoi(argv[3]); gext = atoi(argv[4]);
    int threads = atoi(argv[5]), reps = atoi(argv[6]); double min_time = atof(argv[7]);
    do_cigar = atoi(argv[8]);

    FILE *f = fopen(argv[9], "r");
    if (!f) { perror(argv[9]); return 2; }
    int cap = 1024; pairs = malloc(sizeof(pair_t) * cap);
    char *line = NULL; size_t lcap = 0; ssize_t len;
    while ((len = getline(&line, &lcap, f)) > 0) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        if (npairs == cap) { cap *= 2; pairs = realloc(pairs, sizeof(pair_t) * cap); }
        pair_t *p = &pairs[npairs++];
        p->a = strdup(line); p->la = (int)strlen(p->a);
        p->b = strdup(tab + 1); p->lb = (int)strlen(p->b);
    }
    free(line); fclose(f);
    scores = malloc(sizeof(int) * npairs);

    once(threads);                               /* warm-up, and the scores */
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        double t0 = now(); int k = 0;
        for (;;) { once(threads); ++k; double dt = now() - t0; if (dt >= min_time) { double per = dt / k; if (per < best) best = per; break; } }
    }
    printf("{\"secs\": %.9g, \"saturated\": %d, \"scores\": [", best, atomic_load(&saturated));
    for (int i = 0; i < npairs; ++i) printf(i ? ", %d" : "%d", scores[i]);
    printf("]}\n");
    if (owned) parasail_matrix_free(owned);
    return 0;
}
