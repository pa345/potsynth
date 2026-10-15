/*
 * test_threads.c
 *
 * One shared model workspace evaluated concurrently from several threads,
 * each with its own evaluation workspace; results must equal the serial
 * results bitwise. Also run under ThreadSanitizer (tests/check.sh).
 */

#define TEST_NTHREADS                8

typedef struct
{
  const potsynth_workspace * w;
  size_t n;
  size_t chunk;
  const double * r;
  const double * theta;
  const double * phi;
  double * B;
} test_thread_job;

/* evaluate the job three times with a private evaluation workspace */
static void *
test_thread_worker(void * arg)
{
  test_thread_job * job = (test_thread_job *) arg;
  potsynth_eval_workspace * ew = potsynth_eval_alloc(job->chunk, job->w);
  int k;

  for (k = 0; k < 3; ++k)
    potsynth_eval(job->n, job->r, job->theta, job->phi, job->B, ew);

  potsynth_eval_free(ew);

  return NULL;
}

static void
test_threads_method(const int method)
{
  const double a = 6371.2;
  const double rmin = a + 250.0;
  const double rmax = a + 800.0;
  const size_t L = 60;
  const size_t n = 20000;
  double * gh = test_random_coef(1, L, L);
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * Bs = malloc(3 * n * sizeof(double));
  double * Bt = malloc(TEST_NTHREADS * 3 * n * sizeof(double));
  potsynth_parameters params = potsynth_default_parameters();
  test_thread_job jobs[TEST_NTHREADS];
  potsynth_workspace * w;
  potsynth_eval_workspace * ew;
  int t, nfail = 0;
#ifdef HAVE_PTHREADS
  pthread_t tid[TEST_NTHREADS];
#endif

  test_random_points(n, rmin, rmax, r, theta, phi);

  params.method = method;
  w = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, POTSYNTH_MAG_B, &params);

  ew = potsynth_eval_alloc(0, w);
  potsynth_eval(n, r, theta, phi, Bs, ew);
  potsynth_eval_free(ew);

  for (t = 0; t < TEST_NTHREADS; ++t)
    {
      jobs[t].w = w;
      jobs[t].n = n;
      jobs[t].chunk = 1000 + 337 * (size_t) t;
      jobs[t].r = r;
      jobs[t].theta = theta;
      jobs[t].phi = phi;
      jobs[t].B = Bt + (size_t) t * 3 * n;
    }

#ifdef HAVE_PTHREADS
  for (t = 0; t < TEST_NTHREADS; ++t)
    pthread_create(&tid[t], NULL, test_thread_worker, &jobs[t]);

  for (t = 0; t < TEST_NTHREADS; ++t)
    pthread_join(tid[t], NULL);
#else
  for (t = 0; t < TEST_NTHREADS; ++t)
    test_thread_worker(&jobs[t]);
#endif

  for (t = 0; t < TEST_NTHREADS; ++t)
    {
      if (memcmp(Bs, Bt + (size_t) t * 3 * n, 3 * n * sizeof(double)) != 0)
        ++nfail;
    }

  ps_test(nfail, "threads: method=%d, %d threads sharing one workspace (%d differ)",
          method, TEST_NTHREADS, nfail);

  potsynth_free(w);
  free(gh);
  free(r);
  free(theta);
  free(phi);
  free(Bs);
  free(Bt);
}

static void
test_threads(void)
{
  test_threads_method(POTSYNTH_METHOD_FAST);
  test_threads_method(POTSYNTH_METHOD_DIRECT);
}
