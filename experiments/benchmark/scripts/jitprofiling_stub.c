/*
 * Minimal stub for Intel JIT Profiling API symbols (iJIT_*) that some PyTorch
 * builds reference at runtime.
 *
 * On some systems/environments, importing torch can fail with:
 *   undefined symbol: iJIT_NotifyEvent
 *
 * Those symbols are typically provided by Intel's ITT/JIT profiling runtime
 * (e.g., libittnotify / jitprofiling). For this benchmark we don't need
 * VTune/ITT profiling, so we provide no-op implementations to satisfy the
 * dynamic loader.
 *
 * This is intentionally tiny and dependency-free.
 */

#include <stdatomic.h>

static atomic_uint g_next_method_id = 1;

/* Return a unique-ish method id. */
unsigned int iJIT_GetNewMethodID(void) {
  return atomic_fetch_add(&g_next_method_id, 1);
}

/* 0 = not profiling. */
int iJIT_IsProfilingActive(void) {
  return 0;
}

/* No-op; return 0 to indicate success. */
int iJIT_NotifyEvent(int event_type, void *event_specific_data) {
  (void)event_type;
  (void)event_specific_data;
  return 0;
}

