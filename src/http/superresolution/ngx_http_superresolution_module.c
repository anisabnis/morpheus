#include <nginx.h>
#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>

ngx_module_t ngx_http_superresolution_module;

static ngx_http_output_header_filter_pt ngx_http_next_header_filter;
static ngx_http_output_body_filter_pt   ngx_http_next_body_filter;

ngx_int_t ngx_http_superresolution_scale_video(ngx_http_request_t *r, ngx_chain_t *in, ngx_chain_t *onnx, u_char* buf, int buflen, ngx_str_t *cache_file_name, size_t content_start);
ngx_int_t ngx_http_superresolution_scale_image(ngx_http_request_t *r, ngx_chain_t *in, char* buf, int buflen);
ngx_int_t ngx_http_superresolution_delete_alternate_file(ngx_http_request_t *r);
ngx_int_t ngx_http_superresolution_get_scale(ngx_http_request_t *r);

void preload_all_ort_sessions(ngx_log_t *log);


static ngx_http_module_t ngx_http_superresolution_module_ctx;

static ngx_atomic_t   *CPU_queue_size;
static ngx_atomic_t   *GPU_queue_size;

static ngx_shmtx_t    *queue_mutex;
static ngx_shmtx_t    *rbtree_mutex;
static ngx_shmtx_t     rbtree_mtx_storage;
static ngx_rbtree_t    *task_rbtree;
static ngx_rbtree_node_t   *task_sentinel;
static ngx_atomic_t    *rbtree_lock_ptr;
static ngx_atomic_t    *rbtree_wait_ptr;

typedef struct {
    ngx_rbtree_node_t node;
    ngx_uint_t        id;            /* key + id */
    ngx_msec_t        last_seen;
    ngx_int_t         time_reserved;
    ngx_uint_t        is_gpu;
} ngx_superresolution_task_t;

typedef ngx_int_t (*ngx_superresolution_queue_fn)(ngx_atomic_t *cpu_queue_atomic, ngx_atomic_t *gpu_queue_size,
    void *ctx);

// Forward declaration - must be before ngx_http_superresolution_init_task_shm
static void ngx_superresolution_rbtree_insert(ngx_rbtree_node_t *temp,
    ngx_rbtree_node_t *node, ngx_rbtree_node_t *sentinel);

typedef struct {
  ngx_chain_t    *in;
  ngx_int_t time_reserved;
  ngx_int_t accounted;
  ngx_event_t    watchdog;
  ngx_uint_t     watchdog_registered;
  ngx_uint_t is_gpu;
  ngx_uint_t scale;
} ngx_http_superresolution_ctx_t;

static ngx_slab_pool_t     *queue_shpool;   // owns queue_mutex, CPU/GPU atomics ONLY
static ngx_slab_pool_t     *task_shpool;    // owns task nodes + rbtree

typedef struct {
    ngx_atomic_t cpu_queue;
    ngx_atomic_t gpu_queue;
} ngx_superresolution_queue_shm_t;

typedef struct {
    ngx_rbtree_t      rbtree;
    ngx_rbtree_node_t sentinel;
    ngx_atomic_t      rbtree_lock;
    ngx_atomic_t      rbtree_wait; 
} ngx_superresolution_task_shm_t;


static ngx_int_t
ngx_http_superresolution_init_queue_shm(ngx_shm_zone_t *zone, void *data)
{
    ngx_slab_pool_t *shpool = (ngx_slab_pool_t *) zone->shm.addr;
    ngx_superresolution_queue_shm_t *q;

    if (data) {
        q = data;
    } else {
        q = ngx_slab_alloc(shpool, sizeof(ngx_superresolution_queue_shm_t));
        if (q == NULL) return NGX_ERROR;
        q->cpu_queue = 0;
        q->gpu_queue = 0;
    }

    CPU_queue_size = &q->cpu_queue;
    GPU_queue_size = &q->gpu_queue;
    queue_mutex    = &shpool->mutex;  // queue_shpool mutex ONLY
    queue_shpool   = shpool;
    zone->data     = q;
    return NGX_OK;
}

static ngx_int_t
ngx_http_superresolution_init_task_shm(ngx_shm_zone_t *zone, void *data)
{
    ngx_slab_pool_t *shpool = (ngx_slab_pool_t *) zone->shm.addr;
    ngx_superresolution_task_shm_t *t;

    if (data) {
        t = data;
    } else {
        t = ngx_slab_alloc(shpool, sizeof(ngx_superresolution_task_shm_t));
        if (t == NULL) return NGX_ERROR;
        t->rbtree_lock = 0;
        t->rbtree_wait = 0;

        // zero-initialize sentinel BEFORE ngx_rbtree_init
        // ngx_rbtree_sentinel_init only sets color - left/right/parent stay garbage otherwise
        ngx_memzero(&t->sentinel, sizeof(ngx_rbtree_node_t));
        ngx_rbtree_init(&t->rbtree, &t->sentinel, ngx_superresolution_rbtree_insert);
    }

    ngx_shmtx_sh_t *rbtree_sh = (ngx_shmtx_sh_t *) &t->rbtree_lock;
    if (ngx_shmtx_create(&rbtree_mtx_storage, rbtree_sh, NULL) != NGX_OK) {
        return NGX_ERROR;
    }

    rbtree_mutex    = &rbtree_mtx_storage;
    task_shpool     = shpool;   // DIFFERENT from queue_shpool - no conflict!
    task_rbtree     = &t->rbtree;
    task_sentinel   = &t->sentinel;
    rbtree_lock_ptr = &t->rbtree_lock;
    rbtree_wait_ptr = &t->rbtree_wait;
    zone->data      = t;
    return NGX_OK;
}


static ngx_uint_t
ngx_superresolution_task_id(ngx_http_request_t *r)
{
    return (ngx_uint_t) r->connection->number;
}

static void
ngx_superresolution_rbtree_insert(ngx_rbtree_node_t *temp,
                                  ngx_rbtree_node_t *node,
                                  ngx_rbtree_node_t *sentinel)
{

    ngx_rbtree_node_t **p;
    for ( ;; ) {
        if (node->key < temp->key) {
            p = &temp->left;
        } else if (node->key > temp->key) {
            p = &temp->right;
        } else {
            ngx_superresolution_task_t *n = (ngx_superresolution_task_t *) node;
            ngx_superresolution_task_t *t = (ngx_superresolution_task_t *) temp;
            p = (n->id < t->id) ? &temp->left : &temp->right;
        }

        if (*p == sentinel) {
            break;
        }
        temp = *p;
    }

    *p = node;
    node->parent = temp;
    node->left = sentinel;
    node->right = sentinel;
    ngx_rbt_red(node);

}


static ngx_int_t
ngx_superresolution_task_add(ngx_http_request_t *r, ngx_int_t time_reserved, ngx_uint_t is_gpu)
{
    ngx_superresolution_task_t *t;
    ngx_uint_t id = ngx_superresolution_task_id(r);

    if (task_shpool == NULL || task_rbtree == NULL) {
        ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "superresolution task add failed because shared memory not initialized");
        return NGX_ERROR;
    }


    t = ngx_slab_alloc(task_shpool, sizeof(ngx_superresolution_task_t));
    if (t == NULL) {
        return NGX_ERROR;
    }

    ngx_memzero(t, sizeof(*t));
    t->id = id;
    t->node.key = id;
    t->last_seen = ngx_current_msec;
    t->time_reserved = time_reserved;
    t->is_gpu = is_gpu;

    ngx_log_debug2(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "superresolution task add: adding task to rbtree with id and time_reserved %d and last seen %M", time_reserved, t->last_seen);

    ngx_shmtx_lock(rbtree_mutex);
    ngx_rbtree_insert(task_rbtree, &t->node);
    ngx_shmtx_unlock(rbtree_mutex);
    
    ngx_log_debug2(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "superresolution task add: successfully added task to rbtree with id %u and time_reserved %d", t->id, time_reserved);

    return NGX_OK;
}

static ngx_int_t
ngx_superresolution_task_remove(ngx_http_request_t *r)
{
    ngx_rbtree_node_t *node, *sentinel;
    ngx_uint_t id = ngx_superresolution_task_id(r);

    if (task_rbtree == NULL) {
        return NGX_ERROR;
    }

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "Anirudh superresolution task remove: trying to remove task from rbtree");

    ngx_shmtx_lock(rbtree_mutex);

    node = task_rbtree->root;
    sentinel = task_rbtree->sentinel;

    // sentinel itself could be uninitialized - guard against it
    if (sentinel == NULL) {
        ngx_shmtx_unlock(rbtree_mutex);
        return NGX_OK;
    }

    if (task_rbtree->root == NULL || task_rbtree->root == sentinel) {
        ngx_shmtx_unlock(rbtree_mutex);
        return NGX_OK;
    }

    while (node != sentinel && node != NULL) {
        if (id < node->key) {
            node = node->left;
        } else if (id > node->key) {
            node = node->right;
        } else {
            ngx_superresolution_task_t *t = (ngx_superresolution_task_t *) node;
            if (t->id == id) {
                ngx_rbtree_delete(task_rbtree, node);       
                ngx_shmtx_unlock(rbtree_mutex);
                ngx_slab_free(task_shpool, t);

                ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "Anirudh superresolution removed task from rbtree successfully");

                return NGX_OK;
            }
            node = node->right;
        }
    }

    ngx_shmtx_unlock(rbtree_mutex);
    return NGX_DECLINED;
}

void
ngx_superresolution_task_cleanup(ngx_http_request_t *r)
{
    ngx_rbtree_node_t *node, *next, *sentinel;
    ngx_msec_t now;

    if (task_rbtree == NULL || queue_mutex == NULL || task_shpool == NULL) {
        return;
    }

    now = ngx_current_msec;

    ngx_shmtx_lock(rbtree_mutex);

    sentinel = task_rbtree->sentinel;

    // FIX 1: guard empty tree before ngx_rbtree_min
    if (task_rbtree->root == sentinel) {
        ngx_shmtx_unlock(rbtree_mutex);
        return;
    }

    node = ngx_rbtree_min(task_rbtree->root, sentinel);

    while (node != NULL && node != sentinel) {
        next = ngx_rbtree_next(task_rbtree, node);

        if (next == node) {
            // paranoia: avoid infinite loop
            break;
        }

        ngx_superresolution_task_t *t = (ngx_superresolution_task_t *) node;

        if (now - t->last_seen > (ngx_msec_t) t->time_reserved + 2000) {
            ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
                          "superresolution task stale, releasing queue slot");

            // save before deleting
            ngx_int_t time_reserved = t->time_reserved;
            ngx_uint_t is_gpu = t->is_gpu;

            ngx_rbtree_delete(task_rbtree, node);
            ngx_shmtx_unlock(rbtree_mutex);  // release rbtree_mutex FIRST

            ngx_slab_free(task_shpool, t);   // uses task_shpool->mutex internally - safe

            // NOW acquire queue_mutex - rbtree_mutex is NOT held - no deadlock
            if (time_reserved > 0) {
                ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "sabby trying to acquire queue mutex to release stale task's reserved queue slot");
                ngx_shmtx_lock(queue_mutex);
                if (is_gpu) {
                    ngx_atomic_fetch_add(GPU_queue_size, -time_reserved);
                } else {
                    ngx_atomic_fetch_add(CPU_queue_size, -time_reserved);
                }
                ngx_shmtx_unlock(queue_mutex);
            }

            ngx_shmtx_lock(rbtree_mutex);    // re-acquire for next iteration
            sentinel = task_rbtree->sentinel;

            // guard empty tree on restart
            if (task_rbtree->root == NULL || task_rbtree->root == sentinel) {
                ngx_shmtx_unlock(rbtree_mutex);
                return;
            }

            // restart traversal safely
            next = ngx_rbtree_min(task_rbtree->root, sentinel);
            if (next == NULL || next == sentinel) {
                ngx_shmtx_unlock(rbtree_mutex);
                return;
            }
        }

        node = next;
    }

    ngx_shmtx_unlock(rbtree_mutex);
}

/*
void
ngx_superresolution_task_cleanup(ngx_http_request_t *r)
{
    ngx_rbtree_node_t *node, *next, *sentinel;
    ngx_msec_t now;

    if (task_rbtree == NULL || queue_mutex == NULL || task_shpool == NULL) {
        return;
    }

    now = ngx_current_msec;

    ngx_shmtx_lock(rbtree_mutex);

    sentinel = task_rbtree->sentinel;
    node = ngx_rbtree_min(task_rbtree->root, sentinel);

    while (node != sentinel) {
        next = ngx_rbtree_next(task_rbtree, node);

        ngx_superresolution_task_t *t = (ngx_superresolution_task_t *) node;
        if (now - t->last_seen > (ngx_msec_t) t->time_reserved + 2000) {
            ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
                          "superresolution task stale, releasing queue slot");

            if (t->time_reserved > 0) {
                ngx_shmtx_lock(queue_mutex);
                if (t->is_gpu) {
                    ngx_atomic_fetch_add(GPU_queue_size, -t->time_reserved);
                } else {
                    ngx_atomic_fetch_add(CPU_queue_size, -t->time_reserved);
                }
                ngx_shmtx_unlock(queue_mutex);
            }

            ngx_rbtree_delete(task_rbtree, node);
            ngx_shmtx_unlock(rbtree_mutex);
            ngx_slab_free(task_shpool, t);
            ngx_shmtx_lock(rbtree_mutex);

            sentinel = task_rbtree->sentinel;
            next = ngx_rbtree_min(task_rbtree->root, sentinel);
        }

        node = next;
    }

    ngx_shmtx_unlock(rbtree_mutex);
}
*/
/*
static ngx_int_t
ngx_http_superresolution_init_process(ngx_cycle_t *cycle)
{
    if (task_rbtree == NULL || queue_mutex == NULL || task_shpool == NULL) {
        return NGX_OK;
    }

    ngx_memzero(&task_cleanup_ev, sizeof(ngx_event_t));
    task_cleanup_ev.handler = ngx_superresolution_task_cleanup;
    task_cleanup_ev.log = cycle->log;

    ngx_add_timer(&task_cleanup_ev, task_cleanup_interval);
    return NGX_OK;
}

static void
ngx_http_superresolution_exit_process(ngx_cycle_t *cycle)
{
    if (task_cleanup_ev.timer_set) {
        ngx_del_timer(&task_cleanup_ev);
    }
}
*/
void
update_compute_queue_size(ngx_int_t delta, ngx_http_request_t *r, ngx_uint_t is_gpu)
{
    ngx_shmtx_lock(queue_mutex);
    ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "sabby updating compute queue size by %i", delta);
    if (is_gpu){
        ngx_atomic_fetch_add(GPU_queue_size, delta);
    } else {
        ngx_atomic_fetch_add(CPU_queue_size, delta);
    }
    ngx_shmtx_unlock(queue_mutex);
}


void
update_compute_queue_size_positive(ngx_int_t delta, ngx_http_request_t *r, ngx_uint_t is_gpu)
{

    ngx_shmtx_lock(queue_mutex);
    ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "sabby updating compute queue size by %i", delta);

    if (is_gpu){
        ngx_atomic_fetch_add(GPU_queue_size, delta);
    } else {
        ngx_atomic_fetch_add(CPU_queue_size, delta);
    }
    ngx_shmtx_unlock(queue_mutex);
}

ngx_int_t ngx_release_superresolution_queue_slot_in_ctx(void *data)
{
    ngx_http_request_t *r = (ngx_http_request_t *)data;
    ngx_http_superresolution_ctx_t   *ctx;
    ctx = ngx_http_get_module_ctx(r, ngx_http_superresolution_module);

    if (ctx == NULL) {
        return NGX_ERROR;
    }

    if (ctx->watchdog.timer_set) {
        ngx_del_timer(&ctx->watchdog);
    }

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "Anirudh Releasing superresolution queue slot in ctx for request");

    if (ctx->accounted == 0){
        update_compute_queue_size(-ctx->time_reserved, r, ctx->is_gpu);
        ctx->accounted = 1;
    }

    ngx_superresolution_task_remove(r);

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "Anirudh releases superresolution queue slot in ctx for request successfully");

    return NGX_OK;
}

ngx_int_t 
reserve_superresolution_queue_slot_in_ctx(ngx_http_request_t *r, ngx_int_t time_to_reserve, ngx_int_t accounted, ngx_uint_t is_gpu){
  
    ngx_http_superresolution_ctx_t   *ctx;
    ctx = ngx_http_get_module_ctx(r, ngx_http_superresolution_module);
    
    if (ctx == NULL) {
        return NGX_ERROR;
    }

    ctx->time_reserved = time_to_reserve;
    ctx->accounted = accounted;
    ctx->is_gpu = is_gpu;

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "Anirudh Reserving superresolution queue slot in rbtree for request");

    /* add to task map */
    ngx_superresolution_task_add(r, time_to_reserve, is_gpu);

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "Anirudh Done reserving superresolution queue slot in rbtree for request");

    return NGX_OK;
}


static void
ngx_http_superresolution_watchdog_handler(ngx_event_t *ev)
{
    ngx_http_request_t *r = ev->data;
    ngx_http_superresolution_ctx_t *ctx;

    if (r == NULL) {
        return;
    }

    ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "superresolution watchdog handler called");

    ctx = ngx_http_get_module_ctx(r, ngx_http_superresolution_module);
    if (ctx == NULL) {
        return;
    }

    ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
                  "superresolution watchdog timeout, releasing queue slot");

    if (ngx_release_superresolution_queue_slot_in_ctx(r) != NGX_OK) {
        ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
                      "superresolution watchdog release failed");
    }
}

static void
ngx_http_superresolution_watchdog_cleanup(void *data)
{
    ngx_http_superresolution_ctx_t *ctx = data;

    if (ctx == NULL) {
        return;
    }

    if (ctx->watchdog.timer_set) {
        ngx_del_timer(&ctx->watchdog);
    }
}

void 
delete_watchdog_timer_from_subrequest(ngx_http_request_t *r){
  
    ngx_http_superresolution_ctx_t   *ctx;
    ctx = ngx_http_get_module_ctx(r, ngx_http_superresolution_module);
    
    if (ctx == NULL) {
        return;
    }

    ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "Deleting watchdog timer from subrequest");

    if (ctx->watchdog.timer_set) {
        ngx_del_timer(&ctx->watchdog);
    }
  
}

ngx_int_t
ngx_create_superresolution_ctx(ngx_http_request_t *r)
{

    ngx_int_t rc = NGX_OK;
    ngx_http_superresolution_ctx_t   *ctx;
    ctx = ngx_pcalloc(r->pool, sizeof(ngx_http_superresolution_ctx_t));

    if (ctx == NULL) {
        return NGX_ERROR;
    }

    // Initialize context fields
    ctx->in = NULL;
    ctx->time_reserved = 0;
    ctx->accounted = 0;
    ctx->watchdog_registered = 0;
    ctx->is_gpu = 0;

    ngx_http_set_ctx(r, ctx, ngx_http_superresolution_module);
    return rc;
}

ngx_http_superresolution_ctx_t *ngx_get_superresolution_ctx(ngx_http_request_t *r)
{
    ngx_http_superresolution_ctx_t   *ctx;
    ctx = ngx_http_get_module_ctx(r, ngx_http_superresolution_module);
    return ctx;
}
 

ngx_int_t
ngx_superresolution_queue_lock(ngx_superresolution_queue_fn fn, void *ctx)
{
    ngx_int_t rc = NGX_OK;

    if (queue_mutex == NULL || CPU_queue_size == NULL || GPU_queue_size == NULL || fn == NULL) {
        return NGX_ERROR;
    }

    ngx_shmtx_lock(queue_mutex);
    rc = fn(CPU_queue_size, GPU_queue_size, ctx);
    ngx_shmtx_unlock(queue_mutex);

    return rc;
}

 // Implement a watchdog timer here to prevent hanging subrequests. 
    // It should call ngx_release_superresolution_queue_slot_in_ctx after timeout.
    // Write code below.

void start_superresolution_watchdog_timer(ngx_http_request_t *r, ngx_int_t time_to_wait){
    ngx_pool_cleanup_t *cln;

    ngx_http_superresolution_ctx_t   *ctx;
    ctx = ngx_http_get_module_ctx(r, ngx_http_superresolution_module);

    ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
                   "Creattng superresolution watchdog timer");

    if (ctx == NULL) {
        return;
    }

    if (ctx->watchdog.timer_set) {
        ngx_del_timer(&ctx->watchdog);
    }

    ctx->watchdog.handler = ngx_http_superresolution_watchdog_handler;
    ctx->watchdog.data = r;
    ctx->watchdog.log = r->connection->log;

    if (!ctx->watchdog_registered) {
        cln = ngx_pool_cleanup_add(r->pool, 0);
        if (cln != NULL) {
            cln->handler = ngx_http_superresolution_watchdog_cleanup;
            cln->data = ctx;
        }
        ctx->watchdog_registered = 1;
    }

    ngx_add_timer(&ctx->watchdog, time_to_wait);
}

ngx_int_t
get_compute_cpu_queue_size(void)
{
    return (ngx_int_t) ngx_atomic_fetch_add(CPU_queue_size, 0);
}

ngx_int_t
get_compute_gpu_queue_size(void)
{
    return (ngx_int_t) ngx_atomic_fetch_add(GPU_queue_size, 0);
}


static ngx_int_t
ngx_http_superresolution_header_filter(ngx_http_request_t *r)
{

  r->content_size = r->headers_out.content_length_n;
  
  if (r->cache != NULL){
    ngx_http_clear_content_length(r);
  }
  
  return ngx_http_next_header_filter(r);
}


ngx_int_t
ngx_http_file_cache_name_main_key(ngx_http_request_t *r, ngx_str_t *name)
{
  u_char *p;
  ngx_http_cache_t *c = r->cache;
  ngx_path_t *path    = c->file_cache->path;
  
  name->len = path->name.len + 1 + path->len
    + 2 * NGX_HTTP_CACHE_KEY_LEN;

  name->data = ngx_pnalloc(r->pool, name->len + 1);
  if (name->data == NULL){
    return NGX_ERROR;
  }

  ngx_memcpy(name->data, path->name.data, path->name.len);
  
  p = name->data + path->name.len + 1 + path->len;
  p = ngx_hex_dump(p, c->main, NGX_HTTP_CACHE_KEY_LEN);
  *p = '\0';

  ngx_create_hashed_filename(path, name->data, name->len);  
  return NGX_OK;
}



static ngx_int_t
ngx_http_superresolution_scale_video_from_subrequest(ngx_http_request_t *sr, ngx_http_request_t *r, ngx_chain_t **in, size_t content_start){
  
  // Copy the above *in into a temporary buffer
  // cl now points to an onnx model
  ngx_int_t ret = NGX_OK;
  ngx_chain_t *cl = *in;
  
  // Copy ctx[in] into the above *in
  // in now points to video segment
  ngx_http_superresolution_ctx_t   *ctx;
  ctx = ngx_http_get_module_ctx(r, ngx_http_superresolution_module);
  *in = ctx->in; 

  u_char *buf;
  size_t buffer_size; 

  if (ctx->in->buf->in_file == 0){

    buffer_size = r->content_size;

    // Write code to copy from in->buf to buf
    buf = (u_char *) ngx_pcalloc(r->pool, buffer_size);
    u_char *curr = buf;

    ngx_uint_t copied = 0;
    ngx_int_t to_copy_bytes = 0;

    ngx_chain_t *t = ctx->in;

    while (copied <= buffer_size && t != NULL) {

      to_copy_bytes = t->buf->last - t->buf->pos;
      ngx_memcpy(curr, t->buf->pos, to_copy_bytes);
      copied += to_copy_bytes;
      curr += to_copy_bytes;
      t = t->next;

    }

    ret = ngx_http_superresolution_scale_video(sr, ctx->in, cl, buf, buffer_size, NULL, content_start);

  } 
  else {

    ngx_str_t cache_file_name;
    ret = ngx_http_file_cache_name_main_key(r, &cache_file_name);   
  
    // 3. Call ngx_http_superresolution_scale_video
    // Read the video content from file. Would this change?
  
    int n = 0;
    buffer_size = ctx->in->buf->file_last - ctx->in->buf->file_pos;
    buf = (u_char *) ngx_pcalloc(r->pool, buffer_size);//malloc(ctx->in->buf->file_last - ctx->in->buf->file_pos);
    n = ngx_read_file(ctx->in->buf->file, (u_char *) buf, buffer_size, ctx->in->buf->file_pos);

    if (n <= 0){
      return NGX_DECLINED;
    }
    ret = ngx_http_superresolution_scale_video(sr, ctx->in, cl, buf, buffer_size, &cache_file_name, content_start);
  }
  
  if (ret == NGX_DECLINED){
    return NGX_DECLINED;
  }

  return NGX_OK;
}


static ngx_int_t
ngx_http_superresolution_handler(ngx_http_request_t *r, ngx_chain_t *in)
{

  // This is the second level request i.e., sub sub request.
  if (r != r->main &&
      r->parent != r->main){

    if (r->subsubrequestdone == 1){
      return ngx_http_next_body_filter(r, in);
    }

    r->subsubrequestdone = 1;
    //ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "Sabby, subsubrequest!!");
    
    ngx_int_t ret = NGX_OK;
    ret = ngx_http_superresolution_delete_alternate_file(r);

    //ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "Sabby, subsubrequest %d!!", ret);
    
    if (ret == NGX_DECLINED){
      return ngx_http_next_body_filter(r, in);
    }
    
    return ngx_http_next_body_filter(r, in);
  }
  
  // This code is part of the sub-request.
  if (r != r->main &&
      r->parent != NULL){

      if(r->subrequestdone){
        return ngx_http_next_body_filter(r, in) ;
      }


      if (in->buf->in_file == 1 &&
	      r->upstream->cache_status == NGX_HTTP_CACHE_MISS){

        if (r->body_start == 0){
	        r->body_start = in->buf->file_pos;
        }

        ngx_chain_t *cl = in;
        while(cl->next != NULL){
	        cl = cl->next;
        }
      
        if (cl->buf->file_last >= r->content_size){

	        r->subrequestdone = 1;
		
	        ngx_int_t ret = ngx_http_superresolution_scale_video_from_subrequest(r, r->parent, &in, r->body_start);
		
	        if (ret == NGX_DECLINED){
	          return ngx_http_next_body_filter(r, in);
	        }
        }
        else {

	        u_char *buf = (u_char *) ngx_pcalloc(r->pool, sizeof(u_char));
	        *buf = 'A';

		
	        ngx_buf_t *tmpbuf;
	        tmpbuf = (ngx_buf_t *) ngx_pcalloc(r->pool, sizeof(ngx_buf_t));

	        if (tmpbuf == NULL) {
	          return NGX_DECLINED;
	        }

		
	        tmpbuf->pos = buf;
	        tmpbuf->last = tmpbuf->pos + sizeof(u_char);
	        tmpbuf->start = tmpbuf->pos;
	        tmpbuf->end = tmpbuf->last;
	        tmpbuf->last_buf = 1;
	        tmpbuf->memory = 1;
	        cl->buf = tmpbuf;         	
        }
    }
    else{
      r->subrequestdone = 1;
      ngx_int_t ret = ngx_http_superresolution_scale_video_from_subrequest(r, r->parent, &in, 0);  

      if (ret == NGX_DECLINED){
	      return ngx_http_next_body_filter(r, in);
      }
    }
    return ngx_http_next_body_filter(r, in);
  }
  
  if (r->main->internal) {    
    return ngx_http_next_body_filter(r, in) ;
  }

  // This code is part of the main request.
  // The main request (i) invokes a subrequest and (ii) clears the linked list of buffers  
  r->main->internal = 1;
  
  ngx_str_t            uri;
  ngx_http_request_t  *sr;
  ngx_int_t            rc;

  ngx_uint_t idx;
  
  /* if (in->buf != NULL && */
  /*     in->buf->in_file == 1 && */
  /*     r->upstream->cache_status == NGX_HTTP_CACHE_MISS) { */
  /*   ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "Sabby, internal request got all content!!"); */
  /* } */
  
  /*if(in->buf->in_file == 1 &&
     r->upstream &&
     r->upstream->cache_status == NGX_HTTP_CACHE_HIT &&
     r->cache &&
     r->cache->use_resolvable &&
     r == r->main){*/


  /*  
  if(in->buf->in_file == 1 &&   
      r->upstream &&
      ((r->cache &&  // Cache hit and resolve 
      r->cache->use_resolvable > 0 && 
      r->upstream->cache_status == NGX_HTTP_CACHE_HIT) || 
      (r->upstream->cache_status == NGX_HTTP_CACHE_MISS && r->resolvable_from_origin > 0)) && // Cache miss and resolve
      r == r->main){
  */
  
  ngx_http_superresolution_ctx_t   *ctx;
  ctx = ngx_http_get_module_ctx(r, ngx_http_superresolution_module);

  if (ctx == NULL){
    return ngx_http_next_body_filter(r, in);
  }

 if(((in->buf->in_file == 1 &&   
      r->upstream &&
      r->cache &&  // Cache hit and resolve 
      r->cache->use_resolvable > 0 && 
      r->upstream->cache_status == NGX_HTTP_CACHE_HIT) || 
      (r->upstream && r->upstream->cache_status == NGX_HTTP_CACHE_MISS && 
      r->resolvable_from_origin > 0)) && // Cache miss and resolve
      r == r->main &&
      ctx != NULL){

   
    // If it is a cache miss, then return next filter till it is the last buf
    if (in->buf->in_file == 1 &&
        r->upstream->cache_status == NGX_HTTP_CACHE_MISS) {

        if (r->body_start == 0) {
          r->body_start = in->buf->file_pos;
        }

        ngx_chain_t *cl = in;
        while(cl->next != NULL){
	        cl = cl->next;
        }

        if(cl->buf->file_last >= r->content_size) {
        } else {
          return ngx_http_next_body_filter(r, in); 
      }   
    } 
    else if(r->upstream->cache_status == NGX_HTTP_CACHE_MISS) {

      ngx_chain_t *cl = in;
      ngx_int_t total_bytes = 0;

      while (cl != NULL) {
        total_bytes += (cl->buf->last - cl->buf->pos);
        cl = cl->next;
      }

      if (total_bytes < r->content_size){
          return ngx_http_next_body_filter(r, in);
      }
    }

    
    ngx_uint_t scale_w = ngx_http_superresolution_get_scale(r);
    
    // Make a nginx subrequest if we need to upsample.
    //idx = (r->resolve_to_quality == 2) ? 1 : 0;
    idx = r->cache->use_resolvable - 1;

  if (r->headers_in.superresolution_models == NULL ||
    ((ngx_array_t *) r->headers_in.superresolution_models)->nelts <= idx || scale_w <= 100)
  {
    /* fallback */
    ngx_str_set(&uri, "/espcn.pb");
  } else {
      ngx_str_t *models = (ngx_str_t *) r->headers_in.superresolution_models->elts;
      ngx_str_t *src = &models[idx];

      if (src->len == 0 || src->data == NULL) {
        ngx_str_set(&uri, "/espcn.pb");
      } else {
          uri.len = src->len;
          uri.data = ngx_pnalloc(r->pool, uri.len);
          if (uri.data == NULL) {
            return NGX_ERROR;  /* or fallback */
          }
          ngx_memcpy(uri.data, src->data, uri.len);
      }
  }

  rc = ngx_http_subrequest(r, &uri, NULL, &sr, NULL, 0);
  if(rc == NGX_ERROR){
      return NGX_ERROR;
  }    

  if (in) {
    if (ngx_chain_add_copy(r->pool, &ctx->in, in) != NGX_OK) {
	      return NGX_ERROR;
      }
  }    

    u_char *buf = (u_char *) ngx_pcalloc(r->pool, sizeof(u_char));
    *buf = 'A';
    
    ngx_buf_t *tmpbuf;
    tmpbuf = (ngx_buf_t *) ngx_pcalloc(r->pool, sizeof(ngx_buf_t));

    if (tmpbuf == NULL) {
      return NGX_DECLINED;
    }

    tmpbuf->pos = buf;
    tmpbuf->last = tmpbuf->pos + sizeof(u_char);
    tmpbuf->start = tmpbuf->pos;
    tmpbuf->end = tmpbuf->last;
    tmpbuf->last_buf = 1;
    tmpbuf->memory = 1;
    in->buf = tmpbuf;         
  }

  return ngx_http_next_body_filter(r, in);  
}    

/*
static ngx_int_t
ngx_http_superresolution_init_shm(ngx_shm_zone_t *zone, void *data)
{
    ngx_slab_pool_t *shpool = (ngx_slab_pool_t *) zone->shm.addr;
    ngx_superresolution_shm_t  *q;


    if (shpool == NULL) {
        return NGX_ERROR;
    }

    if (data) {
        q = data;
    } else {
        q = ngx_slab_alloc(shpool, sizeof(ngx_superresolution_shm_t));
        if (q == NULL) {
            return NGX_ERROR;
        }
        q->cpu_queue = 0;
        q->gpu_queue = 0;
        q->rbtree_lock = 0;
        ngx_rbtree_init(&q->rbtree, &q->sentinel, ngx_superresolution_rbtree_insert);
    }

    CPU_queue_size = &q->cpu_queue;
    GPU_queue_size = &q->gpu_queue;
    queue_mutex = &shpool->mutex;

    ngx_shmtx_sh_t *rbtree_sh = (ngx_shmtx_sh_t *) &q->rbtree_lock;
    if (ngx_shmtx_create(&rbtree_mtx_storage, rbtree_sh, NULL) != NGX_OK) {
        return NGX_ERROR;
    }
    rbtree_mutex = &rbtree_mtx_storage;

    task_shpool = shpool;
    task_rbtree = &q->rbtree;
    task_sentinel = &q->sentinel;
    shpool->data = q;
    zone->data = (void *) q;

    return NGX_OK;
}*/

static ngx_int_t
ngx_http_superresolution_init(ngx_conf_t *cf)
{
 
    ngx_http_next_header_filter = ngx_http_top_header_filter;
    ngx_http_top_header_filter  = ngx_http_superresolution_header_filter;
    ngx_http_next_body_filter   = ngx_http_top_body_filter;
    ngx_http_top_body_filter    = ngx_http_superresolution_handler;

    // Zone 1: queue sizes ONLY (small)
    ngx_str_t queue_shm_name = ngx_string("superresolution_queue_shm");
    size_t queue_shm_size = ngx_align(64*1024 + sizeof(ngx_slab_pool_t), ngx_pagesize);
    ngx_shm_zone_t *queue_zone = ngx_shared_memory_add(cf, &queue_shm_name,
                                     queue_shm_size, &ngx_http_superresolution_module);
    if (queue_zone == NULL) return NGX_ERROR;
    queue_zone->init = ngx_http_superresolution_init_queue_shm;

    // Zone 2: rbtree task nodes ONLY
    ngx_str_t task_shm_name = ngx_string("superresolution_task_shm");
    size_t task_shm_size = ngx_align(128*1024 + sizeof(ngx_slab_pool_t) +
                               10000 * sizeof(ngx_superresolution_task_t), ngx_pagesize);
    ngx_shm_zone_t *task_zone = ngx_shared_memory_add(cf, &task_shm_name,
                                     task_shm_size, &ngx_http_superresolution_module);
    if (task_zone == NULL) return NGX_ERROR;
    task_zone->init = ngx_http_superresolution_init_task_shm;

    return NGX_OK;

}


static ngx_int_t
ngx_http_superresolution_init_worker(ngx_cycle_t *cycle)
{
    /* If a previous worker crashed while holding shared memory state,
     * reset everything so this worker starts clean.
     * Safe with worker_processes 1. */

    if (CPU_queue_size) *CPU_queue_size = 0;
    if (GPU_queue_size) *GPU_queue_size = 0;

    if (rbtree_lock_ptr) *rbtree_lock_ptr = 0;
    if (rbtree_wait_ptr) *rbtree_wait_ptr = 0;

    if (task_rbtree && task_sentinel) {
        ngx_memzero(task_sentinel, sizeof(ngx_rbtree_node_t));
        ngx_rbtree_init(task_rbtree, task_sentinel, ngx_superresolution_rbtree_insert);
    }

    return NGX_OK;
}

static ngx_http_module_t ngx_http_superresolution_module_ctx = {
  NULL,     /* preconfiguration */
  ngx_http_superresolution_init,              /* postconfiguration */
  NULL,                                       /* create main configuration */
  NULL,                                       /* init main configuration */
  NULL,                                       /* create server configuration */
  NULL,                                       /* merge server configuration */
  NULL,                                       /* create location configuration */
  NULL
};


ngx_module_t ngx_http_superresolution_module = {
  NGX_MODULE_V1,
  &ngx_http_superresolution_module_ctx, /* module context */
  NULL,                            /* module directives */
  NGX_HTTP_MODULE,                 /* module type */
  NULL,                            /* init master */
  NULL, /* init module */
  ngx_http_superresolution_init_worker, /* init process */
  NULL,                            /* init thread */
  NULL,                            /* exit thread */
  NULL,                            /* exit process */
  NULL,                            /* exit master */
  NGX_MODULE_V1_PADDING
};
