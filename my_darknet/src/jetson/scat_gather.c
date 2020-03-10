#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "scat_gather.h"

struct iovec {
  void*  iov_base;
  size_t iov_len;
};

inline int get_index_2d(int start_idx,
                        int ldw,
                        int h, int w)
{
  return start_idx + w + ldw*h;
}

inline int get_index_3d(int start_idx,
                        int ldh, int ldw,
                        int c, int h, int w)
{
  return start_idx + w + ldw*(h + ldh*c);
}

inline int get_index_4d(int start_idx,
                        int ldc, int ldh, int ldw, 
                        int n, int c, int h, int w)
{
  return start_idx + w + ldw*(h + ldh*(c + ldc*n));
}

size_t scatter(scalar_t* src, struct iovec* iov, size_t count)
{
  size_t i, size=0;
  scalar_t* ptr = src;
  for (i = 0 ; i < count ; ++i) {
    memcpy(iov[i].iov_base, ptr, sizeof(scalar_t)*iov[i].iov_len);
    size += sizeof(scalar_t)*iov[i].iov_len;
    ptr += iov[i].iov_len;
  }
  return size;
}

size_t gather(scalar_t* dst, struct iovec* iov, size_t count)
{
  size_t i, size=0;
  scalar_t* ptr = dst;
  for (i = 0 ; i < count ; ++i) {
    memcpy(ptr, iov[i].iov_base, sizeof(scalar_t)*iov[i].iov_len);
    size += sizeof(scalar_t)*iov[i].iov_len;
    ptr += iov[i].iov_len;
  }
  return size;
}

size_t tile_cpy4d(scalar_t* dst, scalar_t* src, int start_idx,
                  int ldn,int ldc,int ldh,int ldw, int n,int c,int h,int w,
                  int flag)
{
  int concate_level = 0;
  int itr = 0;
  int count = 0;
  struct iovec* iov;
  size_t size;

  scalar_t* scat_ptr = NULL;
  //NOTE Assume NCHW layout.
  if (w == ldw) {
    concate_level++;
    if (h == ldh) {
      concate_level++;
      if (c == ldc) {
        concate_level++;
      }
    }
  }

  if (flag == HOST_TO_DEVICE) { // Gather.
    scat_ptr = src;
  } else
  if (flag == DEVICE_TO_HOST) { // Scatter.
    scat_ptr = dst;
  } else {
    fprintf(stderr, "Invalid flag(<2): %d\n", flag);
    exit(0);
  }

  if (concate_level == 0) {
    count = h*c*n;
    iov = (struct iovec*)calloc(count, sizeof(struct iovec));
    for (itr = 0 ; itr < count ; ++itr) {
      int h_idx = itr % h;
      int c_idx = (itr / h) % c;
      int n_idx = itr / h / c;

      iov[itr].iov_base = scat_ptr + get_index_4d(start_idx,
                                                  ldc, ldh, ldw,
                                                  n_idx, c_idx, h_idx, 0);
      iov[itr].iov_len = (size_t)w;
    }
  } else
  if (concate_level == 1) {
    count = c*n;
    iov = (struct iovec*)calloc(count, sizeof(struct iovec));
    for (itr = 0 ; itr < count ; ++itr) {
      int c_idx = itr % c;
      int n_idx = itr / c;

      iov[itr].iov_base = scat_ptr + get_index_4d(start_idx,
                                                  ldc, ldh, ldw,
                                                  n_idx, c_idx, 0, 0);
      iov[itr].iov_len = (size_t)w*h;
    }
  } else
  if (concate_level == 2) {
    count = n;
    iov = (struct iovec*)calloc(count, sizeof(struct iovec));
    for (itr = 0 ; itr < count ; ++itr) {
      int n_idx = itr;

      iov[itr].iov_base = scat_ptr + get_index_4d(start_idx,
                                                  ldc, ldh, ldw,
                                                  n_idx, 0, 0, 0);
      iov[itr].iov_len = (size_t)w*h*c;
    }
  } else
  if (concate_level == 3) {
    count = 1;
    iov = (struct iovec*)calloc(count, sizeof(struct iovec));
    iov[itr].iov_base = scat_ptr + start_idx;
    iov[itr].iov_len = (size_t)w*h*c*n;
  } else {
    fprintf(stderr, "Invalid concate level(<4): %d\n", concate_level);
    exit(0);
  }

  if (flag == HOST_TO_DEVICE) { // Gather.
    size = gather(dst, iov, count);
  } else
  if (flag == DEVICE_TO_HOST) { // Scatter.
    size = scatter(src, iov, count);
  } else {
    fprintf(stderr, "Invalid flag(<2): %d\n", flag);
    exit(0);
  }
  free(iov);

  return size;
}
