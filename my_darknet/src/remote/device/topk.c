#include "topk.h"

void top_k(float* possibilities, int classes, int k, int* top_k_index)
{
  int i,j;
  for (j = 0 ; j < k ; ++j) top_k_index[j] = -1;  // Initialize
  for (i = 0 ; i < classes ; ++i) {
    int curr = i;
    for (j = 0 ; j < k ; ++j) {
      if ((top_k_index[j] < 0) || possibilities[curr] > possibilities[top_k_index[j]]) {
        int swap = curr;
        curr = top_k_index[j];
        top_k_index[j] = swap;
      }
    }
  }
}

void sort_top_k(float* top_k, int* top_k_idx, int k)
{
  int i,j;
  for (i = k-1 ; i > 0 ; --i) {
    for (j = 0 ; j < i ; ++j) {
      if (top_k[j] < top_k[j+1]) {
        float swap = top_k[j];
        top_k[j] = top_k[j+1];
        top_k[j+1] = swap;
        int swap_idx = top_k_idx[j];
        top_k_idx[j] = top_k_idx[j+1];
        top_k_idx[j+1] = swap_idx;
      }
    }
  }
}
