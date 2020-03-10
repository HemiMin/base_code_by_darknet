#ifndef TOPK_H_
#define TOPK_H_

void top_k(float* possibilities, int classes, int k, int* top_k_index);
void sort_top_k(float* top_k, int* top_k_idx, int k);

#endif
