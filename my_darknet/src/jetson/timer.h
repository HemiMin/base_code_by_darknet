#ifndef TIMER_H_
#define TIMER_H_

#if defined(CUDA) || defined(CUBLAS)

#ifdef __cplusplus
extern "C" {
#endif
void init_timer(void);
void init_local_timer(void);
void start_timer(void);
void stop_timer(float* ms);
void start_local_timer(void);
void stop_local_timer(float* ms);
void free_timer(void);
void free_local_timer(void);
#ifdef __cplusplus
}
#endif

#endif

#endif
