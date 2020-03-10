#ifndef BATCHNORM_H_
#define BATCHNORM_H_

typedef struct batchnorm_factors {
  float scale;
  float mean;
  float variance;
  float bias;
} batch_ft;

#endif
