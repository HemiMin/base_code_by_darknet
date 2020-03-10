#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NAME_STR_SIZE 16
#define CLASS_CNT 1000

typedef enum {
  LOGISTIC=0, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum {
  CROP=0, CONV, BATCHNORM, ACTIV, MAXPOOL, RESIDUAL, ROUTE, BIAS, CONNECTED, AVGPOOL, SOFTMAX
} LAYER;

struct crop {
  int h; // crop height
  int w; // crop width
};

struct convolution {
  int c; // input channel
  int w; // input width
  int h; // input height
  int size; // kernel width and height
  int n; // output channel
  int stride;
  int padding;
};

struct batchnorm {
  int c; // channel
  int h; // height
  int w; // width
};

struct bias {
  int n; // batch
  int c; // channel
  int h; // height
  int w; // width
};

struct maxpool {
  int c; // channel
  int h; // height
  int w; // width
  int size; // kernel size
  int stride;
  int padding;
};

struct activation {
  ACTIVATION act;
  int n; // batch
  int c; // channel
  int h; // height
  int w; // width
};

struct residual {
  int index; // forward index
  int batch;
  int c1; // forward channel
  int h1; // forward height
  int w1; // forward width
  int c2; // output channel
  int h2; // output height
  int w2; // output width
};

struct route {
  int type; // 1: forward from 2 layers before  2: forward from both 1 and 2 layers before
  int batch;
  size_t size_1;
  size_t size_2;
};

struct avgpool {
  int c; // channel
  int h; // height
  int w; // width
};

struct connected {
  int inputs;
  int outputs;
};

struct softmax {
  float temp; // temperature
  int size;
};

struct node {
  LAYER layer;
  void* e;
  struct node* next;
};

struct names {
  size_t cnt;
  char** name;
};

struct name_list {
  struct names conv;
  struct names maxpool;
  struct names residual;
  struct names route;
  struct names avgpool;
  struct names connected;
};

char** parse_names(FILE* fp, size_t* num);
void remove_newline(char* str, size_t size);

struct node* parse_config(char* cfg_file);
struct crop* parse_crop(FILE* fp);
struct convolution* parse_convolution(FILE* fp);
struct batchnorm* parse_batchnorm(FILE* fp);
struct bias* parse_bias(FILE* fp);
struct maxpool* parse_maxpool(FILE* fp);
struct activation* parse_activation(FILE* fp);
struct residual* parse_residual(FILE* fp);
struct route* parse_route(FILE* fp);
struct avgpool* parse_avgpool(FILE* fp);
struct connected* parse_connected(FILE* fp);
struct softmax* parse_softmax(FILE* fp);

void preprocess(void);
void prototype(struct name_list l_name);
void caution(void);
void load_img(struct node* head);
void memory_alloc(struct node* head, struct name_list l_name);
void read_weights(struct node* head, struct name_list l_name);
void erupt_network_code(struct node* head, struct name_list l_name);

void erupt_crop_code(struct crop* crop);
void erupt_convolution_code(struct convolution* conv, char* input, char* output);
void erupt_batchnorm_code(struct batchnorm* bn, char* input, char* output);
void erupt_bias_code(struct bias* bs, char* input, char* output);
void erupt_bias_connected_code(struct bias* bs, char* input, char* output);
void erupt_maxpool_code(struct maxpool* mxp, char* input, char* output);
void erupt_activation_code(struct activation* act, char* input, char* output);
void erupt_residual_code(struct residual* res, char* input1, char* input2, char* output);
void erupt_route_code(struct route* rt, char* input1, char* input2, char* output);
void erupt_avgpool_code(struct avgpool* avg, char* input, char* output);
void erupt_connected_code(struct connected* conn, char* input, char* output);
void erupt_softmax_code(struct softmax* smx, char* input, char* output);

void erupt_topk(int class_cnt);

void deallocate(struct node* head, struct name_list l_name);

char *get_activation_string(ACTIVATION a);
ACTIVATION get_activation(char *s);

int main(int argc, char* argv[])
{
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <config file> <name file> >> <c file>\n", argv[0]);
    exit(0);
  }

  int i;
  FILE* names_fp = fopen(argv[2], "r");
  char tag[64];

  struct name_list l_name = {
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0}
                            };

  while (fgets(tag, 64, names_fp) != NULL) {
    if (strcmp(tag, "\n") == 0) continue;
    if (strcmp(tag, "[convolution]\n") == 0) {
      l_name.conv.name = parse_names(names_fp, &(l_name.conv.cnt));
    } else
    if (strcmp(tag, "[max_pool]\n") == 0) {
      l_name.maxpool.name = parse_names(names_fp, &(l_name.maxpool.cnt));
    } else
    if (strcmp(tag, "[residual]\n") == 0) {
      l_name.residual.name = parse_names(names_fp, &(l_name.residual.cnt));
    } else
    if (strcmp(tag,  "[route]\n") == 0) {
      l_name.route.name = parse_names(names_fp, &(l_name.route.cnt));
    } else
    if (strcmp(tag, "[average_pool]\n") == 0) {
      l_name.avgpool.name = parse_names(names_fp, &(l_name.avgpool.cnt));
    } else
    if (strcmp(tag, "[connected]\n") == 0) {
      l_name.connected.name = parse_names(names_fp, &(l_name.connected.cnt));
    } else {
      fprintf(stderr, "Invalid name tag in name file: %s\n", tag);
      exit(0);
    }
  }

  struct node* layers=NULL;
  layers = parse_config(argv[1]);

  // Generate codes.
  preprocess();
  prototype(l_name);

  printf("int main(int argc, char* argv[])\n");
  printf("{\n");

  caution();

  printf("  int top_K = atoi(argv[3]);\n\n");

  load_img(layers);

  printf("  FILE* wt_fp = fopen(argv[2], \"rb\");\n");
  printf("  if (wt_fp == NULL) {\n");
  printf("    fprintf(stderr, \"File %%s is not opened.\\n\", argv[2]);\n");
  printf("    exit(0);\n");
  printf("  }\n\n");

  memory_alloc(layers, l_name);
  read_weights(layers, l_name);

  printf("  fclose(wt_fp);\n\n");

  printf("#ifdef PLANNER\n");
  printf("#if defined(CUDA) || defined(CUBLAS)\n");
  printf("  scalar_t *in_mem_0=NULL, *wt_mem_0=NULL, *ot_mem_0=NULL;\n");
  printf("  cudaMallocWrapper((void**)&in_mem_0, sizeof(scalar_t)*IN_MEM_SIZE);\n");
  printf("  cudaMallocWrapper((void**)&wt_mem_0, sizeof(scalar_t)*WT_MEM_SIZE);\n");
  printf("  cudaMallocWrapper((void**)&ot_mem_0, sizeof(scalar_t)*OT_MEM_SIZE);\n");
  printf("#else\n");
  printf("  scalar_t* in_mem_0 = (scalar_t*)calloc(IN_MEM_SIZE, sizeof(scalar_t));\n");
  printf("  scalar_t* wt_mem_0 = (scalar_t*)calloc(WT_MEM_SIZE, sizeof(scalar_t));\n");
  printf("  scalar_t* ot_mem_0 = (scalar_t*)calloc(OT_MEM_SIZE, sizeof(scalar_t));\n");
  printf("#endif\n");
  printf("#endif\n\n");

  printf("#if defined(CUDA) || defined(CUBLAS)\n");
  printf("  init_timer();\n");
  printf("#else\n");
  printf("  clock_t start, end;\n\n");
  printf("#endif\n\n");
  printf("#ifdef TIME_ESTIMATE\n");
  printf("  clock_t est_time;\n");
  printf("#endif\n\n");

  printf("  // Run Network\n");
  printf("#if defined(CUDA) || defined(CUBLAS)\n");
  printf("  start_timer();\n");
  printf("#else\n");
  printf("  start = clock();\n\n");
  printf("#endif\n\n");

  erupt_network_code(layers, l_name);

  printf("#if defined(CUDA) || defined(CUBLAS)\n");
  printf("  float elapsed_time_ms;\n");
  printf("  stop_timer(&elapsed_time_ms);\n");
  printf("  printf(\"Elapsed Time (GPU): %%.3f msec\\n\", elapsed_time_ms);\n");
  printf("#else\n");
  printf("  end = clock();\n");
  printf("  printf(\"Elapsed Time (CPU): %%.3f msec\\n\", (float)(end-start)/CLOCKS_PER_SEC*1000);\n");
  printf("#endif\n\n");

  erupt_topk(CLASS_CNT);

  deallocate(layers, l_name);

  printf("  return 0;\n");
  printf("}\n");

  for (i = 0 ; i < l_name.conv.cnt ; ++i)      free(l_name.conv.name[i]);
  for (i = 0 ; i < l_name.maxpool.cnt ; ++i)   free(l_name.maxpool.name[i]);
  for (i = 0 ; i < l_name.residual.cnt ; ++i)  free(l_name.residual.name[i]);
  for (i = 0 ; i < l_name.avgpool.cnt ; ++i)   free(l_name.avgpool.name[i]);
  for (i = 0 ; i < l_name.connected.cnt ; ++i) free(l_name.connected.name[i]);
  free(l_name.conv.name);
  free(l_name.maxpool.name);
  free(l_name.residual.name);
  free(l_name.avgpool.name);
  free(l_name.connected.name);
  struct node *cur, *next;
  cur = layers;
  next = layers->next;
  int cnt = 0;
  while (1) {
    free(cur->e);
    free(cur);
    cur = next;
    if (cur == NULL) break;
    next = cur->next;
  }

  return 0;
}

char** parse_names(FILE* fp, size_t* num)
{
  char** names;
  int i;
  fscanf(fp, "%ld\n", num);

  names = (char**)calloc(*num, sizeof(char*));
  for (i = 0 ; i < *num ; ++i) {
    names[i] = (char*)calloc(NAME_STR_SIZE, sizeof(char));
    char name[NAME_STR_SIZE];
    fgets(name, NAME_STR_SIZE, fp);
    remove_newline(name, NAME_STR_SIZE);
    strncpy(names[i], name, NAME_STR_SIZE);
  }

  return names;
}

void remove_newline(char* str, size_t size)
{
  int i=0;
  while (i < size) {
    if (str[i] == '\0') break;
    if (str[i] == '\n') {
      str[i] = '\0';
      break;
    }
    i++;
  }
}

struct node* parse_config(char* cfg_file)
{
  FILE* fp = fopen(cfg_file, "r");
  char tag[64];
  struct node *head=NULL, *tail=NULL;

  while(fgets(tag, 64, fp) != NULL) {
    if (strcmp(tag, "\n") == 0) continue;
    struct node* nd = (struct node*)malloc(sizeof(struct node));
    if (strcmp(tag, "[crop]\n") == 0) {
      struct crop* e = (struct crop*)malloc(sizeof(struct crop));
      e = parse_crop(fp);

      nd->layer = CROP;
      nd->e = e;
      nd->next = NULL;
    } else
    if (strcmp(tag, "[convolution]\n") == 0) {
      struct convolution* e = (struct convolution*)malloc(sizeof(struct convolution));
      e = parse_convolution(fp);
      
      nd->layer = CONV;
      nd->e = e;
      nd->next = NULL;
    } else
    if (strcmp(tag, "[batch_normalize]\n") == 0) {
      struct batchnorm* e = (struct batchnorm*)malloc(sizeof(struct batchnorm));
      e = parse_batchnorm(fp);

      nd->layer = BATCHNORM;
      nd->e = e;
      nd->next = NULL;
    } else
    if (strcmp(tag, "[bias]\n") == 0) {
      struct bias* e = (struct bias*)malloc(sizeof(struct bias));
      e = parse_bias(fp);

      nd->layer = BIAS;
      nd->e = e;
      nd->next = NULL;
    } else
    if (strcmp(tag, "[max_pool]\n") == 0) {
      struct maxpool* e = (struct maxpool*)malloc(sizeof(struct maxpool));
      e = parse_maxpool(fp);

      nd->layer = MAXPOOL;
      nd->e = e;
      nd->next = NULL;
    } else
    if (strcmp(tag, "[activation]\n") == 0) {
      struct activation* e = (struct activation*)malloc(sizeof(struct activation));
      e = parse_activation(fp);

      nd->layer = ACTIV;
      nd->e = e;
      nd->next = NULL;
    } else
    if (strcmp(tag, "[residual]\n") == 0) {
      struct residual* e = (struct residual*)malloc(sizeof(struct residual));
      e = parse_residual(fp);

      nd->layer = RESIDUAL;
      nd->e = e;
      nd->next = NULL;
    } else
    if (strcmp(tag, "[route]\n") == 0) {
      struct route* e = (struct route*)malloc(sizeof(struct route));
      e = parse_route(fp);

      nd->layer = ROUTE;
      nd->e = e;
      nd->next = NULL;
    } else
    if (strcmp(tag, "[average_pool]\n") == 0) {
      struct avgpool* e = (struct avgpool*)malloc(sizeof(struct avgpool));
      e = parse_avgpool(fp);

      nd->layer = AVGPOOL;
      nd->e = e;
      nd->next = NULL;
    } else
    if (strcmp(tag, "[fully_connected]\n") == 0) {
      struct connected* e = (struct connected*)malloc(sizeof(struct connected));
      e = parse_connected(fp);

      nd->layer = CONNECTED;
      nd->e = e;
      nd->next = NULL;
    } else
    if (strcmp(tag, "[softmax]\n") == 0) {
      struct softmax* e = (struct softmax*)malloc(sizeof(struct softmax));
      e = parse_softmax(fp);

      nd->layer = SOFTMAX;
      nd->e = e;
      nd->next = NULL;
    } else {
      fprintf(stderr, "Invalid config tag: %s\n", tag);
      free(nd);
      exit(0);
    }

    if (head == NULL) {
      head = nd;
      tail = nd;
    } else {
      tail->next = nd;
      tail = nd;
    }
  }
  return head;
}

struct crop* parse_crop(FILE* fp)
{
  struct crop* ret = (struct crop*)malloc(sizeof(struct crop));

  fscanf(fp, "(%d,%d)  # h,w\n", &(ret->h), &(ret->w));

  return ret;
}

struct convolution* parse_convolution(FILE* fp)
{
  struct convolution* ret = (struct convolution*)malloc(sizeof(struct convolution));

  fscanf(fp, "input(%d,%d,%d)  # input c*h*w.\n", &(ret->c), &(ret->h), &(ret->w));
  fscanf(fp, "weight(%d)  # kernel size.\n", &(ret->size));
  fscanf(fp, "output(%d)  # output channel.\n", &(ret->n));
  fscanf(fp, "stride:%d\n", &(ret->stride));
  fscanf(fp, "padding:%d\n", &(ret->padding));

  return ret;
}

struct batchnorm* parse_batchnorm(FILE* fp)
{
  struct batchnorm* ret = (struct batchnorm*)malloc(sizeof(struct batchnorm));
  
  fscanf(fp, "(%d,%d,%d)  # c*h*w\n", &(ret->c), &(ret->h), &(ret->w));

  return ret;
}

struct bias* parse_bias(FILE* fp)
{
  struct bias* ret = (struct bias*)malloc(sizeof(struct bias));

  fscanf(fp, "(%d,%d,%d,%d)  # n*c*h*w\n", &(ret->n), &(ret->c), &(ret->h), &(ret->w));

  return ret;
}

struct maxpool* parse_maxpool(FILE* fp)
{
  struct maxpool* ret = (struct maxpool*)malloc(sizeof(struct maxpool));

  fscanf(fp, "(%d,%d,%d)  # c*h*w\n", &(ret->c), &(ret->h), &(ret->w));
  fscanf(fp, "kernel(%d)  # kernel width and height\n", &(ret->size));
  fscanf(fp, "stride(%d)\n", &(ret->stride));
  fscanf(fp, "padding(%d)\n", &(ret->padding));

  return ret;
}

struct activation* parse_activation(FILE* fp)
{
  struct activation* ret = (struct activation*)malloc(sizeof(struct activation));
  char act_name[16];

  fgets(act_name, 16, fp);
  remove_newline(act_name, 16);
  ret->act = get_activation(act_name);
  fscanf(fp, "(%d,%d,%d,%d)  # n*c*h*w\n", &(ret->n), &(ret->c), &(ret->h), &(ret->w));

  return ret;
}

struct residual* parse_residual(FILE* fp)
{
  struct residual* ret = (struct residual*)malloc(sizeof(struct residual));

  fscanf(fp, "forward layer index:%d\n", &(ret->index));
  fscanf(fp, "batch:%d\n", &(ret->batch));
  fscanf(fp, "forward:(%d,%d,%d)  # c*h*w\n", &(ret->c1), &(ret->h1), &(ret->w1));
  fscanf(fp, "output:(%d,%d,%d)  # c*h*w\n", &(ret->c2), &(ret->h2), &(ret->w2));

  return ret;
}

struct route* parse_route(FILE* fp)
{
  struct route* ret = (struct route*)malloc(sizeof(struct route));
  
  fscanf(fp, "routing_cnt:%d\n", &(ret->type));
  fscanf(fp, "batch:%d\n", &(ret->batch));
  fscanf(fp, "input_size:%ld %ld \n", &(ret->size_1), &(ret->size_2));

  return ret;
}

struct avgpool* parse_avgpool(FILE* fp)
{
  struct avgpool* ret = (struct avgpool*)malloc(sizeof(struct avgpool));

  fscanf(fp, "(%d,%d,%d)  # c*h*w\n", &(ret->c), &(ret->h), &(ret->w));

  return ret;
}

struct connected* parse_connected(FILE* fp)
{
  struct connected* ret = (struct connected*)malloc(sizeof(struct connected));

  fscanf(fp, "input(%d)\n", &(ret->inputs));
  fscanf(fp, "output(%d)\n", &(ret->outputs));

  return ret;
}

struct softmax* parse_softmax(FILE* fp)
{
  struct softmax* ret = (struct softmax*)malloc(sizeof(struct softmax));

  fscanf(fp, "temperature:%f\n", &(ret->temp));
  fscanf(fp, "size:%d\n", &(ret->size));

  return ret;
}

void preprocess(void)
{
  printf("#include <stdio.h>\n");
  printf("#include <stdlib.h>\n");
  printf("#include <time.h>\n");
  printf("\n");
  printf("#include \"ops.h\"\n");
  printf("#include \"type.h\"\n");
  printf("#include \"image.h\"\n");
  printf("#include \"classifier.h\"\n");
  printf("#include \"topk.h\"\n");
  printf("#include \"crop.h\"\n");
  printf("#if defined(CUDA) || defined(CUBLAS)\n");
  printf("#include \"timer.h\"\n");
  printf("#endif\n");
  printf("\n");
  printf("#define  TOPK  5\n");
  printf("#define  IN_MEM_SIZE 524288\n");
  printf("#define  WT_MEM_SIZE 262144\n");
  printf("#define  OT_MEM_SIZE 524288\n");
  printf("\n");
}

void prototype(struct name_list l_name)
{
  size_t i;
  char** conv_names = l_name.conv.name;
  size_t conv_names_cnt = l_name.conv.cnt;

  printf("#ifdef PLANNER\n");
  for (i = 0 ; i < conv_names_cnt ; ++i) {
    printf("extern void %s( scalar_t*, scalar_t*, scalar_t*,\n", conv_names[i]);
    printf("                int, int, int, int, int,\n");
    printf("                int, int, int, int, int,\n");
    printf("                scalar_t*, scalar_t*, scalar_t*);\n");
  }
  printf("#endif\n\n");
}

void caution(void)
{
  printf("  if (argc != 4) {\n");
  printf("    fprintf(stderr, \"Usage: %%s <image file path>.jpg <weights file path>.weights <top k>\", argv[0]);\n");
  printf("    exit(0);\n");
  printf("  }\n\n");
}

void load_img(struct node* head)
{
  int c=3,h=256,w=256;
  if (head->layer == CONV) {
    struct convolution* e = (struct convolution*)head->e;
    c = e->c;
    h = e->h;
    w = e->w;
  }

  printf("  // Image load\n");
  printf("  image im = load_image(argv[1], %d, %d, %d);\n", w, h, c);
}

void memory_alloc(struct node* head, struct name_list l_name)
{
  printf("  // Memory Allocation\n");
  struct node* cur = head;
  int conv_idx = 0;
  int maxpool_idx = 0;
  int residual_idx = 0;
  int route_idx = 0;
  int connected_idx = 0;
  int avgpool_idx = 0;
  while (cur != NULL) {
    if (cur->layer == CROP) {
      // nothing.
    } else
    if (cur->layer == CONV) {
      struct convolution* e = (struct convolution*)cur->e;
      printf("  scalar_t* wt_%s = (scalar_t*)calloc(%d*%d*%d*%d, sizeof(scalar_t));\n", l_name.conv.name[conv_idx],
                                                                                        e->size, e->size, e->c, e->n);
      struct node* next = cur->next;
      if (next->layer == BATCHNORM) {
        cur = next;
        struct batchnorm* ee = (struct batchnorm*)cur->e;
        printf("  batch_ft* bt_%s = (batch_ft*)calloc(%d, sizeof(batch_ft));\n", l_name.conv.name[conv_idx], ee->c);
        printf("  scalar_t* ot_%s = (scalar_t*)calloc(%d*%d*%d, sizeof(scalar_t));\n", l_name.conv.name[conv_idx],
                                                                                        ee->w, ee->h, ee->c);
      } else
      if (next->layer == BIAS) {
        cur = next;
        struct bias* ee = (struct bias*)cur->e;
        printf("  scalar_t* bs_%s = (scalar_t*)calloc(%d, sizeof(scalar_t));\n", l_name.conv.name[conv_idx], ee->c);
        printf("  scalar_t* ot_%s = (scalar_t*)calloc(%d*%d*%d, sizeof(scalar_t));\n", l_name.conv.name[conv_idx],
                                                                                        ee->w, ee->h, ee->c);
      }
      conv_idx++;
    } else
    if (cur->layer == ACTIV) {
      // nothing.
    } else
    if (cur->layer == MAXPOOL) {
      struct maxpool* e = (struct maxpool*)cur->e;
      int out_w = (e->w + e->padding - e->size) / e->stride + 1;
      int out_h = (e->h + e->padding - e->size) / e->stride + 1;
      printf("  scalar_t* ot_%s = (scalar_t*)calloc(%d*%d*%d, sizeof(scalar_t));\n", l_name.maxpool.name[maxpool_idx++], 
                                                                                      out_w, out_h, e->c);
    } else
    if (cur->layer == RESIDUAL) {
      struct residual* e = (struct residual*)cur->e;
      printf("  scalar_t* ot_%s = (scalar_t*)calloc(%d*%d*%d, sizeof(scalar_t));\n", l_name.residual.name[residual_idx++],
                                                                                      e->w2, e->h2, e->c2);
    } else
    if (cur->layer == ROUTE) {
      struct route* e = (struct route*)cur->e;
      if (e->type == 1) {
        printf("  scalar_t* ot_%s = (scalar_t*)calloc(%d*%ld, sizeof(scalar_t));\n", l_name.route.name[route_idx++], 
                                                                                    e->batch, e->size_1);
      } else
      if (e->type == 2) {
        printf("  scalar_t* ot_%s = (scalar_t*)calloc(%d*(%ld+%ld), sizeof(scalar_t));\n", l_name.route.name[route_idx++],
                                                                                        e->batch, e->size_1, e->size_2); 
      } else {
        fprintf(stderr, "Invalid route type in memory allocation: %d\n", e->type);
      }
    } else
    if (cur->layer == CONNECTED) {
      struct connected* e = (struct connected*)cur->e;
      printf("  scalar_t* wt_%s = (scalar_t*)calloc(%d*%d, sizeof(scalar_t));\n", l_name.connected.name[connected_idx],
                                                                                  e->inputs, e->outputs);
      struct node* next = cur->next;
      if (next->layer == BATCHNORM) {
        cur = next;
        struct batchnorm* ee = (struct batchnorm*)cur->e;
        printf("  batch_ft* bt_%s = (batch_ft*)calloc(%d, sizeof(batch_ft));\n", l_name.connected.name[connected_idx],
                                                                                  ee->c);
        printf("  scalar_t* ot_%s = (scalar_t*)calloc(%d*%d*%d, sizeof(scalar_t));\n", l_name.connected.name[connected_idx],
                                                                                        ee->w, ee->h, ee->c);
      } else
      if (next->layer == BIAS) {
        cur = next;
        struct bias* ee = (struct bias*)cur->e;
        printf("  scalar_t* bs_%s = (scalar_t*)calloc(%d, sizeof(scalar_t));\n", l_name.connected.name[connected_idx],
                                                                                  ee->c);
        printf("  scalar_t* ot_%s = (scalar_t*)calloc(%d*%d*%d, sizeof(scalar_t));\n", l_name.connected.name[connected_idx],
                                                                                        ee->w, ee->h, ee->c);
      }
      connected_idx++;
    } else
    if (cur->layer == AVGPOOL) {
      struct avgpool* e = (struct avgpool*)cur->e;
      printf("  scalar_t* ot_%s = (scalar_t*)calloc(%d, sizeof(scalar_t));\n", l_name.avgpool.name[avgpool_idx], e->c);
      avgpool_idx++;
    } else
    if (cur->layer == SOFTMAX) {
      struct softmax* e = (struct softmax*)cur->e;
      printf("  float* ot_softmax = (float*)calloc(%d, sizeof(float));\n", e->size);
    } else {
      fprintf(stderr, "Invalid layer number: %d\n", cur->layer);
      exit(0);
    }
    cur = cur->next;
  }
  printf("\n");
}

void read_weights(struct node* head, struct name_list l_name)
{
  printf("  // Read weights and bias and batch normalization factors from file\n");
  struct node* cur = head;
  int conv_idx=0;
  int connected_idx=0;
  while (cur != NULL) {
    if (cur->layer == CROP) {
      // nothing.
    } else
    if (cur->layer == CONV) {
      struct convolution* e = (struct convolution*)cur->e;
      printf("  fread((char*)wt_%s, sizeof(scalar_t), %d*%d*%d*%d, wt_fp);\n", l_name.conv.name[conv_idx],
                                                                                e->size, e->size, e->c, e->n);
      struct node* next = cur->next;
      if (next->layer == BATCHNORM) {
        cur = next;
        struct batchnorm* ee = (struct batchnorm*)cur->e;
        printf("  fread((char*)bt_%s, sizeof(batch_ft), %d, wt_fp);\n", l_name.conv.name[conv_idx], ee->c);
      } else
      if (next->layer == BIAS) {
        cur = next;
        struct bias* ee = (struct bias*)cur->e;
        printf("  fread((char*)bs_%s, sizeof(scalar_t), %d, wt_fp);\n", l_name.conv.name[conv_idx], ee->c);
      }
      conv_idx++;
    } else
    if (cur->layer == CONNECTED) {
      struct connected* e = (struct connected*)cur->e;
      printf("  fread((char*)wt_%s, sizeof(scalar_t), %d*%d, wt_fp);\n", l_name.connected.name[connected_idx],
                                                                          e->inputs, e->outputs);
      struct node* next = cur->next;
      if (next->layer == BATCHNORM) {
        cur = next;
        struct batchnorm* ee = (struct batchnorm*)cur->e;
        printf("  fread((char*)bt_%s, sizeof(batch_ft), %d, wt_fp);\n", l_name.connected.name[connected_idx], ee->c);
      } else
      if (next->layer == BIAS) {
        cur = next;
        struct bias* ee = (struct bias*)cur->e;
        printf("  fread((char*)bs_%s, sizeof(scalar_t), %d, wt_fp);\n", l_name.connected.name[connected_idx], ee->c);
      }
      connected_idx++;
    } else
    if (cur->layer == ACTIV) {
      // nothing.
    } else
    if (cur->layer == MAXPOOL) {
      // nothing.
    } else
    if (cur->layer == RESIDUAL) {
      // nothing.
    } else
    if (cur->layer == ROUTE) {
      // nothing.
    } else
    if (cur->layer == AVGPOOL) {
      // nothing.
    } else
    if (cur->layer == SOFTMAX) {
      // nothing.
    } else {
      fprintf(stderr, "Invalid layer number in read_weights: %d\n", cur->layer);
      exit(0);
    }
    cur = cur->next;
  }
  printf("\n");
}

void erupt_network_code(struct node* head, struct name_list l_name)
{
  struct node* cur = head;
  int conv_idx = 0;
  int maxpool_idx = 0;
  int residual_idx = 0;
  int route_idx = 0;
  int avgpool_idx = 0;
  int connected_idx = 0;

  char in_name[NAME_STR_SIZE] = "im.data";
  char ot_name[NAME_STR_SIZE];

  while (cur != NULL) {
    if (cur->layer == CROP) {
      struct crop* e = (struct crop*)cur->e;
      erupt_crop_code(e); 
    } else
    if (cur->layer == CONV) {
      struct convolution* e = (struct convolution*)cur->e;
      strncpy(ot_name, l_name.conv.name[conv_idx++], NAME_STR_SIZE);

      erupt_convolution_code(e, in_name, ot_name);

      strncpy(in_name, ot_name, NAME_STR_SIZE);

      struct node* next = cur->next;
      if (next->layer == BIAS) {
        cur = next;
        struct bias* ee = (struct bias*)cur->e;
        erupt_bias_code(ee, in_name, ot_name);
      }
    } else
    if (cur->layer == BATCHNORM) {
      struct batchnorm* e = (struct batchnorm*)cur->e;
      erupt_batchnorm_code(e, in_name, ot_name);
    } else
    if (cur->layer == ACTIV) {
      struct activation* e = (struct activation*)cur->e;
      erupt_activation_code(e, in_name, ot_name);
    } else
    if (cur->layer == MAXPOOL) {
      struct maxpool* e = (struct maxpool*)cur->e;
      strncpy(ot_name, l_name.maxpool.name[maxpool_idx++], NAME_STR_SIZE);

      erupt_maxpool_code(e, in_name, ot_name);

      strncpy(in_name, ot_name, NAME_STR_SIZE);
    } else
    if (cur->layer == RESIDUAL) {
      struct residual* e = (struct residual*)cur->e;
      char fwd_name[NAME_STR_SIZE];
      if (residual_idx == 0)  
        strncpy(fwd_name, l_name.maxpool.name[maxpool_idx-1], NAME_STR_SIZE);
      else
        strncpy(fwd_name, l_name.residual.name[residual_idx-1], NAME_STR_SIZE);
      strncpy(ot_name, l_name.residual.name[residual_idx++], NAME_STR_SIZE);

      erupt_residual_code(e, fwd_name, in_name, ot_name);

      strncpy(in_name, ot_name, NAME_STR_SIZE);
    } else
    if (cur->layer == ROUTE) {
      struct route* e = (struct route*)cur->e;
      char fwd_name[NAME_STR_SIZE] = "";

      strncpy(in_name, l_name.conv.name[conv_idx-2], NAME_STR_SIZE);
      if (e->type == 1) {
        // nothing.
      } else
      if (e->type == 2) {
        strncpy(fwd_name, l_name.conv.name[conv_idx-1], NAME_STR_SIZE);
      } else {
        fprintf(stderr, "Invalid type in eruption for route: %d\n", e->type);
        exit(0);
      }
      strncpy(ot_name, l_name.route.name[route_idx++], NAME_STR_SIZE);

      erupt_route_code(e, in_name, fwd_name, ot_name);

      strncpy(in_name, ot_name, NAME_STR_SIZE);
    } else
    if (cur->layer == CONNECTED) {
      struct connected* e = (struct connected*)cur->e;
      strncpy(ot_name, l_name.connected.name[connected_idx++], NAME_STR_SIZE);

      erupt_connected_code(e, in_name, ot_name);

      strncpy(in_name, ot_name, NAME_STR_SIZE);

      struct node* next = cur->next;
      if (next->layer == BIAS) {
        cur = next;
        struct bias* ee = (struct bias*)cur->e;
        erupt_bias_connected_code(ee, in_name, ot_name);
      }
    } else
    if (cur->layer == AVGPOOL) {
      struct avgpool* e = (struct avgpool*)cur->e;
      strncpy(ot_name, l_name.avgpool.name[avgpool_idx++], NAME_STR_SIZE);

      erupt_avgpool_code(e, in_name, ot_name);

      strncpy(in_name, ot_name, NAME_STR_SIZE);
    } else
    if (cur->layer == SOFTMAX) {
      struct softmax* e = (struct softmax*)cur->e;
      strncpy(ot_name, "softmax", NAME_STR_SIZE);

      erupt_softmax_code(e, in_name, ot_name);
    } else {
      fprintf(stderr, "Invalid layer in erupt_network_code: %d\n", cur->layer);
      exit(0);
    }

    cur = cur->next;
  }
}

void erupt_crop_code(struct crop* crop)
{
  printf("  im = crop(im, %d, %d);\n\n", crop->h, crop->w);
}

void erupt_convolution_code(struct convolution* conv, char* input, char* output)
{
  char in[NAME_STR_SIZE] = "ot_";
  char wt[NAME_STR_SIZE] = "wt_";
  char ot[NAME_STR_SIZE] = "ot_";

  if (strcmp(input, "im.data") == 0)  strncpy(in, input, NAME_STR_SIZE);
  else  strcat(in, input);
  strcat(wt, output);
  strcat(ot, output);

  printf("#ifdef TIME_ESTIMATE\n");
  printf("  est_time = clock();\n");
  printf("#endif\n");
  printf("#ifdef PLANNER\n");
  printf("  %s(%s, %s, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, in_mem_0, wt_mem_0, ot_mem_0);\n",
      output, in, wt, ot, 
      conv->n, conv->c, conv->size, conv->h, conv->w, 
      conv->stride, conv->padding, conv->padding, conv->padding, conv->padding);
  printf("#elif defined(CUDA) || defined(CUBLAS)\n");
  printf("  conv2d_gpu(%s, %s, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d);\n",
      in, wt, ot,
      conv->n, conv->c, conv->size, conv->h, conv->w,
      conv->stride, conv->padding, conv->padding, conv->padding, conv->padding);
  printf("#else\n");
  printf("  conv2d(%s, %s, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d);\n",
      in, wt, ot,
      conv->n, conv->c, conv->size, conv->h, conv->w,
      conv->stride, conv->padding, conv->padding, conv->padding, conv->padding);
  printf("#endif\n");
  printf("#ifdef TIME_ESTIMATE\n");
  printf("  printf(\"%s elapsed time: %%.3f msec\\n\", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);\n", output);
  printf("#endif\n");
}

void erupt_batchnorm_code(struct batchnorm* bn, char* input, char* output)
{
  printf("  batchnorm(ot_%s, ot_%s, bt_%s, 1, %d, %d, %d);\n",
      input, output, output, bn->c, bn->h, bn->w);
}

void erupt_bias_code(struct bias* bs, char* input, char* output)
{
  printf("  addbias(ot_%s, ot_%s, bs_%s, %d, %d*%d);\n",
      input, output, output, bs->c, bs->h, bs->w);
}

void erupt_bias_connected_code(struct bias* bs, char* input, char* output)
{
  printf("  addbias_connected(ot_%s, ot_%s, bs_%s, %d);\n",
      input, output, output, bs->c);
}

void erupt_maxpool_code(struct maxpool* mxp, char* input, char* output)
{
  printf("  maxpool2d(ot_%s, ot_%s, %d, %d, %d, %d, %d, %d);\n\n",
      input, output, mxp->c, mxp->h, mxp->w, mxp->size, mxp->stride, mxp->padding);
}

void erupt_activation_code(struct activation* act, char* input, char* output)
{
  if (act->act == LINEAR) return;

  if (act->act == RELU) {
    printf("  relu");
  } else
  if (act->act == LEAKY) {
    printf("  leaky_relu");
  }
  printf("(ot_%s, ot_%s, %d*%d*%d*%d);\n\n", input, output, act->n, act->c, act->h, act->w);
}

void erupt_residual_code(struct residual* res, char* input1, char* input2, char* output)
{
  printf("  residual(ot_%s, ot_%s, ot_%s, %d, %d, %d, %d, %d, %d, %d);\n",
      input1, input2, output, res->batch, res->c1, res->h1, res->w1, res->c2, res->h2, res->w2);
}

void erupt_route_code(struct route* rt, char* input1, char* input2, char* output)
{
  if (rt->type == 1) {
    printf("  route_1(ot_%s, ot_%s, %d, %ld);\n", input1, output, rt->batch, rt->size_1);
  } else
  if (rt->type == 2) {
    printf("  route_2(ot_%s, ot_%s, ot_%s, %d, %ld, %ld);\n", input1, input2, output, rt->batch, rt->size_1, rt->size_2);
  } else {
    fprintf(stderr, "Invalid type for route: %d\n", rt->type);
    exit(0);
  }
}

void erupt_avgpool_code(struct avgpool* avg, char* input, char* output)
{
  printf("  avgpool2d(ot_%s, ot_%s, %d, %d, %d);\n\n", input, output, avg->c, avg->h, avg->w);
}

void erupt_connected_code(struct connected* conn, char* input, char* output)
{
  printf("  connected(ot_%s, wt_%s, ot_%s, %d, %d);\n", input, output, output, conn->outputs, conn->inputs);
}

void erupt_softmax_code(struct softmax* smx, char* input, char* output)
{
  printf("  softmax(ot_%s, ot_%s, %d, %f);\n\n", input, output, smx->size, smx->temp);
}

void erupt_topk(int class_cnt)
{
  printf("  int i;\n");
  printf("  int topk_idx[top_K];\n");
  printf("  top_k(ot_softmax, %d, top_K, topk_idx);  // Extract indexes of top K possible results\n", class_cnt);
  printf("  float topk_pos[top_K];  // Possibilities of top K results.\n");
  printf("  for (i = 0 ; i < top_K ; ++i) {\n");
  printf("    topk_pos[i] = ot_softmax[topk_idx[i]];\n");
  printf("  }\n");
  printf("  sort_top_k(topk_pos, topk_idx, top_K);  // Sort topk_pos and topk_idx.\n\n");
  
  printf("  for (i = 0 ; i < top_K ; ++i) {\n");
  printf("    int class_num = topk_idx[i];\n");
  printf("    char label[STR_SIZE];\n");
  printf("    get_darknet_label(label, class_num);  // Find labels corresponding to class number.\n");
  printf("    printf(\"(%%5.2f%%%%) %%s\\n\", topk_pos[i]*100, label);\n");
  printf("  }\n\n");
}

void deallocate(struct node* head, struct name_list l_name)
{
  printf("  // Deallocate resources\n");
  struct node* cur = head;
  int conv_idx = 0 ;
  int maxpool_idx = 0;
  int residual_idx = 0;
  int route_idx = 0;
  int connected_idx = 0;
  int avgpool_idx = 0;

  printf("#ifdef PLANNER\n");
  printf("#if defined(CUDA) || defined(CUBLAS)\n");
  printf("  cudaFreeWrapper(in_mem_0);\n");
  printf("  cudaFreeWrapper(wt_mem_0);\n");
  printf("  cudaFreeWrapper(ot_mem_0);\n");
  printf("#else\n");
  printf("  free(in_mem_0);\n");
  printf("  free(wt_mem_0);\n");
  printf("  free(ot_mem_0);\n");
  printf("#endif\n");
  printf("#endif\n");

  while (cur != NULL) {
    if (cur->layer == CROP) {
      // nothing.
    } else
    if (cur->layer == CONV) {
      printf("  free(wt_%s);\n", l_name.conv.name[conv_idx]);

      struct node* next = cur->next;
      if (next->layer == BATCHNORM) {
        cur = next;
        printf("  free(bt_%s);\n", l_name.conv.name[conv_idx]);
        printf("  free(ot_%s);\n", l_name.conv.name[conv_idx]);
      } else
      if (next->layer == BIAS) {
        cur = next;
        printf("  free(bs_%s);\n", l_name.conv.name[conv_idx]);
        printf("  free(ot_%s);\n", l_name.conv.name[conv_idx]);
      }
      conv_idx++;
    } else
    if (cur->layer == ACTIV) {
      // nothing.
    } else
    if (cur->layer == MAXPOOL) {
      printf("  free(ot_%s);\n", l_name.maxpool.name[maxpool_idx++]);
    } else
    if (cur->layer == RESIDUAL) {
      printf("  free(ot_%s);\n", l_name.residual.name[residual_idx++]);
    } else
    if (cur->layer == ROUTE) {
      printf("  free(ot_%s);\n", l_name.route.name[route_idx++]);
    } else
    if (cur->layer == CONNECTED) {
      printf("  free(wt_%s);\n", l_name.connected.name[connected_idx]);

      struct node* next = cur->next;
      if (next->layer == BATCHNORM) {
        cur = next;
        printf("  free(bt_%s);\n", l_name.connected.name[connected_idx]);
        printf("  free(ot_%s);\n", l_name.connected.name[connected_idx]);
      } else
      if (next->layer == BIAS) {
        cur = next;
        printf("  free(bs_%s);\n", l_name.connected.name[connected_idx]);
        printf("  free(ot_%s);\n", l_name.connected.name[connected_idx]);
      }
      connected_idx++;
    } else
    if (cur->layer == AVGPOOL) {
      printf("  free(ot_%s);\n", l_name.avgpool.name[avgpool_idx++]);
    } else
    if (cur->layer == SOFTMAX) {
      printf("  free(ot_softmax);\n");
    } else {
      fprintf(stderr, "Invalid layer number in deallocate: %d\n", cur->layer);
      exit(0);
    }
    cur = cur->next;
  }
  printf("\n");
}

char *get_activation_string(ACTIVATION a)                                       
{                                                                               
    switch(a){                                                                  
        case LOGISTIC:                                                          
            return "logistic";                                                  
        case LOGGY:                                                             
            return "loggy";                                                     
        case RELU:                                                              
            return "relu";                                                      
        case ELU:                                                               
            return "elu";                                                       
        case SELU:                                                              
            return "selu";                                                      
        case RELIE:                                                             
            return "relie";                                                     
        case RAMP:                                                              
            return "ramp";                                                      
        case LINEAR:                                                            
            return "linear";                                                    
        case TANH:                                                              
            return "tanh";                                                      
        case PLSE:                                                              
            return "plse";                                                      
        case LEAKY:                                                             
            return "leaky";                                                     
        case STAIR:                                                             
            return "stair";                                                     
        case HARDTAN:                                                           
            return "hardtan";                                                   
        case LHTAN:                                                             
            return "lhtan";                                                     
        default:                                                                
            break;                                                              
    }                                                                           
    return "relu";                                                              
}   

ACTIVATION get_activation(char *s)                                              
{                                                                               
    if (strcmp(s, "logistic")==0) return LOGISTIC;                              
    if (strcmp(s, "loggy")==0) return LOGGY;                                    
    if (strcmp(s, "relu")==0) return RELU;                                      
    if (strcmp(s, "elu")==0) return ELU;                                        
    if (strcmp(s, "selu")==0) return SELU;                                      
    if (strcmp(s, "relie")==0) return RELIE;                                    
    if (strcmp(s, "plse")==0) return PLSE;                                      
    if (strcmp(s, "hardtan")==0) return HARDTAN;                                
    if (strcmp(s, "lhtan")==0) return LHTAN;                                    
    if (strcmp(s, "linear")==0) return LINEAR;                                  
    if (strcmp(s, "ramp")==0) return RAMP;                                      
    if (strcmp(s, "leaky")==0) return LEAKY;                                    
    if (strcmp(s, "tanh")==0) return TANH;                                      
    if (strcmp(s, "stair")==0) return STAIR;                                    
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s); 
    return RELU;                                                                
}    
