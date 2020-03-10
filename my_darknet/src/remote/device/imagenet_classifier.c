#include "classifier.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LABEL_FILE "labels/imagenet_labels.json"

void get_imagenet_label(char* label, int idx)
{
  char buf[STR_SIZE];
  int i, flg=0, str_idx=0;
  FILE* fp_lbl = fopen(LABEL_FILE, "r");

  if (fp_lbl == NULL) {
    fprintf(stderr, "%s file doesn't exist.", LABEL_FILE);
    exit(0);
  } 

  if (idx >= 1000) {
    fprintf(stderr, "Index is out of bound: less than 1000");
    exit(0);
  }

  fgets(buf, STR_SIZE, fp_lbl);  // remove {
  
  for (i = 0 ; i < idx ; ++i) fgets(buf, STR_SIZE, fp_lbl);

  fgets(buf, STR_SIZE, fp_lbl);

  for (i = 0 ; i < STR_SIZE ; ++i) {
    if (buf[i] == '\0') break;
    else 
    if (buf[i] == '\'') {
      flg = !flg;
    } else
    if (flg) {
      label[str_idx++] = buf[i];
    }
  }
  label[str_idx] = '\0';

  fclose(fp_lbl);
}
