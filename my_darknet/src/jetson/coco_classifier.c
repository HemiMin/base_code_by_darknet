#include "classifier.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LABEL_FILE  "labels/coco_labels.list"

void get_coco_label(char* label, int idx)
{
  FILE* fp_lbl = fopen(LABEL_FILE, "r");

  if (fp_lbl == NULL) {
    fprintf(stderr, "%s file doesn't exist.\n", LABEL_FILE);
    exit(0);
  }

  if (idx >= 80 || idx < 0) {
    fprintf(stderr, "Index is out of bound (%d): less than 80\n", idx);
    exit(0);
  }

  int i;
  for (i = 0 ; i <= idx ; ++i) {
    fgets(label, STR_SIZE, fp_lbl);
    char* pos;
    if ((pos = strchr(label, '\n')) != NULL) {
      *pos = '\0';
    }
  }

  fclose(fp_lbl);
}
