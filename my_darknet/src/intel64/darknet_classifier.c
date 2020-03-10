#include "classifier.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LABEL_FILE "labels/darknet_labels.list"

void get_darknet_label(char* label, int idx)
{
  FILE* fp_lbl = fopen(LABEL_FILE, "r");

  if (fp_lbl == NULL) {
    fprintf(stderr, "%s file doesn't exist.\n", LABEL_FILE);
    exit(0);
  }

  if (idx >= 1000 || idx < 0) {
    fprintf(stderr, "Index is out of bound: less than 1000\n");
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
