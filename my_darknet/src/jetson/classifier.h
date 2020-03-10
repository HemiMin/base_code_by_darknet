#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#define STR_SIZE 256

void get_imagenet_label(char* label, int idx);
void get_darknet_label(char* label, int idx);
void get_coco_label(char* label, int idx);

#endif
