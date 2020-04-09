#include<stdlib.h>
#include<stdio.h>

int countLine(char filename[])
{
  FILE *fp;
  int count = 0;
  char c;
  fp = fopen(filename, "r");

  if (fp == NULL)
  {
      printf("Could not open file %s", filename);
      return 0;
  }

  for (c = getc(fp); c != EOF; c = getc(fp))
  {
    if (c == '\n') count = count + 1;
  }

  fclose(fp);
  return count;
}

float* readFile(char filename[])
{
  FILE *fptr;

  int lines = countLine(filename);
  fptr = fopen(filename, "r");

  float* arr = (float*)malloc(sizeof(float) * lines);
  float num; int i = 0;
  while(fscanf(fptr,"%f",&num) != EOF)
  {
    arr[i] = num;
    i++;
  }

  return arr;
}
