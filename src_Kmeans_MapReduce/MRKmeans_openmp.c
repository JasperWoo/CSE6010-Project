#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include "util.h"
#include "mapper.h"
#include "reducer.h"

int main(int argc, char *argv[]){

  //a guard
  if(argc!= 3){
      fprintf(stderr, "Must provide 2 args -- "
              "a string specifying the filename of datasets, and another"
              " string for filename used to output resulut.\n");
  }

  FILE *data_file = fopen(argv[1],"r");
  char filename[12];
  strcpy(filename, argv[2]);
  strcat(filename,".csv");
  FILE *output_file = fopen(filename,"w");

  int num_segs = NUM_THREADS;

  //read dataset; initialize cluster centroids to be the first <NUM_CLUSTER> points in the dataset
  seg* sHead = read_data(data_file, num_segs);

  init(sHead);

  // run mapper function through the list of data sample points

  omp_set_num_threads(NUM_THREADS);

  double start = omp_get_wtime();

  int i;

  for(i=0; i<MAX_ITERATIONS; i++){
   int is_unchanged = 0;

//Map phase
#pragma omp parallel
#pragma omp single
      {
        seg* sCrawl = sHead;
        while(sCrawl){
#pragma omp task
          mapper_seg(sCrawl);
          sCrawl = sCrawl->next;
        }
      }

//Combine phase and Reduce phase. These two phases can actually be joined together in this project.
#pragma omp parallel default(none) shared(sHead, NUM_CLUSTERS,is_unchanged,cltr_arr)
    {
      point** heads_cltr_list = (point **)malloc(NUM_CLUSTERS*sizeof(point*));
#pragma omp for schedule(static) reduction(+:is_unchanged)
      for(int index=0; index<NUM_CLUSTERS; index++){
        heads_cltr_list[index] = combine(index+1,sHead,heads_cltr_list[index]);
        int temp = reduce(index, heads_cltr_list[index]);
        is_unchanged += temp;
      }
    }
      if (is_unchanged == 6){
        break;
      }
  }
    printf("Iteration up to %d.\nTime w/ openmp md is : \t %f \n", i, omp_get_wtime()-start);

    //Output results
    fprintf(output_file, "ASSIGNED_CLUSTER,XCOORD,YCOORD");
    seg *sCrawl = sHead;
    while(sCrawl){
      point *pCrawl = sCrawl->head;
      while(pCrawl){
        fprintf(output_file,"\n%d,%d,%d",pCrawl->cltr_assigned,pCrawl->xcoord,pCrawl->ycoord);
        pCrawl = pCrawl->next;
      }
      sCrawl = sCrawl->next;
    }
}
