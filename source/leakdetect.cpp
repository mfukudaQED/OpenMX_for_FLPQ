#include <stdio.h>
#include <stdlib.h>
#undef LEAK_DETECT
#include "leakdetect.h"
#include <unordered_map>
#include "mpi.h"
#include <omp.h>

  
struct MEM_IN_FILE{
  size_t size;
  const char *file;
  unsigned int line;
} ;


static int num_alloc_mem = 0;
static std::unordered_map< void *, MEM_IN_FILE > mem_list;

omp_lock_t lock;

/* initialize */
void leak_detect_init(void){

  int numprocs,myid;

  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  if(myid == 0){
    printf("LEAK_DETECT init---------------------------\n");fflush(stdout);
  }
  omp_init_lock(&lock);
  mem_list.clear();
  num_alloc_mem = 0;
}

/* wrapper of malloc */
void *leak_detect_malloc(size_t size, const char *file, unsigned int line){
  int i = 0;
  void *ptr = malloc(size);

  if(ptr == NULL){
    return NULL;
  }

  omp_set_lock(&lock);
  mem_list.insert( std::pair<void*, MEM_IN_FILE >{ptr, MEM_IN_FILE{size, file, line}});
  ++num_alloc_mem;
  omp_unset_lock(&lock);
    
  return ptr;
}

/* wrapper of free */
void leak_detect_free(void *ptr){    
  omp_set_lock(&lock);
  mem_list.erase(ptr);
  omp_unset_lock(&lock);
  free(ptr);
}

/* show memory leak */
void leak_detect_check(void){

  int numprocs,myid;

  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  printf("ID=%2d LEAK_DETECT result (%d / %d)-----------------------\n",myid,(int)mem_list.size(), num_alloc_mem);fflush(stdout);
    
  omp_set_lock(&lock);
  for(auto&& a : mem_list){
    
    printf("ID=%2d memory leak!, addr:%p\n", myid,a.first);
    printf(" size:%u\n", (unsigned int)a.second.size);
    printf(" file,line:%s:%u\n", a.second.file, a.second.line);
    printf("\n");fflush(stdout);
        
  }
  omp_unset_lock(&lock);

  //omp_destroy_lock(&lock);
}

/* finalize */
void leak_detect_finalize(void) {
  omp_destroy_lock(&lock);
}
