//MathMilt.c

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cblas.h>
#include <mpi.h>
//#define INTERACTIVE
//#define DEBUGGING

void initMatrices(double*,double*,double*,int,int);
double getFlops(double,int);
void printResults(double*,int,int);
void mpiMult(const int, const int, double*, double*, double*,int,int,MPI_Status,double *);
double justinTime();


int main(int argc, char* argv[])
{
	MPI_Init(&argc,&argv);
	MPI_Status status;
	int size,rank;
	int initialSize,finalSize;
	int iterations,x;
    	double *a,*b,*c,*timeArray;
    	double time;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	//printf("Processor %d: Beginning MPI\n",rank);

#ifdef INTERACTIVE
	printf("\nEnter the initial size:\n");
	scanf("%d",&initialSize);
	printf("\nEnter the final size:\n");
	scanf("%d",&finalSize);
	printf("\nEnter the iterations:\n");
	scanf("%d",&iterations);
#else
	initialSize=6144;
	finalSize=6144;
	iterations=3;
#endif
	if(rank==0)
	{
		printf("size\titerate\tMeanFLOPS\tMaxFLOPS\tMeanTime\tMinTime \tStDTime\n");
    		for(x=0;x<90;x++)
		printf("_");
    		printf("\n");

	}

    	for(x=initialSize;x<=finalSize;x*=2)
    	{
        	if(x>=1024)
            		iterations=3;
            	a=malloc(x * x/size * sizeof(double));
	    	b=malloc(x * x * sizeof(double));
    		c=malloc(x * x/size * sizeof(double));
		if (rank==0)	
			initMatrices(a,b,c,x,size);
		timeArray=malloc(iterations*sizeof(double));
		mpiMult(x,iterations,a,b,c,size,rank,status,timeArray);
		if(rank==0)
	       		printResults(timeArray,x,iterations);
		free(a);
        	free(b);
	        free(c);
		free(timeArray);
	 }
	MPI_Finalize();
	return 0;
}
void mpiMult(const int NSIZE, const int ITERATIONS, double *a, double *b, double *c,int size, int rank, MPI_Status status,double *arrayOfTime)
{
    int x,y,offset=(NSIZE/size);
for(y=0;y<ITERATIONS;y++)
{
	#ifdef DEBUGGING
		printf("Running iteration %d\n",y);
	#endif
    if(rank==0)
    {

        #ifdef DEBUGGING
		printf("Running iteration %d\n",y);
	#endif
        double flops,time;

	#ifdef DEBUGGING
		printf("Processor %d: Running initMatrices()\n",rank);
	#endif
        time=justinTime();
	#ifdef DEBUGGING
		printf("Processor %d: Starting send...\n",rank);
	#endif
            for(x=1;x<size;x++)
            {
                //MPI_Send(&a[offset*x],NSIZE*NSIZE/4,MPI_double,x,0,MPI_COMM_WORLD);
		MPI_Send(a,NSIZE*NSIZE/size,MPI_double,x,0,MPI_COMM_WORLD);
                MPI_Send(b,NSIZE*NSIZE,MPI_double,x,1,MPI_COMM_WORLD);
            }
	#ifdef DEBUGGING
		printf("Procesor %d: Running CBLAS...\n",rank);
	#endif
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,NSIZE/size,NSIZE,NSIZE,1.0,a,NSIZE,b,NSIZE,1.0,c,NSIZE);
	    for(x=1;x<size;x++)
            {
                MPI_Recv(c,NSIZE*NSIZE/size,MPI_double,x,0,MPI_COMM_WORLD,&status);
//                MPI_Recv(&c[offset*x],NSIZE*NSIZE/4,MPI_double,x,0,MPI_COMM_WORLD,&status);
		#ifdef DEBUGGING
			printf("Processor %d: Receiving data from %d\n",rank,x);
		#endif

            }
            time=justinTime()-time;
            arrayOfTime[y]=time;
    }else if(rank>0)
    {
	#ifdef DEBUGGING
		printf("Procesor %d: Receiving data from Master...\n",rank);
	#endif
        MPI_Recv(a,NSIZE*NSIZE/size,MPI_double,0,0,MPI_COMM_WORLD,&status);
        MPI_Recv(b,NSIZE*NSIZE,MPI_double,0,1,MPI_COMM_WORLD,&status);
	#ifdef DEBUGGING
		printf("Procesor %d: Running CBLAS...\n",rank);
	#endif
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,NSIZE/size,NSIZE,NSIZE,1.0,a,NSIZE,b,NSIZE,1.0,c,NSIZE);

        MPI_Send(c,NSIZE*NSIZE/size,MPI_double,0,0,MPI_COMM_WORLD); 
   }
}
}
void initMatrices(double *h,double *i, double *j,int NSIZE,int size)
{
    int x;
    
    for(x=0;x<NSIZE*NSIZE/size;x++)
    {
        h[x]=(rand()/RAND_MAX-0.5);
        j[x]=0.0;
    }
    for(x=0;x<NSIZE*NSIZE;x++)
	    i[x]=(rand()/RAND_MAX-0.5);

}

double getFlops(double t,int NSIZE)
{
    int s=NSIZE;
    return ((2.0 * s/100.0)*(s/100.0)*(s/100.0)) / t;
}

void printResults(double *array,int NSIZE,int ITERATIONS)
{
    //Get the mean time, the max time and std
    double meanFlops,maxFlops,meanTime=0,minTime=2.0*86400,stdTime=0;
    int x;
    for(x=0;x<ITERATIONS;x++)
    {
        meanTime=meanTime+array[x];
        if(array[x]<minTime)
            minTime=array[x];
    }
    meanTime=meanTime/ITERATIONS;
    for(x=0;x<ITERATIONS;x++)
    {
        stdTime+=(array[x]-meanTime)*(array[x]-meanTime);
    }
    stdTime/= ITERATIONS;
    meanFlops=getFlops(meanTime,NSIZE);
    maxFlops=getFlops(minTime,NSIZE);
    printf("%d\t%d\t%4.4g MFLOPS\t%4.4g MFLOPS\t%4.4g secs\t%4.4g secs\t%4.4g secs\n",NSIZE,ITERATIONS,meanFlops,maxFlops,meanTime,minTime,stdTime);
}
double justinTime()
{
	struct timeval t;
	gettimeofday(&t,NULL);
	return (double)t.tv_sec+1.0e-6*t.tv_usec;
}
