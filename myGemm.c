// Description: MPI Block Matrix Multiplication
// This program performs a block matrix multiplication using MPI.
// It divides the matrices into blocks and distributes the computation across multiple processes.
// Each process computes its assigned block and sends the results back to the master process.
// The program measures the time taken for each iteration and calculates the performance in MFLOPS.
// Author: Justin Moore
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
//#define INTERACTIVE
//#define DEBUG
void blockMultiply(const int,const int,const int,float*,float*,float*,int,int,float*,MPI_Status);
void raj(float*,float*,float*,int,int,int,int,int,int,const int);
void initMatrices(float*,float*,float*,int);
void printResults(float*,int,int);
double getFlops(double,int);
double justinTime();
void printMatrix(float*,const int);

/*******************************************************************/

int main(int argc, char* argv[])
{
    MPI_Init(&argc,&argv);
    MPI_Status status;
    float *a,*b,*c,*timeArray;
    int size,rank;
    int initialSize,finalSize,partition;
    int iterations,x;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
#ifdef INTERACTIVE
    if(rank==0)
    {
        printf("Choose intital problem size: ");
        scanf("%d",&initialSize);
        printf("Choose final problem size: ");
        scanf("%d",&finalSize);
        printf("Choose partition size: ");
        scanf("%d",&partition);
        printf("Choose iterations: ");
        scanf("%d",&iterations);
    }
#else
    initialSize=256;
    finalSize=8192;
    partition=128;
    iterations=10;
#endif
//    printf("Processor: %d, Processors: %d\n",rank,size);

//    if(rank==0)
//        printf("Partition: %d\n",partition);
    
    for(x=initialSize;x<=finalSize;x*=2)
    {
        if(x>=4096)
            iterations=3;
        a=malloc(x * x * sizeof(float));
        b=malloc(x * x * sizeof(float));
        c=malloc(x * x * sizeof(float));
	if(rank==0)
            initMatrices(a,b,c,x);
        timeArray=malloc(iterations*sizeof(float));
        blockMultiply(x,partition,iterations,a,b,c,size,rank,timeArray,status);
        if(rank==0)
        {
            printResults(timeArray,x,iterations);
        }
       // if(rank==0)
           // printMatrix(c,x);
        free(a);
        free(b);
        free(c);
        free(timeArray);
    }
    
    MPI_Finalize();
    return 0;
}

/*******************************************************************************/

void blockMultiply(const int NSIZE,const int NBSIZE,const int ITERATIONS,float *a,float *b,float *c,int size,int rank,float *arrayOfTime,MPI_Status status)
{
    int u,v,w,x,y;
    for(x=0;x<ITERATIONS;x++)
    {
        
        if(rank==0)
        {
            #ifdef DEBUG
                printf("Running iteration %d\n",x);
            #endif
            double time;
            #ifdef DEBUG
                printf("Processor %d: Running initMatrices()...\n",rank);
            #endif
            
            time=justinTime();
            #ifdef DEBUG
                printf("Processor %d: Starting send...\n",rank);
            #endif
            for(y=1;y<size;y++)
            {
                MPI_Send(&a[y*NSIZE*NSIZE/size],NSIZE*NSIZE/size,MPI_FLOAT,y,0,MPI_COMM_WORLD);
                MPI_Send(b,NSIZE*NSIZE,MPI_FLOAT,y,1,MPI_COMM_WORLD);
            }
            #ifdef DEBUG
                printf("Processor %d: Running raj()...\n",rank);
            #endif
            for(u=0;u<(NSIZE/size);u++)
                for(v=0;v<(NSIZE/NBSIZE);v++)
                    for(w=0;w<(NSIZE/NBSIZE);w++)
                        raj(a,b,c,u ,v*NBSIZE,w*NBSIZE,(u+1),(v+1)*NBSIZE,(w+1)*NBSIZE,NSIZE);
            
            for(y=1;y<size;y++)
            {
                #ifdef DEBUG
                    printf("Processor %d: Recieving data from %d\n",rank,y);
                #endif
                MPI_Recv(&c[y*NSIZE*NSIZE/size],NSIZE*NSIZE/size,MPI_FLOAT,y,0,MPI_COMM_WORLD,&status);
            }
            time=justinTime()-time;
            arrayOfTime[x]=time;
        }
        else if(rank>0)
        {
            #ifdef DEBUG
                printf("Processor %d: Recieving from Master...\n",rank);
            #endif
            MPI_Recv(a,NSIZE*NSIZE/size,MPI_FLOAT,0,0,MPI_COMM_WORLD,&status);
            MPI_Recv(b,NSIZE*NSIZE,MPI_FLOAT,0,1,MPI_COMM_WORLD,&status);
            #ifdef DEBUG
                printf("Processor %d: Running raj()..\n",rank);
            #endif
            for(u=0;u<(NSIZE/size);u++)
                for(v=0;v<(NSIZE/NBSIZE);v++)
                    for(w=0;w<(NSIZE/NBSIZE);w++)
                        raj(a,b,c,u,v*NBSIZE,w*NBSIZE,(u+1),(v+1)*NBSIZE,(w+1)*NBSIZE,NSIZE);
            #ifdef DEBUG
                printf("Processor %d: Sending to Master...\n",rank);
            #endif
            MPI_Send(c,NSIZE*NSIZE/size,MPI_FLOAT,0,0,MPI_COMM_WORLD);
        }
    }
    
    
    
    
}

inline void raj(float *a, float *b, float *c, int rowStartI,int rowStartJ, int rowStartK,int endI, int endJ,int endK,const int NSIZE)
{
    int i,j,k;
    float sum1,sum2,sum3,sum4;
    for(i=rowStartI;i<endI;i++)
        for(j=rowStartJ;j<endJ;j+=4)
        {
            sum1=c[i*NSIZE+j];
            sum2=c[i*NSIZE+j+1];
            sum3=c[i*NSIZE+j+2];
            sum4=c[i*NSIZE+j+3];
            for (k=rowStartK; k<endK; k++)
            {
                sum1 += (a[i*NSIZE+k]*b[k*NSIZE+j]);
                sum2 += (a[i*NSIZE+k]*b[k*NSIZE+j+1]);
                sum3 += (a[i*NSIZE+k]*b[k*NSIZE+j+2]);
                sum4 += (a[i*NSIZE+k]*b[k*NSIZE+j+3]);
            }
            c[i*NSIZE+j]=sum1;
            c[i*NSIZE+j+1]=sum2;
            c[i*NSIZE+j+2]=sum3;
            c[i*NSIZE+j+3]=sum4;
        }
    
}

void initMatrices(float *h,float *i, float *j,int NSIZE)
{
    int x;
    
    for(x=0;x<NSIZE*NSIZE;x++)
    {
        h[x]=(rand()/RAND_MAX-0.5);
        j[x]=0.0;
	i[x]=(rand()/RAND_MAX-0.5);
    }
}

void printResults(float *array,int NSIZE,int ITERATIONS)
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
    printf("%d\t%4.4g MFLOPS\t%4.4g MFLOPS\t%4.4g secs\t%4.4g secs\t%4.4g secs\n",NSIZE,meanFlops,maxFlops,meanTime,minTime,stdTime);
}

double getFlops(double t, int NSIZE)
{
    int s=NSIZE;
    return ((2.0 * s/100.0)*(s/100.0)*(s/100.0)) / t;
}

double justinTime()
{
	struct timeval t;
	gettimeofday(&t,NULL);
	return (double)t.tv_sec+1.0e-6*t.tv_usec;
}

void printMatrix(float *a,const int NSIZE)
{
    int i;
    for(i=0;i<NSIZE*NSIZE;i++)
    {
        printf("%1.1f ",a[i]);
        if((i+1)%NSIZE==0)
            printf("\n");
    }
    printf("\n");
}
