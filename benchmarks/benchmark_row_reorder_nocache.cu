#define REORDER 1

#define GOOD_WEATHER 0
#define BAD_WEATHER 1

#define TAG_Car 0
#define TAG_Pedestrian 1

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <random>
//#include <array> 
#include <algorithm> 

#define NUM_CARS 4096
#define NUM_PEDS 16384
#define NUM_STREETS 500
#define MAX_CONNECTIONS 10
#define MAX_LEN 25

using namespace std;

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

typedef struct
{
	float progress;
	int street;
	float max_velocity;
	int tag;

	int a1;
	int a2;
	int a3;
	int a4;
	int a5;
	int a6;
	int a7;
	int a8;
	int a9;
	int a10;
	// TODO: add more fields here to avoid cache locality
} struct_Actor;

typedef struct
{
	float length;
	float max_velocity;
	int neighbor_array_index;

	int s1;
	int s2;
	int s3;
	int s4;
	int s5;
} struct_Street;

typedef struct
{
	int tag;
	int id;
} tag_id_pair;

__device__ struct_Actor *d_Actors;
__device__ struct_Street *d_Streets;
__device__ int *d_Array_Street_arrays;
__device__ int *d_Array_Street_offset;
__device__ int *d_Array_Street_size;
__device__ int *d_input_actor_id;
__device__ int *d_jobs;
__device__ int *d_randomn;

__device__ void method_Car_move(int car_id, int weather)
{
	float weather_multiplier;
	if (weather == GOOD_WEATHER) 
	{
		weather_multiplier = 1.0;
	}
	else if (weather == BAD_WEATHER)
	{
		weather_multiplier = 0.75;
	}

	float speed = min(d_Actors[car_id].max_velocity, d_Streets[d_Actors[car_id].street].max_velocity) * weather_multiplier;
	d_Actors[car_id].progress = d_Actors[car_id].progress + (speed / 60.0); /* 1 tick = 1 minute */

	if (d_Actors[car_id].progress >= d_Streets[d_Actors[car_id].street].length)
	{
		// move to different street
		int array_id = d_Streets[d_Actors[car_id].street].neighbor_array_index;
		int neighbor_index = d_randomn[d_Actors[car_id].street] % d_Array_Street_size[array_id];
		d_Actors[car_id].street = d_Array_Street_arrays[d_Array_Street_offset[array_id] + neighbor_index];
		d_Actors[car_id].progress = 0.0f;
	}
}

__device__ void method_Pedestrian_move(int ped_id, int weather)
{
	float speed = d_randomn[((int) (d_Actors[ped_id].progress*d_Actors[ped_id].progress)) % NUM_STREETS] % 7 - 2;
	d_Actors[ped_id].progress = d_Actors[ped_id].progress + (speed / 60.0);

	if (d_Actors[ped_id].progress >= d_Streets[d_Actors[ped_id].street].length)
	{
		// move to different street
		int array_id = d_Streets[d_Actors[ped_id].street].neighbor_array_index;
		int neighbor_index = d_randomn[d_Actors[ped_id].street] % d_Array_Street_size[array_id];
		d_Actors[ped_id].street = d_Array_Street_arrays[d_Array_Street_offset[array_id] + neighbor_index];
		d_Actors[ped_id].progress = 0.0f;
	}
}

__device__ void block(int actor_id, int weather, int ticks)
{
	for (int i = 0; i < ticks; i++)
	{
		switch (d_Actors[actor_id].tag)
		{
			case TAG_Car:
				method_Car_move(actor_id, weather);
				break;
			case TAG_Pedestrian:
				method_Pedestrian_move(actor_id, weather);
				break;
		}
	}
}

__global__ void kernel(int weather,	int ticks,
	struct_Actor *v_d_Actors, struct_Street *v_d_Streets,
	int *v_d_Array_Street_size, int *v_d_Array_Street_offset, int *v_d_Array_Street_arrays,
	int *v_d_input_actor_id, int *v_d_jobs, int *v_d_randomn)
{
	d_Actors = v_d_Actors;
	d_Streets = v_d_Streets;
	d_Array_Street_size = v_d_Array_Street_size;
	d_Array_Street_offset = v_d_Array_Street_offset;
	d_Array_Street_arrays = v_d_Array_Street_arrays;
	d_input_actor_id = v_d_input_actor_id;
	d_jobs = v_d_jobs;
	d_randomn = v_d_randomn;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	__syncthreads();

#if (REORDER)
	block(d_input_actor_id[d_jobs[tid]], weather, ticks);
#else
	block(d_input_actor_id[tid], weather, ticks);
#endif
}

int main()
{
	printf("Setting up scenario...\n");
	srand(42);

	// streets
	float *Street_length = new float[NUM_STREETS];
	float *Street_max_velocity = new float[NUM_STREETS];
	int *Street_neighbors = new int[NUM_STREETS];

	for (int i = 0; i < NUM_STREETS; i++)
	{
		Street_length[i] = rand() % MAX_LEN + 1;
		Street_max_velocity[i] = rand() % 40 + 45;	/* speed between 45 and 105 */
		Street_neighbors[i] = i;
	}

	// neighbors
	int *Array_Street_offset = new int[NUM_STREETS];
	int *Array_Street_size = new int[NUM_STREETS];
	int num_connections = 0;

	for (int i = 0; i < NUM_STREETS; i++)
	{
		Array_Street_offset[i] = num_connections;
		int connections = rand() % MAX_CONNECTIONS + 1;
		Array_Street_size[i] = connections;
		num_connections += connections;
	}

	int *Array_Street_arrays = new int[num_connections];
	for (int i = 0; i < num_connections; i++)
	{
		Array_Street_arrays[i] = rand() % NUM_STREETS;
	}

	// actors
	int *Actor_street = new int[NUM_PEDS + NUM_CARS];
	float *Actor_progress = new float[NUM_PEDS + NUM_CARS];
	float *Car_max_velocity = new float[NUM_CARS + NUM_PEDS];
	int *Actor_id = new int[NUM_PEDS + NUM_CARS];

	for (int i = 0; i < NUM_PEDS + NUM_CARS; i++)
	{
		Actor_street[i] = rand() % NUM_STREETS;
		Actor_progress[i] = rand() % 10;
		Car_max_velocity[i] = rand() % 20 + 65;
	}

	for (int i = 0; i < NUM_PEDS + NUM_CARS; i++)
	{
		Actor_id[i] = i;
	}

	// jobs (dummy)
	int *jobs = new int[NUM_PEDS + NUM_CARS];

	for (int i = 0; i < NUM_CARS + NUM_PEDS; i++)
	{
		jobs[i] = i;
	}

	// random numbers
	int *randomn = new int[NUM_STREETS];
	for (int i = 0; i < NUM_STREETS; i++)
	{
		// TODO: real random
		randomn[i] = rand() % NUM_STREETS; 
	}

	printf("Scenario set up.\n");

	printf("Converting data to row format...\n");
	struct_Actor *actors = new struct_Actor[NUM_CARS + NUM_PEDS];
	struct_Street *streets = new struct_Street[NUM_STREETS];
	for (int i = 0; i < NUM_PEDS; i++)
	{
		actors[i].progress = Actor_progress[i];
		actors[i].street = Actor_street[i];
		actors[i].tag = TAG_Pedestrian;
	}
	for (int i = NUM_PEDS; i < NUM_CARS + NUM_PEDS; i++)
	{
		actors[i].progress = Actor_progress[i];
		actors[i].street = Actor_street[i];
		actors[i].max_velocity = Car_max_velocity[i];
		actors[i].tag = TAG_Car;
	}
	for (int i = 0; i < NUM_STREETS; i++)
	{
		streets[i].length = Street_length[i];
		streets[i].max_velocity = Street_max_velocity[i];
		streets[i].neighbor_array_index = Street_neighbors[i];
	}

	std::srand(42);
#if !(REORDER)
	random_shuffle(actors, actors + NUM_CARS + NUM_PEDS);
#endif
	printf("Done converting data.\n");

	printf("Copying data to GPU...\n");
	struct_Actor *v_d_Actors;
	struct_Street *v_d_Streets;
	int *v_d_Array_Street_size;
	int *v_d_Array_Street_offset;
	int *v_d_Array_Street_arrays;
	int *v_d_input_actor_tag;
	int *v_d_input_actor_id;
	int *v_d_jobs;
	int *v_d_randomn;

	CudaSafeCall(cudaMalloc((void**) &v_d_Actors, sizeof(struct_Actor) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_Streets, sizeof(struct_Street) * NUM_STREETS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Array_Street_size, sizeof(int) * NUM_STREETS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Array_Street_offset, sizeof(int) * NUM_STREETS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Array_Street_arrays, sizeof(int) * num_connections));
	CudaSafeCall(cudaMalloc((void**) &v_d_input_actor_tag, sizeof(int) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_input_actor_id, sizeof(int) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_jobs, sizeof(int) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_randomn, sizeof(int) * NUM_STREETS));

	CudaSafeCall(cudaMemcpy(v_d_Actors, &actors[0], sizeof(struct_Actor) * (NUM_CARS + NUM_PEDS), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Streets, &streets[0], sizeof(struct_Street) * NUM_STREETS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Array_Street_size, &Array_Street_size[0], sizeof(int) * NUM_STREETS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Array_Street_offset, &Array_Street_offset[0], sizeof(int) * NUM_STREETS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Array_Street_arrays, &Array_Street_arrays[0], sizeof(int) * num_connections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_input_actor_id, &Actor_id[0], sizeof(int) * (NUM_PEDS + NUM_CARS), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_jobs, &jobs[0], sizeof(int) * (NUM_PEDS + NUM_CARS), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_randomn, &randomn[0], sizeof(int) * NUM_STREETS, cudaMemcpyHostToDevice));

	printf("Finished copying data.\n");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("Launching kernel...\n");
	cudaEventRecord(start);
	kernel<<<dim3(32), dim3((NUM_PEDS + NUM_CARS) / 32)>>>(GOOD_WEATHER, 1000000, 
		v_d_Actors, v_d_Streets,
		v_d_Array_Street_size, v_d_Array_Street_offset, v_d_Array_Street_arrays,
		v_d_input_actor_id, v_d_jobs, v_d_randomn);
	CudaCheckError();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	CudaCheckError();
	printf("Kernel finished.\n");

//	cudaMemcpy(Actor_progress, v_d_Actor_progress, sizeof(float) * (NUM_PEDS + NUM_CARS), cudaMemcpyDeviceToHost);
//	for (int i = 0; i < NUM_PEDS + NUM_CARS; i++)
//	{
//		printf(" %f ", Actor_progress[i]);
//	}

//	cudaMemcpy(Actor_street, v_d_Actor_street, sizeof(int) * (NUM_PEDS + NUM_CARS), cudaMemcpyDeviceToHost);
//	for (int i = 0; i < NUM_PEDS + NUM_CARS; i++)
//	{
//		printf(" %i ", Actor_street[i]);
//	}

	printf("\n\n\nElapsed time millis: %f\n", milliseconds);
}
