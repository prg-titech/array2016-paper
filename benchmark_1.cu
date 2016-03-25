
#define GOOD_WEATHER 0
#define BAD_WEATHER 1

#define TAG_Car 0
#define TAG_Pedestrian 1

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <array> 
#include <algorithm> 

#include <curand.h>
#include <curand_kernel.h>

#define NUM_CARS 1024
#define NUM_PEDS 4096
#define NUM_STREETS 500
#define MAX_CONNECTIONS 5
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

__device__ float *d_Actor_progress;
__device__ int *d_Actor_street;
__device__ float *d_Car_max_velocity;
__device__ float *d_Street_length;
__device__ float *d_Street_max_velocity;
__device__ int *d_Street_neighbors;
__device__ int *d_Array_Street_size;
__device__ int *d_Array_Street_offset;
__device__ int *d_Array_Street_arrays;
__device__ int *d_input_actor_tag;
__device__ int *d_input_actor_id;
__device__ int *d_jobs;
__shared__ curandState_t d_rand_state;

__device__ void method_Car_move(int actor_id, int weather)
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

	float speed = min(d_Car_max_velocity[actor_id], d_Street_max_velocity[d_Actor_street[actor_id]]) * weather_multiplier;
	d_Actor_progress[actor_id] = d_Actor_progress[actor_id] + (speed / 60.0); /* 1 tick = 1 minute */

	if (d_Actor_progress[actor_id] >= d_Street_length[d_Actor_street[actor_id]])
	{
		// move to different street
		int array_id = d_Street_neighbors[d_Actor_street[actor_id]];
		int neighbor_index = curand(&d_rand_state) % d_Array_Street_size[array_id];
		d_Actor_street[actor_id] = d_Array_Street_arrays[d_Array_Street_offset[array_id] + neighbor_index];
	}
}

__device__ void method_Pedestrian_move(int actor_id, int weather)
{
	float speed = curand(&d_rand_state) % 7 - 2;
	d_Actor_progress[actor_id] = d_Actor_progress[actor_id] + (speed / 60.0); /* 1 tick = 1 minute */

	if (d_Actor_progress[actor_id] >= d_Street_length[d_Actor_street[actor_id]])
	{
		// move to different street
		int array_id = d_Street_neighbors[d_Actor_street[actor_id]];
		int neighbor_index = curand(&d_rand_state) % d_Array_Street_size[array_id];
		d_Actor_street[actor_id] = d_Array_Street_arrays[d_Array_Street_offset[array_id] + neighbor_index];
	}
}

__device__ void block(int actor_tag, int actor_id, int weather, int ticks)
{
	for (int i = 0; i < ticks; i++)
	{
		switch (actor_tag)
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

__global__ void kernel(int weather,	int ticks)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x == 1)
	{
		curand_init(42, 0, 0, &d_rand_state);
	}

	__syncthreads();

	block(d_input_actor_tag[d_jobs[tid]], d_input_actor_id[d_jobs[tid]], weather, ticks);
}

int main()
{
	printf("Setting up scenario...\n");
	srand(42);

	// streets
	int *Street_length = new int[NUM_STREETS];
	int *Street_max_velocity = new int[NUM_STREETS];
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
	int *Car_max_velocity = new int[NUM_CARS + NUM_PEDS];
	int *Actor_tag = new int[NUM_PEDS + NUM_CARS];
	int *Actor_id = new int[NUM_PEDS + NUM_CARS];

	for (int i = 0; i < NUM_PEDS + NUM_CARS; i++)
	{
		Actor_street[i] = rand() % NUM_STREETS;
		Actor_progress[i] = 0.0f;
		Car_max_velocity[i] = rand() % 20 + 65;
	}

	for (int i = 0; i < NUM_PEDS; i++)
	{
		Actor_tag[i] = TAG_Pedestrian;
		Actor_id[i] = i;
	}

	for (int i = NUM_PEDS; i < NUM_PEDS + NUM_CARS; i++)
	{
		Actor_tag[i] = TAG_Car;
		Actor_id[i] = i;
	}

	//shuffle(Actor_tag, Actor_tag + NUM_CARS + NUM_PEDS, std::default_random_engine(42));

	// jobs (dummy)
	int *jobs = new int[NUM_PEDS + NUM_CARS];

	for (int i = 0; i < NUM_CARS + NUM_PEDS; i++)
	{
		jobs[i] = i;
	}

	printf("Scenario set up.\n");

	printf("Copying data to GPU...\n");
	CudaSafeCall(cudaMemcpyToSymbol(d_Actor_progress, &Actor_progress[0], sizeof(float) * (NUM_PEDS + NUM_CARS), size_t(0), cudaMemcpyHostToDevice));
	cudaMemcpyToSymbol("d_Actor_street", &Actor_street[0], sizeof(int) * (NUM_PEDS + NUM_CARS), size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("d_Car_max_velocity", &Car_max_velocity[0], sizeof(float) * (NUM_PEDS + NUM_CARS), size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("d_Street_length", &Street_length[0], sizeof(float) * NUM_STREETS, size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("d_Street_max_velocity", &Street_max_velocity[0], sizeof(float) * NUM_STREETS, size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("d_Street_neighbors", &Street_neighbors[0], sizeof(int) * NUM_STREETS, size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("d_Array_Street_size", &Array_Street_size[0], sizeof(int) * NUM_STREETS, size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("d_Array_Street_offset", &Array_Street_offset[0], sizeof(int) * NUM_STREETS, size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("d_Array_Street_arrays", &Array_Street_arrays[0], sizeof(int) * num_connections, size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("d_input_actor_tag", &Actor_tag[0], sizeof(int) * (NUM_PEDS + NUM_CARS), size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("d_input_actor_id", &Actor_id[0], sizeof(int) * (NUM_PEDS + NUM_CARS), size_t(0), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("d_jobs", &jobs[0], sizeof(int) * (NUM_PEDS + NUM_CARS), size_t(0), cudaMemcpyHostToDevice);
	printf("Finished copying data.\n");

	printf("Launching kernel...\n");
	kernel<<<dim3(32), dim3((NUM_PEDS + NUM_CARS) / 32)>>>(GOOD_WEATHER, 1000000);
	CudaCheckError();
	printf("Kernel finished.\n");

	//cudaMemcpy(Actor_progress, d_Actor_progress,)
}