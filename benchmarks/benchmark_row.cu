#define REORDER 0

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
} struct_Car;

typedef struct
{
	float progress;
	int street;	
} struct_Pedestrian;

typedef struct
{
	float length;
	float max_velocity;
	int neighbor_array_index;
} struct_Street;

typedef struct
{
	int tag;
	int id;
} tag_id_pair;

__device__ struct_Car *d_Cars;
__device__ struct_Pedestrian *d_Pedestrians;
__device__ struct_Street *d_Streets;
__device__ int *d_Array_Street_arrays;
__device__ int *d_Array_Street_offset;
__device__ int *d_Array_Street_size;
__device__ int *d_input_actor_tag;
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

	float speed = min(d_Cars[car_id].max_velocity, d_Streets[d_Cars[car_id].street].max_velocity) * weather_multiplier;
	d_Cars[car_id].progress = d_Cars[car_id].progress + (speed / 60.0); /* 1 tick = 1 minute */

	if (d_Cars[car_id].progress >= d_Streets[d_Cars[car_id].street].length)
	{
		// move to different street
		int array_id = d_Streets[d_Cars[car_id].street].neighbor_array_index;
		int neighbor_index = d_randomn[d_Cars[car_id].street] % d_Array_Street_size[array_id];
		d_Cars[car_id].street = d_Array_Street_arrays[d_Array_Street_offset[array_id] + neighbor_index];
		d_Cars[car_id].progress = 0.0f;
	}
}

__device__ void method_Pedestrian_move(int ped_id, int weather)
{
	float speed = d_randomn[((int) (d_Pedestrians[ped_id].progress*d_Pedestrians[ped_id].progress)) % NUM_STREETS] % 7 - 2;
	d_Pedestrians[ped_id].progress = d_Pedestrians[ped_id].progress + (speed / 60.0);

	if (d_Pedestrians[ped_id].progress >= d_Streets[d_Pedestrians[ped_id].street].length)
	{
		// move to different street
		int array_id = d_Streets[d_Pedestrians[ped_id].street].neighbor_array_index;
		int neighbor_index = d_randomn[d_Pedestrians[ped_id].street] % d_Array_Street_size[array_id];
		d_Pedestrians[ped_id].street = d_Array_Street_arrays[d_Array_Street_offset[array_id] + neighbor_index];
		d_Pedestrians[ped_id].progress = 0.0f;
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

__global__ void kernel(int weather,	int ticks,
	struct_Car *v_d_Cars, struct_Pedestrian *v_d_Pedestrians, struct_Street *v_d_Streets,
	int *v_d_Array_Street_size, int *v_d_Array_Street_offset, int *v_d_Array_Street_arrays,
	int *v_d_input_actor_tag, int *v_d_input_actor_id, int *v_d_jobs, int *v_d_randomn)
{
	d_Cars = v_d_Cars;
	d_Pedestrians = v_d_Pedestrians;
	d_Streets = v_d_Streets;
	d_Array_Street_size = v_d_Array_Street_size;
	d_Array_Street_offset = v_d_Array_Street_offset;
	d_Array_Street_arrays = v_d_Array_Street_arrays;
	d_input_actor_tag = v_d_input_actor_tag;
	d_input_actor_id = v_d_input_actor_id;
	d_jobs = v_d_jobs;
	d_randomn = v_d_randomn;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	__syncthreads();

#if (REORDER)
	block(d_input_actor_tag[d_jobs[tid]], d_input_actor_id[d_jobs[tid]], weather, ticks);
#else
	block(d_input_actor_tag[tid], d_input_actor_id[tid], weather, ticks);
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
	int *Actor_tag = new int[NUM_PEDS + NUM_CARS];
	int *Actor_id = new int[NUM_PEDS + NUM_CARS];

	for (int i = 0; i < NUM_PEDS + NUM_CARS; i++)
	{
		Actor_street[i] = rand() % NUM_STREETS;
		Actor_progress[i] = rand() % 10;
		Car_max_velocity[i] = rand() % 20 + 65;
	}

	tag_id_pair *input_pairs = new tag_id_pair[NUM_PEDS + NUM_CARS];
	for (int i = 0; i < NUM_PEDS; i++)
	{
		input_pairs[i].tag = TAG_Pedestrian;
		input_pairs[i].id = i;
	}
	for (int i = 0; i < NUM_CARS; i++)
	{
		input_pairs[i + NUM_PEDS].tag = TAG_Car;
		input_pairs[i + NUM_PEDS].id = i;
	}

	std::srand(42);
#if !(REORDER)
	random_shuffle(input_pairs, input_pairs + NUM_CARS + NUM_PEDS);
#endif

	for (int i = 0; i < NUM_PEDS + NUM_CARS; i++)
	{
		Actor_tag[i] = input_pairs[i].tag;
		Actor_id[i] = input_pairs[i].id;
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
		randomn[i] = i; 
	}

	printf("Scenario set up.\n");

	printf("Converting data to row format...\n");
	struct_Car *cars = new struct_Car[NUM_CARS];
	struct_Pedestrian *pedestrians = new struct_Pedestrian[NUM_PEDS];
	struct_Street *streets = new struct_Street[NUM_STREETS];
	for (int i = 0; i < NUM_PEDS; i++)
	{
		pedestrians[i].progress = Actor_progress[i];
		pedestrians[i].street = Actor_street[i];
	}
	for (int i = 0; i < NUM_CARS; i++)
	{
		cars[i].progress = Actor_progress[i + NUM_PEDS];
		cars[i].street = Actor_street[i + NUM_PEDS];
		cars[i].max_velocity = Car_max_velocity[i + NUM_PEDS];
	}
	for (int i = 0; i < NUM_STREETS; i++)
	{
		streets[i].length = Street_length[i];
		streets[i].max_velocity = Street_max_velocity[i];
		streets[i].neighbor_array_index = Street_neighbors[i];
	}
	printf("Done converting data.\n");

	printf("Copying data to GPU...\n");
	struct_Pedestrian *v_d_Pedestrians;
	struct_Car *v_d_Cars;
	struct_Street *v_d_Streets;
	int *v_d_Array_Street_size;
	int *v_d_Array_Street_offset;
	int *v_d_Array_Street_arrays;
	int *v_d_input_actor_tag;
	int *v_d_input_actor_id;
	int *v_d_jobs;
	int *v_d_randomn;

	CudaSafeCall(cudaMalloc((void**) &v_d_Pedestrians, sizeof(struct_Pedestrian) * NUM_PEDS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Cars, sizeof(struct_Car) * NUM_CARS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Streets, sizeof(struct_Street) * NUM_STREETS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Array_Street_size, sizeof(int) * NUM_STREETS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Array_Street_offset, sizeof(int) * NUM_STREETS));
	CudaSafeCall(cudaMalloc((void**) &v_d_Array_Street_arrays, sizeof(int) * num_connections));
	CudaSafeCall(cudaMalloc((void**) &v_d_input_actor_tag, sizeof(int) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_input_actor_id, sizeof(int) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_jobs, sizeof(int) * (NUM_PEDS + NUM_CARS)));
	CudaSafeCall(cudaMalloc((void**) &v_d_randomn, sizeof(int) * NUM_STREETS));

	CudaSafeCall(cudaMemcpy(v_d_Pedestrians, &pedestrians[0], sizeof(struct_Pedestrian) * NUM_PEDS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Cars, &cars[0], sizeof(struct_Car) * NUM_CARS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Streets, &streets[0], sizeof(struct_Street) * NUM_STREETS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Array_Street_size, &Array_Street_size[0], sizeof(int) * NUM_STREETS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Array_Street_offset, &Array_Street_offset[0], sizeof(int) * NUM_STREETS, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_Array_Street_arrays, &Array_Street_arrays[0], sizeof(int) * num_connections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(v_d_input_actor_tag, &Actor_tag[0], sizeof(int) * (NUM_PEDS + NUM_CARS), cudaMemcpyHostToDevice));
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
		v_d_Cars, v_d_Pedestrians, v_d_Streets,
		v_d_Array_Street_size, v_d_Array_Street_offset, v_d_Array_Street_arrays, v_d_input_actor_tag,
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
