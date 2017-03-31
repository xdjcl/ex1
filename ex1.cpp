/* 
	First ex to test the Sync CD algorithm 
	JingChang Liu
	2017 03 30
*/

#include<stdio.h>
#include<math.h>
#include<algorithm>
#include<mpi.h>

#define M 1000 // M 个有限和 || 优化变量个数
#define N 11 // 调用的processes 的数目
#define mu 0.1 // 参数
#define s 15 // 最大delay

double f(double * , const int); // return f(x)
double par_f(double); // return f 偏导
double prox_l1(double, double); //计算 l1-norm prox值

int main()
{
	int argc;
	char **argv;
	int iter[N-1] = {0}; //记录各个workers的迭代次数
	int process_num, process_rank; //进程数目，rank
	int x[M] = { 0 }; // 优化变量
	int k; // 迭代次数

	MPI_Init(&argc,  &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &process_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

	if (process_num != N)
	{
		printf("Processes number != N ! \n");
		return 0;
	}

	// Master
	else if (process_num == 0)
	{
		double U[M] = { 0 }; //接收的更新值
		MPI_Status master_status;

		for (int i = 0; i < M; i++)
		{
			MPI_Recv(U + i, 1, MPI_DOUBLE, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, &master_status);
			x[i] = x[i] + U[i];
		}
		printf("Iter = %d", iter);
		k = k + 1;
	}

	// Workers
	else
	{
		double local_x[M]; // 定义 workers 上的局部x
		double local_U[M]; // workers 上的更新数值
		MPI_Status worker_status;
		// 从 Master 之中接收数据
		MPI_Recv(local_x, M,   , 0, 99, MPI_COMM_WORLD, &worker_status);

		// 当 delay 在容忍范围中
		if (*std::max_element(iter, iter + N - 1) - iter[process_rank] < s)
		{
			//每个进程计算、发送一部分值
			// 1进程： 1,11,21,31，，，，
			// 2进程： 2,12,22,32，，，
			for (int i = 1; i <= M; i += (N - 1))
			{
				local_U[i] = prox_l1(local_x[i] - mu * par_f(local_x[i]), 0.1) - local_x[i];
				// 将更新数值发总给master
				MPI_Send(local_U + i, 1, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
			}
		}

		// 该worker的迭代次数将+1
		iter[process_rank] = iter[process_rank] + 1;
	}

	MPI_Finalize();
}

// 计算f(x), n: 数组大小
double f(double *x, const int n)
{
	int sum; 
	for (int i = 0; i < n; i++)
	{
		sum = sum + log(1 + mu * (x[i] - 1) * (x[i] - 1));
	}

	return 0.5*sum;
}

// 计算 f 关于 x 的偏导
double par_f(double x)
{
	return mu * (x - 1) / (1 + (x - 1) * (x - 1)) ;
}

// f = ||.||_1, 求 f 的 prox 值 eta: 参数值
double prox_l1(double x, const double eta)
{
	return fmax(0, x - 1.0 / eta) - fmax(0, -x - 1.0 / eta);
}
