/* 
	First ex to test the Sync CD algorithm 
	JingChang Liu
	2017 03 30
*/

#include<stdio.h>
#include<math.h>
#include<algorithm>
#include<mpi.h>

#define M 1000 // M �����޺� || �Ż���������
#define N 11 // ���õ�processes ����Ŀ
#define mu 0.1 // ����
#define s 15 // ���delay

double f(double * , const int); // return f(x)
double par_f(double); // return f ƫ��
double prox_l1(double, double); //���� l1-norm proxֵ

int main()
{
	int argc;
	char **argv;
	int iter[N-1] = {0}; //��¼����workers�ĵ�������
	int process_num, process_rank; //������Ŀ��rank
	int x[M] = { 0 }; // �Ż�����
	int k; // ��������

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
		double U[M] = { 0 }; //���յĸ���ֵ
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
		double local_x[M]; // ���� workers �ϵľֲ�x
		double local_U[M]; // workers �ϵĸ�����ֵ
		MPI_Status worker_status;
		// �� Master ֮�н�������
		MPI_Recv(local_x, M,   , 0, 99, MPI_COMM_WORLD, &worker_status);

		// �� delay �����̷�Χ��
		if (*std::max_element(iter, iter + N - 1) - iter[process_rank] < s)
		{
			//ÿ�����̼��㡢����һ����ֵ
			// 1���̣� 1,11,21,31��������
			// 2���̣� 2,12,22,32������
			for (int i = 1; i <= M; i += (N - 1))
			{
				local_U[i] = prox_l1(local_x[i] - mu * par_f(local_x[i]), 0.1) - local_x[i];
				// ��������ֵ���ܸ�master
				MPI_Send(local_U + i, 1, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
			}
		}

		// ��worker�ĵ���������+1
		iter[process_rank] = iter[process_rank] + 1;
	}

	MPI_Finalize();
}

// ����f(x), n: �����С
double f(double *x, const int n)
{
	int sum; 
	for (int i = 0; i < n; i++)
	{
		sum = sum + log(1 + mu * (x[i] - 1) * (x[i] - 1));
	}

	return 0.5*sum;
}

// ���� f ���� x ��ƫ��
double par_f(double x)
{
	return mu * (x - 1) / (1 + (x - 1) * (x - 1)) ;
}

// f = ||.||_1, �� f �� prox ֵ eta: ����ֵ
double prox_l1(double x, const double eta)
{
	return fmax(0, x - 1.0 / eta) - fmax(0, -x - 1.0 / eta);
}
