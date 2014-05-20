/*
 ============================================================================
 Name        : kernel.c
 Author      : Mayowa Aregbesola
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>    // time()
#define _TI_ENHANCED_MATH_H 1
#include <math.h>
#include <sys/time.h>

struct svm_node {
	int idx;				//index
	float value;			//value
};

struct svm_problem {
	int idx;				//index
	float y;				// y value
	struct svm_node *x;	// x value
};

struct svm_parameter {
	int svm_type;
	int kernel_type;
	int degree; /* for poly */
	double gamma; /* for poly/rbf/sigmoid */
	double coef0; /* for poly/sigmoid */

//	/* these are for training only */
//	double cache_size; /* in MB */
//	double eps; /* stopping criteria */
//	double C; /* for C_SVC, EPSILON_SVR and NU_SVR */
//	int nr_weight; /* for C_SVC */
//	int *weight_label; /* for C_SVC */
//	double* weight; /* for C_SVC */
//	double nu; /* for NU_SVC, ONE_CLASS, and NU_SVR */
//	double p; /* for EPSILON_SVR */
//	int shrinking; /* use the shrinking heuristics */
//	int probability; /* do probability estimates */
};

/* Some convenient typedef*/
typedef struct svm_problem Problem;
typedef struct svm_node Node;
typedef struct svm_parameter Parameter;

enum {
	LINEAR, POLY, RBF, SIGMOID
};

unsigned long start_timer();
unsigned long stop_timer(unsigned long start_time, char *name);
void randomInit(float *data, unsigned long int size);
void formK(float *K, Problem* probs, int M, int N);

/* kernel_type */

double dot(const Node *px, const Node *py, int N) {
	double sum = 0;
	while (px->idx != -1 && py->idx != -1) {
		if (px->idx == py->idx) {
			sum += px->value * py->value;
			++px;
			++py;
		} else {
			if (px->idx > py->idx)
				++py;
			else
				++px;
		}
	}
	return sum;
}

/*
double k_function(const Node *x, const Node *y, const Parameter param) {
	switch (param.kernel_type) {
	case LINEAR:
		return dot(x, y);
	case POLY:
		return pow(param.gamma * dot(x, y) + param.coef0, param.degree);
	case RBF: {
		double sum = 0;
		while (x->idx != -1 && y->idx != -1) {
			if (x->idx == y->idx) {
				double d = x->value - y->value;
				sum += d * d;
				++x;
				++y;
			} else {
				if (x->idx > y->idx) {
					sum += y->value * y->value;
					++y;
				} else {
					sum += x->value * x->value;
					++x;
				}
			}
		}

		while (x->idx != -1) {
			sum += x->value * x->value;
			++x;
		}

		while (y->idx != -1) {
			sum += y->value * y->value;
			++y;
		}

		return exp(-param.gamma * sum);
	}
	case SIGMOID:
		return tanh(param.gamma * dot(x, y) + param.coef0);
	default:
		return 0;  // Unreachable
	}
}
 */

void getX( float *x, float *X, int I, int J, int M, int N) {

	for (unsigned int i = I; i < M; i++) {
		for (unsigned int j = J; j < N; j++) {
			x[i*M + j] = X[i*N+j];
		}
	}
}
void formK(float *K, Problem* probs, int M, int N) {

	for (unsigned int i = 0; i < M; i++) {
		for (unsigned int j = 0; j < i; j++) {
//		    T = X(i:20,:) * X(i, :)';
//		    KK(i:20, i) = T;
//		    KK(i, i:20) = T';
			K[i*M + j] = K[j*M+i] = 1.0;//dot(probs[i].x, probs[j].x);
		}
	}
}

void mv(float *y, const float *A, const float *x, unsigned int hA, unsigned int wA) {
	for (unsigned long int i = 0; i < hA; ++i) {
		y[i] = 0.0;
		for (unsigned long int j = 0; j < wA; ++j) {
			y[i] += A[i * wA + j] * x[j];
		}
	}
}

int main(void) {
	int N = 100;
	int M = 20;
	// The number of classes determines the number of problems
	Problem* probs = (Problem*) malloc(M * sizeof(Problem));
	// Initialize the parameters for the SVM one class
	Parameter param;
//	Node *x_space;

	// Type of SVM
	param.svm_type = 0;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;
	param.coef0 = 0;
//	param.nu = 0.05;
//	param.cache_size = 1000;
//	param.C = 1;
//	param.eps = 1e-3;
//	param.p = 0.1;
//	param.shrinking = 1;
//	param.probability = 0;
//	param.nr_weight = 0;
//	param.weight_label = NULL;
//	param.weight = NULL;

	// Generate random Matrix and vector
	unsigned long vector_start_time = start_timer();
	unsigned long int mem_size_X = sizeof(float) * M * N;
	unsigned long int mem_size_K = sizeof(float) * M * M;
	unsigned long int mem_size_Y = sizeof(float) * M;
	float *Y = (float *) malloc(mem_size_Y);
	float *X = (float *) malloc(mem_size_X);
	float *K = (float *) malloc(mem_size_K);
	// initialize host memory
	srand(time(NULL ));
	randomInit(X, M * N);
	srand(time(NULL ));
	randomInit(Y, M);

	Node* x_space = malloc(N * sizeof(Node*));
	for (unsigned int i = 0; i < M; i++) {
		probs[i].idx = i;
		probs[i].y = Y[i];

		for (unsigned int j = 0; j < N; j++) {
			x_space[j].idx = j;
			x_space[j].value = X[i * N + j];
		}
		probs[i].x = x_space;

	}
	stop_timer(vector_start_time, "Vector generation");

	printf("\nUsing LINEAR\n");
	unsigned long ker_start_time = start_timer();
	//formK(K, probs, M);

	stop_timer(ker_start_time, "Time LINEAR\t");

	printf("\nUsing POLY\n");
	ker_start_time = start_timer();
	stop_timer(ker_start_time, "Time POLY\t");

	printf("\nUsing RBF\n");
	ker_start_time = start_timer();
	stop_timer(ker_start_time, "Time RBF\t");

	printf("\nUsing SIGMOID\n");
	ker_start_time = start_timer();
	stop_timer(ker_start_time, "Time SIGMOID\t");

	return EXIT_SUCCESS;
}

// Allocates a matrix with random float entries.
void randomInit(float *data, unsigned long int size) {
	for (unsigned long int i = 0; i < size; ++i) {
		data[i] = rand() / (float) RAND_MAX;
		data[i] = round(data[i] * 100.0) / 100.0;
		data[i] = 2.0 * (data[i]) - 1;
	}
}

// Returns the current time in microseconds
unsigned long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL );
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

// Prints the time elapsed since the specified time
unsigned long stop_timer(unsigned long start_time, char *name) {
	struct timeval tv;
	gettimeofday(&tv, NULL );
	unsigned long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
	printf("%s: %.5f sec\n", name,
			((float) (end_time - start_time)) / (1000000));
	return end_time - start_time;
}
