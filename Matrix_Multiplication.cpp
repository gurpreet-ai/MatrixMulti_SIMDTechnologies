/*  Copyright (C) 2014 by Gurpreet Singh

>   4 x 4 Matrix Multiplication Using the following methods:
>	   1) C Program
>	   2) C with in-line Assembly
>	   3) Intrinsics
>	   4) Vector Classes
>	   5) Automatic Vectorization

    By: Gurpreet Singh, 2014 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <fvec.h>
#include <conio.h>
#include <math.h>
#include <ctime>
#include <Windows.h>

using namespace std;

#define MAX_NUM 10
#define MAX_DIM 64

void C_matrix_multi (float mat_a [MAX_DIM][MAX_DIM], float mat_b[MAX_DIM][MAX_DIM], float mat_result[MAX_DIM][MAX_DIM]);
void asm_matrix_multi (float mat_a [MAX_DIM][MAX_DIM], float mat_b[MAX_DIM][MAX_DIM], float mat_result[MAX_DIM][MAX_DIM]);
void Intrens_matrix_multi (float mat_a [MAX_DIM][MAX_DIM], float mat_b[MAX_DIM][MAX_DIM], float mat_result[MAX_DIM][MAX_DIM]);
void Vec_class_matrix_multi (float mat_a [MAX_DIM][MAX_DIM], float mat_b[MAX_DIM][MAX_DIM], float mat_result[MAX_DIM][MAX_DIM]);
void auto_vec_matrix_multi (float *mat_a, float *mat_b, float *mat_result);

void print_martrix (float mat [MAX_DIM][MAX_DIM]);
void print1d_martrix (float mat [MAX_DIM]);

int main() {
    __declspec(align(16)) float mat_a[MAX_DIM][MAX_DIM];
    __declspec(align(16)) float mat_b[MAX_DIM][MAX_DIM];
    __declspec(align(16)) float C_result[MAX_DIM][MAX_DIM];
    __declspec(align(16)) float Intrens_result[MAX_DIM][MAX_DIM];
    __declspec(align(16)) float Vec_class_result[MAX_DIM][MAX_DIM];

    // srand((unsigned)time(0));

    // Create Matrix A
    for(int i = 0; i < MAX_DIM; i++)
	   for(int j = 0; j < MAX_DIM; j++)
            mat_a[i][j] = rand() % MAX_NUM;
    
    // Create Matrix B
    for(int i = 0; i < MAX_DIM; i++)
	   for(int j = 0; j < MAX_DIM; j++)
            mat_b[i][j] = rand() % MAX_NUM;
    
    cout << "Matrix Multiplication using C \n";
    // Start timing the code.
    __int64 ctr1 = 0, ctr2 = 0, freq = 0;
	if (QueryPerformanceCounter((LARGE_INTEGER *)&ctr1)!= 0)
	{
		// Code segment is being timed.
	    C_matrix_multi (mat_a, mat_b, C_result); 

		// Finish timing the code.
		QueryPerformanceCounter((LARGE_INTEGER *)&ctr2);
		cout << "Start Value: " << ctr1 <<endl; 
		cout << "Start Value: " << ctr2 <<endl; 
	
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
	
		cout << "QueryPerformanceCounter minimum resolution: 1/" <<freq << " Seconds."<<endl;
		cout << "100 Increment time: " << ((ctr2 - ctr1) * 1.0 / freq) << " seconds."<<endl << endl;
	} else {
		DWORD dwError = GetLastError();
		cout<<"Error value = " << dwError <<endl;
	}


    cout << "\nMatrix Multiplication using SSE Intrensics\n";
    __int64 ctr11 = 0, ctr22 = 0, freq1 = 0;
	if (QueryPerformanceCounter((LARGE_INTEGER *)&ctr11)!= 0)
	{
		// Code segment is being timed.
	    Intrens_matrix_multi (mat_a, mat_b, Intrens_result);

		// Finish timing the code.
		QueryPerformanceCounter((LARGE_INTEGER *)&ctr22);
		cout << "Start Value: " << ctr11 <<endl; 
		cout << "Start Value: " << ctr22 <<endl; 
	
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq1);
	
		cout << "QueryPerformanceCounter minimum resolution: 1/" <<freq1 << " Seconds."<<endl;
		cout << "100 Increment time: " << ((ctr22 - ctr11) * 1.0 / freq1) << " seconds."<<endl << endl;
	} else {
		DWORD dwError = GetLastError();
		cout<<"Error value = " << dwError <<endl;
	}
	__declspec(align(16)) float mat_transpose[MAX_DIM][MAX_DIM];

    for(int i = 0; i < MAX_DIM; i++)
	   for(int j = 0; j < MAX_DIM; j++)
		  mat_transpose[i][j] = mat_b[j][i];

    cout << "\nMatrix Multiplication using C++ Vector Classes\n";
    __int64 ctr111 = 0, ctr222 = 0, freq11 = 0;
	if (QueryPerformanceCounter((LARGE_INTEGER *)&ctr111)!= 0)
	{
		// Code segment is being timed.
	    Vec_class_matrix_multi(mat_a, mat_transpose, Vec_class_result);
		// Finish timing the code.
		QueryPerformanceCounter((LARGE_INTEGER *)&ctr222);
		cout << "Start Value: " << ctr111 <<endl; 
		cout << "Start Value: " << ctr222 <<endl; 
	
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq11);
	
		cout << "QueryPerformanceCounter minimum resolution: 1/" <<freq11 << " Seconds."<<endl;
		cout << "100 Increment time: " << ((ctr222 - ctr111) * 1.0 / freq11) << " seconds."<<endl << endl;
	} else {
		DWORD dwError = GetLastError();
		cout<<"Error value = " << dwError <<endl;
	}

    cout << "\nMatrix Multiplication using C++ Auto Vectorization\n";
    __declspec(align(16)) float mat_aa[MAX_DIM * MAX_DIM];
    __declspec(align(16)) float mat_bb[MAX_DIM * MAX_DIM];
    __declspec(align(16)) float auto_Vec_class_result[MAX_DIM * MAX_DIM];
    
    int k = 0;
    for(int i = 0; i < MAX_DIM; i++)
	   for(int j = 0; j < MAX_DIM; j++)
		  mat_aa[k++]   = mat_a[i][j];

    k = 0;
    for(int i = 0; i < MAX_DIM; i++)
	   for(int j = 0; j < MAX_DIM; j++)
		  mat_bb[k++]   = mat_b[i][j];
    
    cout << "\nMatrix Multiplication using C++ Auto Vectorization\n";
    __int64 ctr1111 = 0, ctr2222 = 0, freq111 = 0;
	if (QueryPerformanceCounter((LARGE_INTEGER *)&ctr1111)!= 0)
	{
		// Code segment is being timed.
	    auto_vec_matrix_multi(mat_aa, mat_bb, auto_Vec_class_result);
	    // Finish timing the code.
		QueryPerformanceCounter((LARGE_INTEGER *)&ctr2222);
		cout << "Start Value: " << ctr1111 <<endl; 
		cout << "Start Value: " << ctr2222 <<endl; 
	
		QueryPerformanceFrequency((LARGE_INTEGER *)&freq111);
	
		cout << "QueryPerformanceCounter minimum resolution: 1/" <<freq111 << " Seconds."<<endl;
		cout << "100 Increment time: " << ((ctr2222 - ctr1111) * 1.0 / freq111) << " seconds."<<endl << endl;
	} else {
		DWORD dwError = GetLastError();
		cout<<"Error value = " << dwError <<endl;
	}

    system("pause");
    return 0;
}

void print_martrix (float mat [MAX_DIM][MAX_DIM]) {
    for(int i = 0; i < MAX_DIM; i++) {
	   for(int j = 0; j < MAX_DIM; j++) {
            cout << mat[i][j] << " ";
	   }
	   cout << endl;
    }
}

void print1d_martrix (float mat [MAX_DIM]) {
    for (int i = 0; i < MAX_DIM * MAX_DIM; i+=MAX_DIM) {
	   for (int j = 0; j < MAX_DIM; j++) {
		  cout << mat[i + j] <<" ";
	   }
	   cout << "\n";
    }
}

void C_matrix_multi (float mat_a [MAX_DIM][MAX_DIM], float mat_b[MAX_DIM][MAX_DIM], float mat_result[MAX_DIM][MAX_DIM]) {
    for(int i = 0; i < MAX_DIM; ++i) {
	   for(int j = 0; j < MAX_DIM; ++j) {
		  mat_result[i][j] = 0;
		  for(int k = 0; k < MAX_DIM; k += 4) {
                mat_result[i][j] += mat_a[i][k] * mat_b[k][j];
                mat_result[i][j] += mat_a[i][k+1] * mat_b[k+1][j];
                mat_result[i][j] += mat_a[i][k+2] * mat_b[k+2][j];
                mat_result[i][j] += mat_a[i][k+3] * mat_b[k+3][j];
            }
        }
    }
}

void Intrens_matrix_multi (float mat_a [MAX_DIM][MAX_DIM], float mat_b[MAX_DIM][MAX_DIM], float mat_result[MAX_DIM][MAX_DIM]) {
    __declspec(align(16)) float mat_transpose[MAX_DIM][MAX_DIM];

    for(int i = 0; i < MAX_DIM; i++)
	   for(int j = 0; j < MAX_DIM; j++)
		  mat_transpose[i][j] = mat_b[j][i];

    for(int i = 0; i < MAX_DIM; i++) {
	   for(int j = 0; j < MAX_DIM; j++) {
            __m128 *m3 = (__m128*)mat_a[i];
		  __m128 *m4 = (__m128*)mat_transpose[j];
            float* res;
		  mat_result[i][j] = 0;
		  for(int k = 0; k < MAX_DIM; k += 4) {
                __m128 m5 = _mm_mul_ps(*m3,*m4);
                res = (float*)&m5;
			 mat_result[i][j] += res[0]+res[1]+res[2]+res[3];
                m3++;
                m4++;
            }
        }
    }
}

void Vec_class_matrix_multi (float mat_a [MAX_DIM][MAX_DIM], float mat_b[MAX_DIM][MAX_DIM], float mat_result[MAX_DIM][MAX_DIM]) {
    for(int i = 0; i < MAX_DIM; i++) {
	   for(int j = 0; j < MAX_DIM; j++) {
		  F32vec4 *m3 = (F32vec4*)mat_a[i];
		  F32vec4 *m4 = (F32vec4*)mat_b[j];
		  float* res;
		  mat_result[i][j] = 0;
		  for(int k = 0; k < MAX_DIM; k += 4) {
			 F32vec4 m5 = *m3 * *m4;
			 res = (float*)&m5;
			 mat_result[i][j] += res[0] + res[1] + res[2] + res[3];
			 m3++;
                m4++;
            }
        }
    }
}

void auto_vec_matrix_multi (float *mat_a, float *mat_b, float *mat_result) {
    float sum;
    for (int i = 0; i < MAX_DIM; i++) {
	   for (int j = 0; j < MAX_DIM; j++) {
		  sum = 0.0;
		  for (int k = 0; k < MAX_DIM; k++) {
			 sum = sum + mat_a[i * MAX_DIM + k] * mat_b[k * MAX_DIM + j];
		  }
		  *(mat_result + i * MAX_DIM + j) = sum; 
	   }
    }
}