#include<stdio.h>
#include<stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include<vector>
#include <math.h>
#include <iomanip>
#include <omp.h>
using namespace std;

struct MatUnion {
	double* m_left;   // left matrix
	double* m_right;  // right matrix
	int dim_x;
	int dim_y;
	int dim_z;
};

void mult_dim(int dim_y_x, int dim_y, int dim_z);	
MatUnion init_matrix(string file_path);     
string* mat_mult_ser_base(double* mat_left, double* mat_right, int X, int Y, int Z);  
void print_mat(double* mat, int X, int Y);
int total_threads_num = 32;


void mult_dim(int dim_x, int dim_y, int dim_z) {
	cout << "matrix multiplication:(" << dim_x << "*" << dim_y << ")X(" << dim_y << "*" << dim_z << ")=("
		<< dim_x << "*" << dim_z << ")" << endl;
}






void SplitString(const string& s_single, vector<double>& v_keep, const string& c_sep) {
	std::string::size_type pos1, pos2;
	pos2 = s_single.find(c_sep);
	pos1 = 0;
	while (std::string::npos != pos2) {
		double num = atof(s_single.substr(pos1, pos2 - pos1).c_str());
		v_keep.push_back(num);
		pos1 = pos2 + c_sep.size();
		pos2 = s_single.find(c_sep, pos1);
	}

	if (pos1 != s_single.length()) {//the last element in a row
		double num = atof(s_single.substr(pos1).c_str());
		v_keep.push_back(num);
	}

}



MatUnion init_matrix(string file_name) {
	MatUnion mat_union;
	string s_line;
	vector<double> v_mat;
	ifstream in_file;
	in_file.open("input.txt");
	if (!in_file.is_open()) {
		cout << "Can not find " << file_name << endl;
		system("pause");
	}
	getline(in_file, s_line);      //get the size of the first matrix
	SplitString(s_line, v_mat, ",");
	mat_union.dim_x = int(v_mat[0]);
	mat_union.dim_y = int(v_mat[1]);
	mat_union.m_left = new double[mat_union.dim_x * mat_union.dim_y];

	vector<double>().swap(v_mat);
	int count = 0; 
	//create the array to store each line of text
	string* s_line_array_first = new string[mat_union.dim_x];
	string* s_line_array_second = new string[mat_union.dim_y];
	//divide the text into 100 blocks
	int block_num = 100;
	// the size of each block
	int block_size_1 = mat_union.dim_x / block_num;
	int block_size_2 = mat_union.dim_y / block_num;
	for (int iteration = 0;iteration < block_num;iteration++) {
		if (iteration == 0) {// the master thread need to get the first block
			for (count = 0;count < block_size_1;count++) {
				getline(in_file, s_line_array_first[count]);
			}
		}
		int master_ptr = iteration + 1;
		if (master_ptr == block_num) {
			getline(in_file, s_line);      //get the size of the second matrix
			SplitString(s_line, v_mat, ",");
			mat_union.dim_z = int(v_mat[1]);
			int dim_z = int(v_mat[1]);
			mat_union.m_right = new double[mat_union.dim_y * mat_union.dim_z];
			vector<double>().swap(v_mat);
		}
#pragma omp parallel
		{
#pragma omp master
			{
				if (master_ptr != block_num) {
					for (count = master_ptr  * block_size_1;count < (master_ptr + 1) * block_size_1;count++) {
						getline(in_file, s_line_array_first[count]);
					}
				}
				else {
					for (count = 0;count < block_size_2;count++) {//we have read all the string of matrix A and start to read the first block of Matrix B
						getline(in_file, s_line_array_second[count]);
					}
				}
			}
#pragma omp for schedule(static) private(v_mat) 
			for (count = iteration * block_size_1;count < (iteration + 1) * block_size_1;count++) {
				SplitString(s_line_array_first[count], v_mat, ",");
				for (int i = 0; i < v_mat.size(); i++) {
					mat_union.m_left[count * mat_union.dim_y + i] = v_mat[i];
				}
				vector<double>().swap(v_mat);
			}
		}

	}
	delete[] s_line_array_first;
	for (int iteration = 0;iteration < block_num;iteration++) {
		int master_ptr = iteration + 1;
#pragma omp parallel
	{
#pragma omp master
		{
			if (master_ptr != block_num) {// all the blocks are read when master_ptr equals block_num here
				for (count = master_ptr  * block_size_2;count <( master_ptr  + 1) * block_size_2;count++) {
					getline(in_file, s_line_array_second[count]);
				}
			}
		}
#pragma omp for schedule(static) private(v_mat) 
		for (count = iteration  * block_size_2;count < (iteration + 1) * block_size_2;count++) {
			SplitString(s_line_array_second[count], v_mat, ",");
			for (int i = 0; i < v_mat.size(); i++) {
				mat_union.m_right[count * mat_union.dim_z + i] = v_mat[i];
			}
			vector<double>().swap(v_mat);
		}
	}
}

	delete[] s_line_array_second;
	in_file.close();
	return mat_union;
}



string* mat_mult_ser_base(double* mat1, double* mat2, int dim_x, int dim_y, int dim_z) {
	cout << mat1[0] << " " << mat2[0] << endl;
	double* result_data = new double[dim_x * dim_z];
	string* result = new string[dim_x];
	cout << "calculating" << endl;

#pragma omp parallel 
	{
#pragma omp for schedule(static) 
		for (int x = 0; x < dim_x; x++) {
			for (int z = 0; z < dim_z; z++) {
				*(result_data + (z + x * dim_z)) = 0;
				for (int y = 0; y < dim_y; y++) {
					*(result_data + (x * dim_z + z)) += (*(mat1 + (x * dim_y + y))) * (*(mat2 + (y * dim_z + z)));
				}
				// add each element to the result string
				//the next three lines have better performance or efficiency than using the function setprecision()
				// By storing the output into a array, we have the ability to store a sequential output by parallel computing
				long long temp = (long long)((*((result_data + (x * dim_z + z))) * 100) + .5);
				double new_element = (double)temp / 100;
				result[x] += to_string(new_element).substr(0, 12);
				if (z != dim_z - 1) {
					result[x] += ",";
				}
			}
		}
	}
	delete[] result_data;
	return result;
}




void res_write_array(string result[]) {
	ofstream fout("output.txt");
	for (int i = 0;i < 4900;i++) {
		fout << result[i] << endl;
	}
	fout.close();
}

void print_mat(double* mat, int Y, int X) {

	for (int y = 0; y < Y; y++) {
		for (int x = 0; x < X; x++) {
			cout << *(mat + (x + y * X)) << "\t";
		}
		cout << endl;
	}
	cout << endl;
}


#define IS_SHOW false



int main() {
	double start, end;
	start = omp_get_wtime();
	MatUnion mat = init_matrix("input.txt"); 
	end = omp_get_wtime();
	double time = end - start;
	cout << "reading time: " << time << " seconds" << endl;
	if (IS_SHOW)            //whether to print the matrix 
	{
		printf("mat_left:\n");
		print_mat(mat.m_left, mat.dim_x, mat.dim_y);
		printf("mat_right:\n");
		print_mat(mat.m_right, mat.dim_y, mat.dim_z);
	}
	mult_dim(mat.dim_x, mat.dim_y, mat.dim_z);    
	string* result = new string[mat.dim_x];
	start = omp_get_wtime();
	result = mat_mult_ser_base(mat.m_left, mat.m_right, mat.dim_x, mat.dim_y, mat.dim_z);
	end = omp_get_wtime();
	time = end - start;
	cout << "Parallel run time:" << time << " seconds" << endl;
	clock_t s_start, s_end;
	s_start = clock();
	res_write_array(result);
	s_end = clock();
	double s_time = s_end - s_start;
	cout << "The total write time is" << s_time / CLOCKS_PER_SEC << "seconds" << endl;
	cout << "result write success" << endl;
	// relase the memory
	delete[] mat.m_left;
	delete[] mat.m_right;
	delete[] result;
	return 0;
}
