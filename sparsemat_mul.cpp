#include <iostream>
#include <mpi.h>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <algorithm>
using namespace std;

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N, M, P;
    vector<double> A_values, B_values;
    vector<int> A_colidx, B_colidx;
    vector<int> A_rowptr, B_rowptr;

    // decide input source
    istream* input_stream = nullptr;
    ifstream fin;

    if (argc == 2) {
        if (rank == 0) {
            fin.open(argv[1]);
            if (!fin) {
                cerr << "Error: cannot open input file\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        input_stream = &fin;
    } else {
        // benchmark.py will pipe input here
        input_stream = &cin;
    }

    if (rank == 0) {
        // dimensions
        (*input_stream) >> N >> M >> P; // A - N x M, B - M x P

        A_rowptr.push_back(0);
        B_rowptr.push_back(0);

        // A to csr
        for (int i = 0; i < N; i++) {
            int k;
            (*input_stream) >> k;
            for (int j = 0; j < k; j++) {
                int col;
                double val;
                (*input_stream) >> col >> val;
                A_colidx.push_back(col);
                A_values.push_back(val);
            }
            A_rowptr.push_back(A_rowptr.back() + k);
        }

        // B to csr
        for (int i = 0; i < M; i++) {
            int k;
            (*input_stream) >> k;
            for (int j = 0; j < k; j++) {
                int col;
                double val;
                (*input_stream) >> col >> val;
                B_colidx.push_back(col);
                B_values.push_back(val);
            }
            B_rowptr.push_back(B_rowptr.back() + k);
        }
    }

    if (fin.is_open()) fin.close();


    // broadcast B----------------------------------------------------------------------------------------------------------------------------------------------
    int Bval_cnt, rowptr_sizeB;

    if (rank == 0) {

        Bval_cnt = B_values.size();
        rowptr_sizeB = B_rowptr.size();

    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Bval_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowptr_sizeB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        B_values.resize(Bval_cnt);
        B_colidx.resize(Bval_cnt);
        B_rowptr.resize(rowptr_sizeB);
    }

    MPI_Bcast(B_values.data(), Bval_cnt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_colidx.data(), Bval_cnt, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_rowptr.data(), rowptr_sizeB, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // scatter A---------------------------------------------------------------------------------------------------------------------------------------------

    int rowsperproc = N / size;
    int rem = N % size;
    
    vector<int> sendcount(size, 0);  
    vector<int> offset(size, 0);    
    vector<int> rowcount(size, 0);   // rows/proc cnt
    vector<int> rowoffset(size, 0);  // starting row index

    vector<double> local_A_values;
    vector<int> local_A_colidx, local_A_rowptr;

    if (rank == 0) {
        //roes per proc
        for (int i = 0; i < size; i++) {
            rowcount[i] = rowsperproc + (i < rem ? 1 : 0);
            rowoffset[i] = (i == 0) ? 0 : rowoffset[i-1] + rowcount[i-1];
        }
        // nnze per proc
        for (int i = 0; i < size; i++) {
            int start_row = rowoffset[i];
            int end_row = start_row + rowcount[i];
            
            if (end_row > start_row) {
                sendcount[i] = A_rowptr[end_row] - A_rowptr[start_row];
                offset[i] = A_rowptr[start_row];
            }
        }
    }

    int local_cnt;
    MPI_Scatter(sendcount.data(), 1, MPI_INT, &local_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize 
    local_A_values.resize(local_cnt);
    local_A_colidx.resize(local_cnt);
    MPI_Barrier(MPI_COMM_WORLD);

    
    MPI_Scatterv(A_values.data(), sendcount.data(), offset.data(), MPI_DOUBLE,local_A_values.data(), local_cnt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(A_colidx.data(), sendcount.data(), offset.data(), MPI_INT,local_A_colidx.data(), local_cnt, MPI_INT, 0, MPI_COMM_WORLD);

   //comput local rowptr
    vector<int> rowptr_sendcount(size, 0);
    vector<int> rowptr_offset(size, 0);
    vector<vector<int>> local_rowptrs(size);

    if (rank == 0) {
        
        for (int proc = 0; proc < size; proc++) {
            int start_row = rowoffset[proc];
            int num_rows = rowcount[proc];
            
            if (num_rows > 0) {
                local_rowptrs[proc].push_back(0);  
                
                for (int i = 1; i <= num_rows; i++) {
                    int global_row = start_row + i;
                    int local_nnz = A_rowptr[global_row] - A_rowptr[start_row];
                    local_rowptrs[proc].push_back(local_nnz);
                }
                
                rowptr_sendcount[proc] = local_rowptrs[proc].size();
                rowptr_offset[proc] = (proc == 0) ? 0 : rowptr_offset[proc-1] + rowptr_sendcount[proc-1];
            }
        }
        vector<int> all_local_rowptrs;
        for (int proc = 0; proc < size; proc++) {
            all_local_rowptrs.insert(all_local_rowptrs.end(), local_rowptrs[proc].begin(),local_rowptrs[proc].end());
        }

        // Scatter row pointers
        int my_rowptr_size;
        MPI_Scatter(rowptr_sendcount.data(), 1, MPI_INT, &my_rowptr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        local_A_rowptr.resize(my_rowptr_size);

        MPI_Scatterv(all_local_rowptrs.data(), rowptr_sendcount.data(), rowptr_offset.data(), MPI_INT,local_A_rowptr.data(), my_rowptr_size, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        int my_rowptr_size;
        MPI_Scatter(rowptr_sendcount.data(), 1, MPI_INT, &my_rowptr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        local_A_rowptr.resize(my_rowptr_size);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,local_A_rowptr.data(), my_rowptr_size, MPI_INT, 0, MPI_COMM_WORLD);
    }

     MPI_Barrier(MPI_COMM_WORLD);

     //compute c---------------------------------------------------------------------------------------------------------------------------------------------

    int local_rows = (local_A_rowptr.size() > 0) ? local_A_rowptr.size() - 1 : 0;
   
    vector<unordered_map<int, double>> local_C_sparse(local_rows);
    vector<double> local_C_values;
    vector<int> local_C_colidx;
    vector<int> local_C_rowptr;
    
    if (local_rows > 0) {
        //each row of A
        for (int i = 0; i < local_rows; i++) {
            int start = local_A_rowptr[i];
            int end = local_A_rowptr[i + 1];
            
            // nzele
            for (int j = start; j < end; j++) {
                int a_col = local_A_colidx[j];  
                double a_val = local_A_values[j];
                
                // corresponding row of B
                int b_start = B_rowptr[a_col];
                int b_end = B_rowptr[a_col + 1];
                
                for (int k = b_start; k < b_end; k++) {
                    int b_col = B_colidx[k];
                    double b_val = B_values[k];
                    local_C_sparse[i][b_col] += a_val * b_val;
                }
            }
        }
        local_C_rowptr.push_back(0);
        for (int i = 0; i < local_rows; i++) {
            
            vector<pair<int, double>> row_elements;
            for (auto& elem : local_C_sparse[i]) {
                if (abs(elem.second) > 1e-12) { 
                    row_elements.push_back({elem.first, elem.second});
                }
            }
            sort(row_elements.begin(), row_elements.end());
            
            // Add  CSR vectors
            for (auto& elem : row_elements) {
                local_C_colidx.push_back(elem.first);
                local_C_values.push_back(elem.second);
            }
            local_C_rowptr.push_back(local_C_values.size());
        }
    } else {

        local_C_rowptr.push_back(0);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // GATHER c-----------------------------------------------------------------------------------------------------------------------------------------------------

    int local_nnz = local_C_values.size();
    vector<int> all_nnz(size, 0);
    MPI_Gather(&local_nnz, 1, MPI_INT, all_nnz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather nnze per proc in c
    vector<int> all_rows(size, 0);
    MPI_Gather(&local_rows, 1, MPI_INT, all_rows.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> nnz_offsets(size, 0);
    vector<int> row_offsets(size, 0);
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            nnz_offsets[i] = nnz_offsets[i-1] + all_nnz[i-1];
            row_offsets[i] = row_offsets[i-1] + all_rows[i-1];
        }
    }
    // Gather values and colidx
    vector<double> final_C_values;
    vector<int> final_C_colidx;
    if (rank == 0) {
        int total_nnz = nnz_offsets[size-1] + all_nnz[size-1];
        final_C_values.resize(total_nnz);
        final_C_colidx.resize(total_nnz);
    }

    MPI_Gatherv(local_C_values.data(), local_nnz, MPI_DOUBLE,final_C_values.data(), all_nnz.data(), nnz_offsets.data(), MPI_DOUBLE,0, MPI_COMM_WORLD);
    MPI_Gatherv(local_C_colidx.data(), local_nnz, MPI_INT,final_C_colidx.data(), all_nnz.data(), nnz_offsets.data(), MPI_INT,0, MPI_COMM_WORLD);

    // Gather row pointers
    vector<int> rowptr_counts(size, 0);
    vector<int> rowptr_offsets(size, 0);
    
    if (rank == 0) {
        rowptr_counts[0] = local_C_rowptr.size();
        for (int i = 1; i < size; i++) {//excluding 0
            rowptr_counts[i] = all_rows[i];  
            rowptr_offsets[i] = rowptr_offsets[i-1] + rowptr_counts[i-1];
        }
    }
    // Send row pointer counts
    int my_rowptr_count;
    MPI_Scatter(rowptr_counts.data(), 1, MPI_INT, &my_rowptr_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> send_rowptr;
    if (rank == 0) {
        send_rowptr = local_C_rowptr; 
    } else {
        
        for (int i = 1; i < local_C_rowptr.size(); i++) {
            send_rowptr.push_back(local_C_rowptr[i] + nnz_offsets[rank]);
        }
    }
    vector<int> final_C_rowptr;
    if (rank == 0) {
        int total_rowptr_size = rowptr_offsets[size-1] + rowptr_counts[size-1];
        final_C_rowptr.resize(total_rowptr_size);
    }
    MPI_Gatherv(send_rowptr.data(), my_rowptr_count, MPI_INT,final_C_rowptr.data(), rowptr_counts.data(), rowptr_offsets.data(), MPI_INT,0, MPI_COMM_WORLD);
    
    if (rank == 0) {

        //final c 
        for (int proc = 1; proc < size; proc++){
            int start_idx = rowptr_offsets[proc];
            int count = rowptr_counts[proc];
            for (int i = 0; i < count; i++) {
                final_C_rowptr[start_idx + i] += nnz_offsets[proc];
            }
        }
        
           // Write result to stdout (benchmark.py captures this)
        if (rank == 0) {
            for (int i = 0; i < N; i++) {
                int row_start = final_C_rowptr[i];
                int row_end   = final_C_rowptr[i+1];
                int k = row_end - row_start;  

                if (k == 0) {
                    cout << "0\n";   
                    continue;
                }

                cout << k;
                for (int j = row_start; j < row_end; j++) {
                    cout << " " << final_C_colidx[j] << " " << final_C_values[j];
                }
                cout << "\n";
            }
            cout.flush();
        }

    }
    MPI_Finalize();

    return 0;
}