#include <iostream>
#include <random>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>
using namespace std;


void coo_to_csr(vector<pair<int,int>>& coo, int num_row, int num_col, string file_name) {
  int num_nonzero = coo.size();
  vector<int> uncompressed(num_row+1);
  for (pair<int, int>& coor : coo) {
    uncompressed[coor.first+1]++;
  }
  for (int i = 0; i < num_row; i++) {
    uncompressed[i+1] += uncompressed[i];
  }
 
  ofstream out("./" + file_name+".csr", ios::binary | ios::out);
  out.write((char*)&num_row, sizeof(int));
  out.write((char*)&num_col, sizeof(int));
  out.write((char*)&num_nonzero, sizeof(int));

  for (int i = 0; i<num_row+1; i++) {
    int src = uncompressed[i];
    out.write((char*)&src, sizeof(src));
  }

  for (int i = 0; i<num_nonzero; i++) {
    int dst = coo[i].second;
    out.write((char*)&dst, sizeof(dst));
  }
  out.close();
}

int main(int argc, char* argv[]) {
  //vector<pair<float,string>> ss;
  if (argc != 2) { cout << "Wrong arguments" << endl; exit(-1); }
  string arg(argv[1]);
  fstream arg_file(arg);
  vector<string> mtx_names;
  string mtx_name;
  for (; getline(arg_file, mtx_name);) {
    mtx_names.push_back(mtx_name);
  }

  int avgrow = 0;
  int avgcol = 0;
  int avgnum = 0;

  #pragma omp parallel for schedule(dynamic,1)
  for (int i = 0; i<mtx_names.size(); i++) {
    string mtx_name = mtx_names[i];
    random_device rd;
    mt19937 gens(rd());
    default_random_engine gen(chrono::system_clock::now().time_since_epoch().count() * (omp_get_thread_num()+1));
 
    //////////////////////////
    // Reading CSR A from file
    //////////////////////////
    int num_row, num_col, num_nonzero;
    fstream info("../"+mtx_name+".feature");
    string line;
    vector<float> weights;
    while (getline(info, line)) {
      stringstream ss(line);
      float weight;
      while (ss>>weight) {
        weights.push_back(weight);
      }
    }
    
    uniform_int_distribution<> rc(10, 17);
    uniform_int_distribution<> brc(0, 4);
    uniform_int_distribution<> nnz(100000, 10000000);
    discrete_distribution<> d(weights.begin(), weights.end());
    #pragma omp critical
    {
      cout << i << " " << mtx_name << endl;
    }   
 
    int rep=10;
    while (rep--) {
      num_row = 1 << rc(gen);
      num_col = 1 << rc(gen);
      num_nonzero = nnz(gen);
      double density = (double)num_nonzero/((double)num_row*num_col);
      if (density > 1.0) { continue; }

      // pick block size
      int bsizer = 1<<brc(gen);
      int bsizec = 1<<brc(gen);
      int num_block = num_nonzero/(bsizer*bsizec);
      // 1px becomes.. 
      int localsizer = num_row/256;
      int localsizec = num_col/256;
      if (localsizer < bsizer || localsizec < bsizec) continue;
      // inside real tensor (local idx)
      uniform_int_distribution<int> localrd(0, localsizer/bsizer-1);
      uniform_int_distribution<int> localcd(0, localsizec/bsizec-1);
      
      vector<pair<int,int>> coo;
      for (int n = 0; n<num_block; n++) {
        //picking global pixel r,c
        int glbidx = d(gen);
        int glbr = glbidx/256, glbc = glbidx%256;
        
        //picking local idx r,c
        int localr = localrd(gen);
        int localc = localcd(gen);

        for (int br=0; br<bsizer; br++) {
          for (int bc=0; bc<bsizec; bc++) {
            int r = glbr * localsizer + localr * bsizer + br;
            int c = glbc * localsizec + localc * bsizec + bc;
            coo.push_back({r,c});
          }
        }
      }
      sort( coo.begin(), coo.end() );
      coo.erase( unique( coo.begin(), coo.end() ), coo.end() );
      if (coo.size() < 100000) continue;
      if (coo.size() > 9000000) continue;
      coo_to_csr(coo, num_row, num_col, mtx_name+"_"+to_string(bsizer)+"x"+to_string(bsizec)+"_"+to_string(rep));
    }

   
    ///////////////
    // Augmentation
    ///////////////
    //vector<pair<int,int>> coo;
    //for (int i = 0; i<num_row; i++) {
    //  for (int j = A_pos[i]; j<A_pos[i+1]; j++) {
    //    coo.push_back({i, A_crd[j]});
    //  }
    //}
    //coo_to_csr(coo, num_row, num_col, mtx_name);
  }
  return 0;
}  
