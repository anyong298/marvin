//nvcc -std=c++11 compute_realtsdf_GPU.cu
#include <algorithm>
#include <vector>
#include <cmath>
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sys/time.h>
using namespace std;
void FatalError(const int lineNumber=0) {
  cerr << "FatalError";
  if (lineNumber!=0) cerr<<" at LINE "<<lineNumber;
  cerr << ". Program Terminated." << endl;
  cudaDeviceReset();
  exit(EXIT_FAILURE);
}

void checkCUDA(const int lineNumber, cudaError_t status) {
  if (status != cudaSuccess) {
    cerr << "CUDA failure at LINE " << lineNumber << ": " << status << endl;
    FatalError();
  }
}


unsigned long long get_timestamp_dss(){
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (unsigned long long)now.tv_sec * 1000000;
};


struct ImgObjInd{
  int ImageId;
  int ObjId; 
  int TarId;
  ImgObjInd(int i,int j,int k){
    ImageId = i;
    ObjId = j;
    TarId = k;
  }
};
struct RGBDpixel{
  uint8_t R;
  uint8_t G;
  uint8_t B;
  uint8_t D;
  uint8_t D_;
};
struct Box3D{
  unsigned int category;
  float base[9];
  float center[3];
  float coeff[3];
};

struct Boxdiff{
  float diff[6];
};


void __global__ compute_xyzkernel(float * XYZimage, RGBDpixel * RGBDimage, float * K, float * R){
            int ix = blockIdx.x;
            int iy = threadIdx.x;
            int height = blockDim.x;
            //
            //float depth = float(*((uint16_t*)(&(RGBDimage[iy + ix * height].D))))/1000.0;
            uint16_t D = (uint16_t)RGBDimage[iy + ix * height].D;
            uint16_t D_ = (uint16_t)RGBDimage[iy + ix * height].D_;
            D_ = D_<<8;
            float depth = float(D|D_)/1000.0;
            //printf("%d,%d,%f\n",RGBDimage[iy + ix * height].D,D_,depth);
            
            // project the depth point to 3d
            float tdx = (float(ix + 1) - K[2]) * depth / K[0];
            float tdz =  - (float(iy + 1) - K[5]) * depth / K[4];
            float tdy = depth;

            XYZimage[3 * (iy + ix * height) + 0] = R[0] * tdx + R[1] * tdy + R[2] * tdz;
            XYZimage[3 * (iy + ix * height) + 1] = R[3] * tdx + R[4] * tdy + R[5] * tdz;
            XYZimage[3 * (iy + ix * height) + 2] = R[6] * tdx + R[7] * tdy + R[8] * tdz;

}
void __global__ fillInBeIndexFull(unsigned int* beIndexFull, unsigned int* beIndex, unsigned int* beLinIdx, unsigned int len_beLinIdx){
     const int index = threadIdx.x + blockIdx.x * blockDim.x;
     if (index>=len_beLinIdx) {
        return;
     }
     else{
        beIndexFull[2*beLinIdx[index]+0] =  beIndex[2*index+0];
        beIndexFull[2*beLinIdx[index]+1] =  beIndex[2*index+1];
     }
}

class Scene3D{
public:
  // defined in .list file
  string filename;
  string seqname;
  float K[9];
  float R[9];
  unsigned int width;
  unsigned int height;
  unsigned int len_pcIndex;
  unsigned int len_beIndex;
  unsigned int len_beLinIdx;
  vector<Box3D> objects;

  bool  GPUdata;
  
  // defined in .data file
  unsigned int* grid_range;
  float* begin_range;
  float grid_delta;
  RGBDpixel* RGBDimage;
  unsigned int* beIndex;
  unsigned int* beLinIdx;
  unsigned int* pcIndex;
  float* XYZimage;
  float* K_GPU;
  float* R_GPU;

  

  Scene3D(): RGBDimage(NULL), beIndex(NULL), pcIndex(NULL), beLinIdx(NULL),XYZimage(NULL), grid_range(NULL), begin_range(NULL),K_GPU(NULL),R_GPU(NULL),GPUdata(false){};

  void compute_xyz() {
    XYZimage = new float[width*height*3];
    //printf("scene->K:%f,%f,%f\n%f,%f,%f\n%f,%f,%f\n",K[0],K[1],K[2],K[3],K[4],K[5],K[6],K[7],K[8]);
    for (int ix = 0; ix < width; ix++){
      for (int iy = 0; iy < height; iy++){
          float depth = float(*((uint16_t*)(&(RGBDimage[iy + ix * height].D))))/1000.0;
          //printf("%d,%f\n",RGBDimage[iy + ix * height].D,RGBDimage[iy + ix * height].D_,depth);
          // project the depth point to 3d
          float tdx = (float(ix + 1) - K[2]) * depth / K[0];
          float tdz =  - (float(iy + 1) - K[5]) * depth / K[4];
          float tdy = depth;

          XYZimage[3 * (iy + ix * height) + 0] = R[0] * tdx + R[1] * tdy + R[2] * tdz;
          XYZimage[3 * (iy + ix * height) + 1] = R[3] * tdx + R[4] * tdy + R[5] * tdz;
          XYZimage[3 * (iy + ix * height) + 2] = R[6] * tdx + R[7] * tdy + R[8] * tdz;
      }

    }
  };
  void compute_xyzGPU() {
      if (!GPUdata){
         cout<< "Data is not at GPU cannot compute_xyz at GPU"<<endl;
         FatalError(__LINE__);
      }
      if (XYZimage!=NULL){
         cout<< "XYZimage!=NULL"<<endl;
         FatalError(__LINE__);
      }
      checkCUDA(__LINE__, cudaMalloc(&XYZimage, sizeof(float)*width*height*3));
      compute_xyzkernel<<<width,height>>>(XYZimage,RGBDimage,K_GPU,R_GPU);
  }

  int load(){
    int filesize =0;
    if (RGBDimage==NULL||beIndex==NULL||pcIndex==NULL||XYZimage==NULL){
      //cout<< "loading image "<< filename<<endl;
      free();
      FILE* fp = fopen(filename.c_str(),"rb");
      if (fp==NULL) { cout<<"in load() :fail to open file: "<<filename<<endl; exit(EXIT_FAILURE); }
      grid_range = new unsigned int[3];
      filesize += fread((void*)(grid_range), sizeof(unsigned int), 3, fp);
      
      begin_range = new float[3];
      filesize += fread((void*)(begin_range), sizeof(float), 3, fp);
      filesize += fread((void*)(&grid_delta), sizeof(float), 1, fp);

      RGBDimage = new RGBDpixel[width*height];
      filesize += fread((void*)(RGBDimage), sizeof(RGBDpixel), width*height, fp);

      filesize +=  fread((void*)(&len_beIndex), sizeof(unsigned int), 1, fp);
      beIndex   = new unsigned int [len_beIndex];
      filesize += fread((void*)(beIndex), sizeof(unsigned int), len_beIndex, fp);

      filesize +=  fread((void*)(&len_beLinIdx), sizeof(unsigned int), 1, fp);
      beLinIdx  = new unsigned int [len_beLinIdx];
      filesize += fread((void*)(beLinIdx), sizeof(unsigned int), len_beLinIdx, fp);

      filesize += fread((void*)(&len_pcIndex), sizeof(unsigned int), 1, fp);
      pcIndex   = new unsigned int [len_pcIndex];
      filesize += fread((void*)(pcIndex), sizeof(unsigned int), len_pcIndex, fp);
      fclose(fp);

      GPUdata = false;
      
      //cout<<"size of RGBDpixel: "<<sizeof(RGBDpixel)/sizeof(unsigned char)<<endl;
      //cout<<"RGB:"<<int(RGBDimage[0].R)<<","<<(int)RGBDimage[0].G<<","<<(int)RGBDimage[0].B<<","<<(int)RGBDimage[0].D<<","<<endl;

      //cout<<len<<endl;
      //cout<<pcIndex[0]<<"-"<<pcIndex[1]<<endl;
      //cout<<len<<endl;
      //cout<<beIndex[0]<<"-"<<beIndex[1]<<endl;
    }
    return filesize;
  };
  void cpu2gpu(){
    if (!GPUdata){
       if (beIndex!=NULL){
           unsigned int* beIndexCPU = beIndex;
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMalloc(&beIndex, sizeof(unsigned int)*len_beIndex));
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMemcpy(beIndex, beIndexCPU,sizeof(unsigned int)*len_beIndex, cudaMemcpyHostToDevice));
           delete [] beIndexCPU;
       }
       else{
           cout << "beIndex is NULL"<<endl;
       }

       if (beLinIdx!=NULL){
           unsigned int* beLinIdxCPU = beLinIdx;
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMalloc(&beLinIdx, sizeof(unsigned int)*len_beLinIdx));
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMemcpy(beLinIdx, beLinIdxCPU,sizeof(unsigned int)*len_beLinIdx, cudaMemcpyHostToDevice));
           delete [] beLinIdxCPU;
       }
       else{
           cout << "beLinIdx is NULL"<<endl;
       }

       // make it to full matrix to skip searching 
       unsigned int * beIndexFull;
       unsigned int sz = 2*sizeof(unsigned int)*(grid_range[0]+1)*(grid_range[1]+1)*(grid_range[2]+1);
       checkCUDA(__LINE__, cudaMalloc(&beIndexFull, sz));
       checkCUDA(__LINE__, cudaMemset(beIndexFull, 0, sz));
       int THREADS_NUM = 1024;
       int BLOCK_NUM = int((len_beLinIdx + size_t(THREADS_NUM) - 1) / THREADS_NUM);
       fillInBeIndexFull<<<BLOCK_NUM,THREADS_NUM>>>(beIndexFull,beIndex,beLinIdx,len_beLinIdx);
       checkCUDA(__LINE__,cudaGetLastError());
       checkCUDA(__LINE__, cudaFree(beIndex));      beIndex = NULL;
       checkCUDA(__LINE__, cudaFree(beLinIdx));     beLinIdx = NULL;
       beIndex = beIndexFull;

       if (pcIndex!=NULL){
          unsigned int* pcIndexCPU = pcIndex;
          checkCUDA(__LINE__, cudaMalloc(&pcIndex, sizeof(unsigned int)*len_pcIndex));
          checkCUDA(__LINE__, cudaMemcpy(pcIndex, pcIndexCPU,sizeof(unsigned int)*len_pcIndex, cudaMemcpyHostToDevice));
          delete [] pcIndexCPU;
       }
       else{
           cout << "pcIndexCPU is NULL"<<endl;
       }
       

       if (RGBDimage!=NULL){
         RGBDpixel* RGBDimageCPU = RGBDimage;
         checkCUDA(__LINE__, cudaMalloc(&RGBDimage, sizeof(RGBDpixel)*width*height));
         checkCUDA(__LINE__, cudaMemcpy( RGBDimage, RGBDimageCPU, sizeof(RGBDpixel)*width*height, cudaMemcpyHostToDevice));
         delete [] RGBDimageCPU;
       }
       else{
           cout << "RGBDimage is NULL"<<endl;
       }
       /* 
       if (XYZimage!=NULL){ 
          float * XYZimageCPU = XYZimage;
          checkCUDA(__LINE__, cudaMalloc(&XYZimage, sizeof(float)*width*height*3));
          checkCUDA(__LINE__, cudaMemcpy(XYZimage, XYZimageCPU, sizeof(float)*width*height*3, cudaMemcpyHostToDevice));
          delete [] XYZimageCPU;
       }
       else{
          cout << "XYZimage is NULL"<<endl;
       }
       */

       if (grid_range!=NULL){ 
          unsigned int * grid_rangeCPU = grid_range;
          checkCUDA(__LINE__, cudaMalloc(&grid_range, sizeof(unsigned int)*3));
          checkCUDA(__LINE__, cudaMemcpy(grid_range, grid_rangeCPU, 3*sizeof(unsigned int), cudaMemcpyHostToDevice));
          delete [] grid_rangeCPU;
       }
       else{
          cout << "grid_range is NULL"<<endl;
       }

       if (begin_range!=NULL){ 
          float * begin_rangeCPU = begin_range;
          checkCUDA(__LINE__, cudaMalloc(&begin_range, sizeof(float)*3));
          checkCUDA(__LINE__, cudaMemcpy(begin_range, begin_rangeCPU, sizeof(float)*3, cudaMemcpyHostToDevice));
          delete [] begin_rangeCPU;
       }
       else{
          cout << "grid_range is NULL"<<endl;
       }


       checkCUDA(__LINE__, cudaMalloc(&K_GPU, sizeof(float)*9));
       checkCUDA(__LINE__, cudaMemcpy(K_GPU, (float*)K, sizeof(float)*9, cudaMemcpyHostToDevice));

      
       checkCUDA(__LINE__, cudaMalloc(&R_GPU, sizeof(float)*9));
       checkCUDA(__LINE__, cudaMemcpy(R_GPU, (float*)R, sizeof(float)*9, cudaMemcpyHostToDevice)); 

       GPUdata = true;

    }
  };

  void free(){
    if (GPUdata){
      //cout<< "free GPUdata"<<endl;
      if (RGBDimage   !=NULL) {checkCUDA(__LINE__, cudaFree(RGBDimage));    RGBDimage = NULL;}
      if (beIndex     !=NULL) {checkCUDA(__LINE__, cudaFree(beIndex));      beIndex = NULL;}
      if (beLinIdx    !=NULL) {checkCUDA(__LINE__, cudaFree(beLinIdx));     beLinIdx = NULL;}
      if (pcIndex     !=NULL) {checkCUDA(__LINE__, cudaFree(pcIndex));      pcIndex = NULL;}
      if (XYZimage    !=NULL) {checkCUDA(__LINE__, cudaFree(XYZimage));     XYZimage = NULL;}
      if (R_GPU       !=NULL) {checkCUDA(__LINE__, cudaFree(R_GPU));        R_GPU = NULL;}
      if (K_GPU       !=NULL) {checkCUDA(__LINE__, cudaFree(K_GPU));        K_GPU = NULL;}
      if (grid_range  !=NULL) {checkCUDA(__LINE__, cudaFree(grid_range));   grid_range = NULL;}
      if (begin_range !=NULL) {checkCUDA(__LINE__, cudaFree(begin_range));  begin_range = NULL;}
      GPUdata = false;
    }
    else{
      //cout<< "free CPUdata"<<endl;
      if (RGBDimage   !=NULL) {delete [] RGBDimage;    RGBDimage   = NULL;}
      if (beIndex     !=NULL) {delete [] beIndex;      beIndex     = NULL;}
      if (beLinIdx    !=NULL) {delete [] beLinIdx;     beLinIdx    = NULL;}
      if (pcIndex     !=NULL) {delete [] pcIndex;      pcIndex     = NULL;}
      if (XYZimage    !=NULL) {delete [] XYZimage;     XYZimage    = NULL;}
      if (grid_range  !=NULL) {delete [] grid_range;   grid_range  = NULL;}
      if (begin_range !=NULL) {delete [] begin_range;  begin_range = NULL;}
    }
  };
  ~Scene3D(){
    free();
  };
};



__global__ void compute_TSDFGPUbox(float* tsdf_data, float* R_data, float* K_data,  float* range, float grid_delta,  unsigned int *grid_range,
                                  RGBDpixel* RGBDimage,  unsigned int* star_end_indx_data ,unsigned int*  pc_lin_indx_data,float* XYZimage,
                                  const float* bb3d_data, int tsdf_size,int tsdf_size1,int tsdf_size2, int fdim, int im_w, int im_h, const int encode_type,const float scale)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;;
    int volume_size = tsdf_size * tsdf_size1 * tsdf_size2;
    //if (index > volume_size) return;
    float delta_x = 2 * bb3d_data[12] / float(tsdf_size);  
    float delta_y = 2 * bb3d_data[13] / float(tsdf_size1);  
    float delta_z = 2 * bb3d_data[14] / float(tsdf_size2);  
    float surface_thick = 0.1;
    const float MaxDis = surface_thick + 20;
    //printf("delta_x:%f,%f,%f\n",R_data[0],R_data[1],R_data[2]); 
    // caculate tsdf for this box
    /*
    float x = float(index % tsdf_size);
    float y = float((index / tsdf_size) % tsdf_size);   
    float z = float((index / tsdf_size / tsdf_size) % tsdf_size);
    */
    float x = float((index / (tsdf_size1*tsdf_size2))%tsdf_size) ;
    float y = float((index / tsdf_size2) % tsdf_size1);
    float z = float(index % tsdf_size2);

    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = 0;
    }

    // get grid world coordinate
    float temp_x = - bb3d_data[12] + (x + 0.5) * delta_x;
    float temp_y = - bb3d_data[13] + (y + 0.5) * delta_y;
    float temp_z = - bb3d_data[14] + (z + 0.5) * delta_z;

    x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
        + bb3d_data[9];
    y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
        + bb3d_data[10];
    z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
        + bb3d_data[11]; 

    // project to image plane decides the sign
    // rotate back and swap y, z and -y
    float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
    float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
    float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
    int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

    
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001){
        return;
    } 

    // find the most nearby point 
    float disTosurfaceMin = MaxDis;
    int idx_min = 0;
    int x_grid = floor((x-range[0])/grid_delta);
    int y_grid = floor((y-range[1])/grid_delta);
    int z_grid = floor((z-range[2])/grid_delta);
    //grid_range =  [w,d,h];  linearInd =x(i)*d*h+y(i)*h+z(i);
    //if (x_grid < 0 || x_grid >= grid_range[0] || y_grid < 0 || y_grid >= grid_range[1] || z_grid < 0 || z_grid >= grid_range[2]){
    if (x_grid < 0 || x_grid > grid_range[0] || y_grid < 0 || y_grid > grid_range[1] || z_grid < 0 || z_grid > grid_range[2]){
        return;
    }
    int linearInd =x_grid*grid_range[1]*grid_range[2]+y_grid*grid_range[2]+z_grid;      
    int search_region =1;
    if (star_end_indx_data[2*linearInd+0]>0){
        search_region =0;
    }  
    int find_close_point = -1;

    while(find_close_point<0&&search_region<3){
      for (int iix = max(0,x_grid-search_region); iix < min((int)grid_range[0],x_grid+search_region+1); iix++){
        for (int iiy = max(0,y_grid-search_region); iiy < min((int)grid_range[1],y_grid+search_region+1); iiy++){
          for (int iiz = max(0,z_grid-search_region); iiz < min((int)grid_range[2],z_grid+search_region+1); iiz++){
              unsigned int iilinearInd = iix*grid_range[1]*grid_range[2] + iiy*grid_range[2] + iiz;

              for (int pid = star_end_indx_data[2*iilinearInd+0]-1; pid < star_end_indx_data[2*iilinearInd+1]-1;pid++){
                 
                 //printf("%d-%d\n",star_end_indx_data[2*iilinearInd+0],star_end_indx_data[2*iilinearInd+1]);
                 unsigned int p_idx_lin = pc_lin_indx_data[pid];
                 float xp = XYZimage[3*p_idx_lin+0];
                 float yp = XYZimage[3*p_idx_lin+1];
                 float zp = XYZimage[3*p_idx_lin+2];
                 // distance
                 float xd = abs(x - xp);
                 float yd = abs(y - yp);
                 float zd = abs(z - zp);
                 if (xd < 2.0 * delta_x||yd < 2.0 * delta_x|| zd < 2.0 * delta_x){
                    float disTosurface = sqrt(xd * xd + yd * yd + zd * zd);
                    if (disTosurface < disTosurfaceMin){
                       disTosurfaceMin = disTosurface;
                       idx_min = p_idx_lin;
                       find_close_point = 1;
                       //printf("x:%f,%f,%f,xp,%f,%f,%f,xd%f,%f,%f,%f\n",x,y,z,xp,yp,zp,xd,yd,zd,disTosurfaceMin);
                       
                    }
                }
              } // for all points in this grid
            

          }
        }
      }
      search_region ++;
    }//while 
    
    float tsdf_x = MaxDis;
    float tsdf_y = MaxDis;
    float tsdf_z = MaxDis;


    float color_b =0;
    float color_g =0;
    float color_r =0;

    float xnear = 0;
    float ynear = 0;
    float znear = 0;
    if (find_close_point>0){
        
        xnear = XYZimage[3*idx_min+0];
        ynear = XYZimage[3*idx_min+1];
        znear = XYZimage[3*idx_min+2];
        if (encode_type == 100){
          tsdf_x = abs(x - xnear);
          tsdf_y = abs(y - ynear);
          tsdf_z = abs(z - znear);
        }

        color_b = float(RGBDimage[idx_min].B)/255.0;
        color_g = float(RGBDimage[idx_min].G)/255.0;
        color_r = float(RGBDimage[idx_min].R)/255.0;

        //printf("x:%f,tsdf_x:%f,%f,%f\n",disTosurfaceMin,tsdf_x,tsdf_y,tsdf_z);          
    }

    disTosurfaceMin = min(disTosurfaceMin/surface_thick,float(1.0));
    float ratio = 1.0 - disTosurfaceMin;
    float second_ratio =0;
    if (ratio > 0.5) {
       second_ratio = 1 - ratio;
    }
    else{
       second_ratio = ratio;
    }

    if (disTosurfaceMin > 0.999){
        tsdf_x = MaxDis;
        tsdf_y = MaxDis;
        tsdf_z = MaxDis;
    }

    if (encode_type == 101){ 
      tsdf_x = min(tsdf_x, surface_thick);
      tsdf_y = min(tsdf_y, surface_thick);
      tsdf_z = min(tsdf_z, surface_thick);
    }
    else{
      tsdf_x = min(tsdf_x, float(2.0 * delta_x));
      tsdf_y = min(tsdf_y, float(2.0 * delta_y));
      tsdf_z = min(tsdf_z, float(2.0 * delta_z));
    }


   

    float depth_project   = XYZimage[3*(ix * im_h + iy)+1];  
    if (zz > depth_project) {
      tsdf_x = - tsdf_x;
      tsdf_y = - tsdf_y;
      tsdf_z = - tsdf_z;
      disTosurfaceMin = - disTosurfaceMin;
      second_ratio = - second_ratio;
    }

    // encode_type 
    if (encode_type == 100||encode_type == 101){
      tsdf_data[index + 0 * volume_size] = tsdf_x;
      tsdf_data[index + 1 * volume_size] = tsdf_y;
      tsdf_data[index + 2 * volume_size] = tsdf_z;
    }
    if (encode_type == 102){
      tsdf_data[index + 0 * volume_size] = tsdf_x;
      tsdf_data[index + 1 * volume_size] = tsdf_y;
      tsdf_data[index + 2 * volume_size] = tsdf_z;
      tsdf_data[index + 3 * volume_size] = color_b;
      tsdf_data[index + 4 * volume_size] = color_g;
      tsdf_data[index + 5 * volume_size] = color_r;
    }
    // scale feature 
    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = scale*tsdf_data[index + i * volume_size];
    }

    //}// end for each index in each box
};

__global__ void compute_TSDFGPUbox_proj(float* tsdf_data, float* R_data, float* K_data, RGBDpixel* RGBDimage, float* XYZimage,
                                      const float* bb3d_data, int tsdf_size,int tsdf_size1,int tsdf_size2, int fdim, int im_w, int im_h, const int encode_type,const float scale)
{
  const int index = threadIdx.x + blockIdx.x * blockDim.x;;
    int volume_size = tsdf_size * tsdf_size1 * tsdf_size2;
    //if (index > volume_size) return;
    float delta_x = 2 * bb3d_data[12] / float(tsdf_size);  
    float delta_y = 2 * bb3d_data[13] / float(tsdf_size1);  
    float delta_z = 2 * bb3d_data[14] / float(tsdf_size2);  
    float surface_thick = 0.1;
    const float MaxDis = surface_thick + 20;

    float x = float((index / (tsdf_size1*tsdf_size2))%tsdf_size) ;
    float y = float((index / tsdf_size2) % tsdf_size1);
    float z = float(index % tsdf_size2);

    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = 0;
    }

    // get grid world coordinate
    float temp_x = - bb3d_data[12] + (x + 0.5) * delta_x;
    float temp_y = - bb3d_data[13] + (y + 0.5) * delta_y;
    float temp_z = - bb3d_data[14] + (z + 0.5) * delta_z;

    x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
        + bb3d_data[9];
    y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
        + bb3d_data[10];
    z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
        + bb3d_data[11]; 

    // project to image plane decides the sign
    // rotate back and swap y, z and -y
    float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
    float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
    float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
    int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

    printf("%d,%d,%d,%d\n",ix,iy,im_w,im_y);
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001){
        return;
    } 
    printf("pass\n");
    
    float x_project   = XYZimage[3*(ix * im_h + iy)+0];
    float y_project   = XYZimage[3*(ix * im_h + iy)+1];
    float z_project   = XYZimage[3*(ix * im_h + iy)+2]; 


    float tsdf_x = abs(x - x_project);
    float tsdf_y = abs(y - y_project);
    float tsdf_z = abs(z - z_project);

    float color_b = float(RGBDimage[(ix * im_h + iy)].B)/255.0;
    float color_g = float(RGBDimage[(ix * im_h + iy)].G)/255.0;
    float color_r = float(RGBDimage[(ix * im_h + iy)].R)/255.0;

    float disTosurfaceMin = sqrt(tsdf_x * tsdf_x + tsdf_y * tsdf_y + tsdf_z * tsdf_z);
    disTosurfaceMin = min(disTosurfaceMin/surface_thick,float(1.0));
    float ratio = 1.0 - disTosurfaceMin;
    float second_ratio =0;
    if (ratio > 0.5) {
       second_ratio = 1 - ratio;
    }
    else{
       second_ratio = ratio;
    }
    if (disTosurfaceMin > 0.999){
        tsdf_x = MaxDis;
        tsdf_y = MaxDis;
        tsdf_z = MaxDis;
    }

    tsdf_x = min(tsdf_x, float(2.0 * delta_x));
    tsdf_y = min(tsdf_y, float(2.0 * delta_y));
    tsdf_z = min(tsdf_z, float(2.0 * delta_z));

    if (zz > y_project) {
      tsdf_x = - tsdf_x;
      tsdf_y = - tsdf_y;
      tsdf_z = - tsdf_z;
      disTosurfaceMin = - disTosurfaceMin;
      second_ratio = - second_ratio;
    }

    // encode_type 
    if (encode_type == 0){
      tsdf_data[index + 0 * volume_size] = tsdf_x;
      tsdf_data[index + 1 * volume_size] = tsdf_y;
      tsdf_data[index + 2 * volume_size] = tsdf_z;
    }
    // scale feature 
    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = scale*tsdf_data[index + i * volume_size];
    }
}

void compute_TSDF_Space(Scene3D* scene , Box3D SpaceBox, float* tsdf_data_GPU, vector<int> grid_size, int encode_type, float scale){
   scene ->load();
   scene -> cpu2gpu();
   scene -> compute_xyzGPU();
   float* bb3d_data;
   checkCUDA(__LINE__, cudaMalloc(&bb3d_data,  sizeof(float)*15));
   checkCUDA(__LINE__, cudaMemcpy(bb3d_data    , SpaceBox.base, sizeof(float)*15, cudaMemcpyHostToDevice));
   unsigned int * grid_range = scene->grid_range;
   float* R_data = scene->R_GPU;
   float* K_data = scene->K_GPU;
   float* range  = scene->begin_range;
   RGBDpixel* RGBDimage = scene->RGBDimage;
   unsigned int* star_end_indx_data = scene->beIndex;
   unsigned int* pc_lin_indx_data = scene->pcIndex;
   float* XYZimage  = scene->XYZimage;

   int THREADS_NUM = 1024;
   int BLOCK_NUM = int((grid_size[1]*grid_size[2]*grid_size[3] + size_t(THREADS_NUM) - 1) / THREADS_NUM);

   compute_TSDFGPUbox<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_data_GPU, R_data, K_data, range, scene->grid_delta, grid_range, RGBDimage, 
                           star_end_indx_data, pc_lin_indx_data, XYZimage, bb3d_data, grid_size[1],grid_size[2],grid_size[3],grid_size[0], 
                           scene->width, scene->height, encode_type, scale);



   
   scene -> free();
   checkCUDA(__LINE__,cudaGetLastError());
   checkCUDA(__LINE__, cudaFree(bb3d_data));
      
}

void compute_TSDF (vector<Scene3D*> *chosen_scenes_ptr, vector<int> *chosen_box_id, float* datamem, vector<int> grid_size, int encode_type, float scale) {
    // for each scene 
    int totalcounter = 0;
    float tsdf_size = grid_size[1];
    if (grid_size[1]!=grid_size[2]||grid_size[1]!=grid_size[3]){
        cerr << "grid_size[1]!=grid_size[2]||grid_size[1]!=grid_size[3]" <<endl;
        exit(EXIT_FAILURE);
    }

    int numeltsdf = grid_size[0]*tsdf_size*tsdf_size*tsdf_size;
    int THREADS_NUM = 1024;
    int BLOCK_NUM = int((tsdf_size*tsdf_size*tsdf_size + size_t(THREADS_NUM) - 1) / THREADS_NUM);
    float* bb3d_data;

    //int tmpD; cudaGetDevice(&tmpD); cout<<"GPU at LINE "<<__LINE__<<" = "<<tmpD<<endl;
    //checkCUDA(__LINE__,cudaDeviceSynchronize());
    checkCUDA(__LINE__, cudaMalloc(&bb3d_data,  sizeof(float)*15));
    //checkCUDA(__LINE__,cudaDeviceSynchronize());
    
    unsigned long long transformtime =0;
    unsigned long long loadtime =0;
    unsigned long long copygputime =0;
    unsigned int sz = 0;
    Scene3D* scene_prev = NULL;
    for (int sceneId = 0;sceneId<(*chosen_scenes_ptr).size();sceneId++){
        // caculate in CPU mode
        //compute_TSDFCPUbox(tsdf_data,&((*chosen_scenes_ptr)[sceneId]),boxId,grid_size,encode_type,scale);
        // caculate in GPU mode
        
        unsigned long long  time0,time1,time2,time3,time4;
        Scene3D* scene = (*chosen_scenes_ptr)[sceneId];
        //int tmpD; cudaGetDevice(&tmpD); cout<<"GPU at LINE "<<__LINE__<<" = "<<tmpD<<endl;
        // perpare scene
        if (scene!=scene_prev){
            if (scene_prev!=NULL){
               //checkCUDA(__LINE__,cudaDeviceSynchronize());
               scene_prev -> free();
               //checkCUDA(__LINE__,cudaDeviceSynchronize());
            }

            //checkCUDA(__LINE__,cudaDeviceSynchronize());
            time0 = get_timestamp_dss();
            scene ->load();
            time1 = get_timestamp_dss();
            //checkCUDA(__LINE__,cudaDeviceSynchronize());
            //sz += 2*sizeof(unsigned int)*(scene->grid_range[0]+1)*(scene->grid_range[1]+1)*(scene->grid_range[2]+1);
            scene -> cpu2gpu();
            //checkCUDA(__LINE__,cudaDeviceSynchronize());
            scene -> compute_xyzGPU();
            //checkCUDA(__LINE__,cudaDeviceSynchronize());
            time2 = get_timestamp_dss();   
            //cout<< sceneId <<"ld "<< time1-time0 <<", cp "<< time2-time1 << endl;
        }
        
        int boxId = (*chosen_box_id)[sceneId];
        checkCUDA(__LINE__, cudaMemcpy(bb3d_data, scene->objects[boxId].base, sizeof(float)*15, cudaMemcpyHostToDevice));

        unsigned int * grid_range = scene->grid_range;
        float* R_data = scene->R_GPU;
        float* K_data = scene->K_GPU;
        float* range  = scene->begin_range;
        
        RGBDpixel* RGBDimage = scene->RGBDimage;
        unsigned int* star_end_indx_data = scene->beIndex;
        //unsigned int* beLinIdx_data = scene->beLinIdx;
        unsigned int* pc_lin_indx_data = scene->pcIndex;
        float* XYZimage  = scene->XYZimage;
        
        // output
        float * tsdf_data = &datamem[totalcounter*numeltsdf];

        time3 = get_timestamp_dss();   
        //checkCUDA(__LINE__,cudaDeviceSynchronize());
         if (encode_type > 99){
            compute_TSDFGPUbox<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_data, R_data, K_data, range, scene->grid_delta, grid_range, RGBDimage, 
                           star_end_indx_data, pc_lin_indx_data, XYZimage, bb3d_data, grid_size[1],grid_size[2],grid_size[3], grid_size[0], 
                           scene->width, scene->height, encode_type, scale);

        }
        else{
          //cout<<"compute_TSDFGPUbox_proj"<<endl;
          compute_TSDFGPUbox_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_data, R_data, K_data, RGBDimage, XYZimage,
                                                             bb3d_data, grid_size[1],grid_size[2],grid_size[3], grid_size[0], 
                                                             scene->width, scene->height, encode_type, scale);
        }
        
        //checkCUDA(__LINE__,cudaDeviceSynchronize());
        checkCUDA(__LINE__,cudaGetLastError());
        time4 = get_timestamp_dss();

        //

        ++totalcounter;

        scene_prev = scene;
        loadtime += time1-time0;
        copygputime += time2-time1;
        transformtime += time4-time3;
    }
    checkCUDA(__LINE__, cudaFree(bb3d_data));
    
    // free the loaded images
    for (int sceneId = 0;sceneId<(*chosen_scenes_ptr).size();sceneId++){
        (*chosen_scenes_ptr)[sceneId]->free();
    }
    
    
    //cout << "compute_TSDF: read disk " << loadtime/1000 << " ms, " << "copygputime " 
    //<< copygputime/1000 << "transform " << transformtime/1000 << " ms" <<endl;  
}

Box3D processbox (Box3D box,float context_pad,int tsdf_size){
     if (context_pad > 0){
        float context_scale = float(tsdf_size) / (float(tsdf_size) - 2*context_pad);
        box.coeff[0] = box.coeff[0] * context_scale;
        box.coeff[1] = box.coeff[1] * context_scale;
        box.coeff[2] = box.coeff[2] * context_scale;
     }
     // change the oreintation 
     if (box.base[1]<0){
        box.base[0] = -1*box.base[0];
        box.base[1] = -1*box.base[1];
        box.base[2] = -1*box.base[2];
     }
     if (box.base[4]<0){
        box.base[3] = -1*box.base[3];
        box.base[4] = -1*box.base[4];
        box.base[5] = -1*box.base[5];
     }

     if(box.base[1]<box.base[4]){
        // swap first two dim 
        float tmpbase[3];
        tmpbase[0] = box.base[0];
        tmpbase[1] = box.base[1];
        tmpbase[2] = box.base[2];

        box.base[0] = box.base[3];
        box.base[1] = box.base[4];
        box.base[2] = box.base[5];

        box.base[3] = tmpbase[0];
        box.base[4] = tmpbase[1];
        box.base[5] = tmpbase[2];
        float tmpcoeff =  box.coeff[0];
        box.coeff[0] = box.coeff[1];
        box.coeff[1] = tmpcoeff;
     }
     return box;
}

int main(){
    string file_list = "DSS/boxfile/boxes_NYU_trainfea_debug.list";
    //string data_root = "DSS/sunrgbd_dss_data/";
    string data_root =  "/home/shurans/deepDetectLocal/sunrgbd_dss_data/";
    vector<Scene3D*> scenes;

    //int count = 0;
    int object_count = 0;
    float scale =100;
    float context_pad =3;
    vector<int> grid_size {3,30,30,30};
    int encode_type = 0;

    cout<<"loading file "<<file_list<<"\n";
    FILE* fp = fopen(file_list.c_str(),"rb");
    if (fp==NULL) { cout<<"fail to open file: "<<file_list<<endl; exit(EXIT_FAILURE); }
    while (feof(fp)==0) {
      Scene3D* scene = new Scene3D();
      unsigned int len = 0;
      fread((void*)(&len), sizeof(unsigned int), 1, fp);    
      if (len==0) break;
      scene->filename.resize(len);
      if (len>0) fread((void*)(scene->filename.data()), sizeof(char), len, fp);
      scene->filename = data_root+scene->filename+".bin"; 
      fread((void*)(scene->R), sizeof(float), 9, fp);
      fread((void*)(scene->K), sizeof(float), 9, fp);
      fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);  
      fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 
      /*
      printf("scene->R:%f,%f,%f,%f,%f,%f\n",scene->R[0],scene->R[1],scene->R[2],scene->R[3],scene->R[4],scene->R[5]);
      printf("scene->K:%f,%f,%f,%f,%f,%f\n",scene->K[0],scene->K[1],scene->K[2],scene->K[3],scene->K[4],scene->K[5]);
      printf("scene->width:%d,%d\n",scene->width,scene->height);
      */

      fread((void*)(&len),    sizeof(unsigned int),   1, fp);
      scene->objects.resize(len);
      if (len>0){
          for (int i=0;i<len;++i){
              Box3D box;
              fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
              fread((void*)(box.base),        sizeof(float), 9, fp);
              fread((void*)(box.center),      sizeof(float), 3, fp);
              fread((void*)(box.coeff),       sizeof(float), 3, fp);
              //process box pad contex oreintation 
              box = processbox (box, context_pad, grid_size[1]);
              scene->objects[i]=box;

              object_count++;
              //num_categories = max(num_categories, box.category);
              /*
              printf("category:%d\n",box.category);
              printf("box.base:%f,%f,%f,%f,%f,%f\n",box.base[0],box.base[1],box.base[2],box.base[3],box.base[4],box.base[5]);
              printf("box.base:%f,%f,%f,%f,%f,%f\n",box.base[0],box.base[1],box.base[2],box.base[3],box.base[4],box.base[5]);
              printf("box.center:%f,%f,%f\n",box.center[0],box.center[1],box.center[2]);
              printf("box.coeff:%f,%f,%f\n",box.coeff[0],box.coeff[1],box.coeff[2]);
              */
          }
      }
      scenes.push_back(scene);

    }
    fclose(fp);

    vector<Scene3D*> chosen_scenes;
    vector<int> chosen_box_id;
    for (int i = 0;i<scenes.size();++i){
       for (int j =0; j < scenes[i]->objects.size();++j){
            chosen_scenes.push_back(scenes[i]);
            chosen_box_id.push_back(j);
       } 
    }

    
    cout<<"object_count:" <<object_count <<endl;
    float* dataCPUmem = new float[(object_count)*3*30*30*30];
    float* dataGPUmem;
    checkCUDA(__LINE__, cudaMalloc(&dataGPUmem, (object_count)*3*30*30*30*sizeof(float)));

    compute_TSDF(&chosen_scenes, &chosen_box_id, dataGPUmem,grid_size,encode_type,scale);
    checkCUDA(__LINE__, cudaMemcpy(dataCPUmem, dataGPUmem,(object_count)*3*30*30*30*sizeof(float), cudaMemcpyDeviceToHost) );
        

    string outputfile = "DSS/feature.bin";

    FILE * fid = fopen(outputfile.c_str(),"wb");
    fwrite(dataCPUmem,sizeof(float),(object_count)*3*30*30*30,fid);
    fclose(fid);
    return 1;
}
 