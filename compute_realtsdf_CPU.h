
//g++ -std=c++11 compute_realtsdf_CPU.cpp
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

unsigned long long get_timestamp_dss(){
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (unsigned long long)now.tv_sec * 1000000;
};

struct ImgObjInd{
  int ImageId;
  int ObjId; 
  ImgObjInd(int i,int j){
    ImageId = i;
    ObjId = j;
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
class Scene3D{
public:
  // defined in .list file
  string filename;
  float* K;
  float* R;
  unsigned int width;
  unsigned int height;
  int len_pcIndex;
  int len_beIndex;
  vector<Box3D> objects;

  // defined in .data file
  unsigned int* grid_range;
  float* begin_range;
  float grid_delta;
  RGBDpixel* RGBDimage;
  unsigned int* beIndex;
  unsigned int* pcIndex;
  float* XYZimage;
  bool  GPUdata;
 

  Scene3D(): RGBDimage(NULL), beIndex(NULL), pcIndex(NULL), XYZimage(NULL), grid_range(NULL), begin_range(NULL), K(NULL), R(NULL){};

  void compute_xyz() {
    XYZimage = new float[width*height*3];
    //printf("scene.K:%f,%f,%f\n%f,%f,%f\n%f,%f,%f\n",K[0],K[1],K[2],K[3],K[4],K[5],K[6],K[7],K[8]);
    for (int ix = 0; ix < width; ix++){
      for (int iy = 0; iy < height; iy++){
          float depth = float(*((uint16_t*)(&(RGBDimage[iy + ix * height].D))))/1000.0;
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
  void load(){
    if (RGBDimage==NULL||beIndex==NULL||pcIndex==NULL){
      free();
      FILE* fp = fopen(filename.c_str(),"rb");
      if (fp==NULL) { cout<<"fail to open file: "<<filename<<endl; exit(EXIT_FAILURE); }
      grid_range = new unsigned int[3];
      fread((void*)(grid_range), sizeof(unsigned int), 3, fp);
      
      begin_range = new float[3];
      fread((void*)(begin_range), sizeof(float), 3, fp);

      fread((void*)(&grid_delta), sizeof(float), 1, fp);

      RGBDimage = new RGBDpixel[width*height];
      fread((void*)(RGBDimage), sizeof(RGBDpixel), width*height, fp);

      fread((void*)(&len_beIndex), sizeof(unsigned int), 1, fp);
      beIndex = new unsigned int [len_beIndex];
      fread((void*)(beIndex), sizeof(unsigned int), len_beIndex, fp);

      fread((void*)(&len_pcIndex), sizeof(unsigned int), 1, fp);
      pcIndex = new unsigned int [len_pcIndex];
      fread((void*)(pcIndex), sizeof(unsigned int), len_pcIndex, fp);
      fclose(fp);

      compute_xyz();
      GPUdata = false;
      //cout<<"size of RGBDpixel: "<<sizeof(RGBDpixel)/sizeof(unsigned char)<<endl;
      //cout<<"RGB:"<<int(RGBDimage[0].R)<<","<<(int)RGBDimage[0].G<<","<<(int)RGBDimage[0].B<<","<<(int)RGBDimage[0].D<<","<<endl;

      //cout<<len<<endl;
      //cout<<pcIndex[0]<<"-"<<pcIndex[1]<<endl;
      //cout<<len<<endl;
      //cout<<beIndex[0]<<"-"<<beIndex[1]<<endl;
    }
  };
  void cpu2gpu(){
    if (!GPUdata){
       if (beIndex!=NULL){
           unsigned int* beIndexCPU = beIndex;
           cudaMalloc(beIndex, sizeof(unsigned int)*len_beIndex);
           cudaMemcpy(beIndex, beIndexCPU,sizeof(unsigned int)*len_beIndex, cudaMemcpyHostToDevice));
           delete [] beIndexCPU;
       }
       else{
           cout << "beIndex is NULL"<<endl;
       }

       if (pcIndex!=NULL){
          unsigned int* pcIndexCPU = pcIndex;
          cudaMalloc(pcIndex, sizeof(unsigned int)*len_pcIndex);
          cudaMemcpy(pcIndex, pcIndexCPU,sizeof(unsigned int)*len_pcIndex, cudaMemcpyHostToDevice));
          delete [] pcIndexCPU;
       }
       else{
           cout << "pcIndexCPU is NULL"<<endl;
       }

       if (RGBDimage!=NULL){
         RGBDpixel* RGBDimageCPU = RGBDimage;
         cudaMalloc(RGBDimage, sizeof(RGBDpixel)*width*height);
         cudaMemcpy(RGBDimage, RGBDimageCPU, sizeof(RGBDpixel)*width*height, cudaMemcpyHostToDevice));
         delete [] RGBDimageCPU;
       }
       else{
           cout << "RGBDimage is NULL"<<endl;
       }
        
       if (XYZimage!=NULL){ 
          float * XYZimageCPU = XYZimage;
          cudaMalloc(XYZimage, sizeof(float)*width*height*3);
          cudaMemcpy(XYZimage, XYZimageCPU, sizeof(float)*width*height*3, cudaMemcpyHostToDevice));
          delete [] XYZimageCPU;
       }
       else{
          cout << "XYZimage is NULL"<<endl;
       }

       if (grid_range!=NULL){ 
          float * grid_rangeCPU = grid_range;
          cudaMalloc(grid_range, 3);
          cudaMemcpy(grid_range, grid_rangeCPU, 3, cudaMemcpyHostToDevice));
          delete [] XYZimageCPU;
       }
       else{
          cout << "grid_range is NULL"<<endl;
       }

       if (begin_range!=NULL){ 
          float * begin_rangeCPU = begin_range;
          cudaMalloc(begin_range, 3);
          cudaMemcpy(begin_range, begin_rangeCPU, 3, cudaMemcpyHostToDevice));
          delete [] begin_rangeCPU;
       }
       else{
          cout << "grid_range is NULL"<<endl;
       }


       if (K!=NULL){ 
          float * K_CPU = K;
          cudaMalloc(K, 9);
          cudaMemcpy(K, K_CPU, 9, cudaMemcpyHostToDevice));
          delete [] K_CPU;
       }
       else{
          cout << "K is NULL"<<endl;
       }

      
       if (R!=NULL){ 
          float* R_CPU = R;
          cudaMalloc(R,sizeof(float)*9);
          cudaMemcpy(R, R_CPU, sizeof(float)*9, cudaMemcpyHostToDevice)); 
       }
       else{
          cout << "R is NULL"<<endl;
       }

       GPUdata = true;

    }
    else{
        cout<< "It is already GPUdata skip copying"<<endl;
    } 
    
  };

  void free(){
    if (GPUdata){
      if (RGBDimage   !=NULL) checkCUDA(__LINE__, cudaFree(RGBDimage));
      if (beIndex     !=NULL) checkCUDA(__LINE__, cudaFree(beIndex));
      if (pcIndex     !=NULL) checkCUDA(__LINE__, cudaFree(pcIndex));
      if (XYZimage    !=NULL) checkCUDA(__LINE__, cudaFree(XYZimage));
      if (K           !=NULL) checkCUDA(__LINE__, cudaFree(K));
      if (R           !=NULL) checkCUDA(__LINE__, cudaFree(R));
      if (grid_range  !=NULL) checkCUDA(__LINE__, cudaFree(grid_range));
      if (begin_range !=NULL) checkCUDA(__LINE__, cudaFree(begin_range));
    }
    else{
      if (RGBDimage   !=NULL) delete [] RGBDimage;
      if (beIndex     !=NULL) delete [] beIndex;
      if (pcIndex     !=NULL) delete [] pcIndex;
      if (XYZimage    !=NULL) delete [] XYZimage;
      if (K           !=NULL) delete [] K;
      if (R           !=NULL) delete [] R;
      if (grid_range  !=NULL) delete [] grid_range;
      if (begin_range !=NULL) delete [] begin_range;
    }
  };
  ~Scene3D(){
    free();
  };
};


void compute_TSDFCPUbox(float* tsdf_data, const Scene3D* scene, int boxId, 
                        const vector<int> tsdf_grid_size, const int encode_type,const float scale)
{
    int tsdf_size = tsdf_grid_size[1];
    int fdim = tsdf_grid_size[0];
    

    // pull the data from scene
    float* bb3d_data = (float*)(scene->objects[boxId].base);
                       
    unsigned int im_w = scene->width;
    unsigned int im_h = scene->height;
    unsigned int * grid_range = (unsigned int *)scene->grid_range;
    float* R_data = (float*)scene->R;
    float* K_data = (float*)scene->K;
    float* range = (float*)scene->begin_range;
    
    RGBDpixel* RGBDimage = scene->RGBDimage;
    unsigned int* star_end_indx_data = scene->beIndex;
    unsigned int* pc_lin_indx_data = scene->pcIndex;
    float* XYZimage  = scene->XYZimage;
    
    //
    int volume_size = tsdf_size * tsdf_size * tsdf_size;
    float delta_x = 2 * bb3d_data[12] / float(tsdf_size);  
    float delta_y = 2 * bb3d_data[13] / float(tsdf_size);  
    float delta_z = 2 * bb3d_data[14] / float(tsdf_size);  
    float surface_thick = 0.1;
    //printf("delta_x:%f,%f,%f\n",delta_x,delta_y,delta_z); 
    // caculate tsdf for this box
    for (int index = 0; index < volume_size; index++){
      float x = float(index % tsdf_size);
      float y = float((index / tsdf_size) % tsdf_size);   
      float z = float((index / tsdf_size / tsdf_size) % tsdf_size);
       
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
      float xx = R_data[0] * x + R_data[3] * y + R_data[6] * z;
      float zz = R_data[1] * x + R_data[4] * y + R_data[7] * z;
      float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
      int ix = std::round(xx * K_data[0] / zz + K_data[2]) - 1;
      int iy = std::round(yy * K_data[4] / zz + K_data[5]) - 1;
      if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001){
          continue;
      }
     
      // find the most nearby point 
      float disTosurfaceMin = surface_thick+100.0;
      int idx_min = 0;
      int x_grid = floor((x-range[0])/scene->grid_delta);
      int y_grid = floor((y-range[1])/scene->grid_delta);
      int z_grid = floor((z-range[2])/scene->grid_delta);
      //grid_range =  [w,d,h];  linearInd =x(i)*d*h+y(i)*h+z(i);
      if (x_grid < 0 || x_grid >= grid_range[0] || y_grid < 0 || y_grid >= grid_range[1] || z_grid < 0 || z_grid >= grid_range[2]){
          continue;
      }
      int linearInd =x_grid*grid_range[1]*grid_range[2]+y_grid*grid_range[2]+z_grid;      
      int search_regoin =1;
      if (star_end_indx_data[2*linearInd+0]>0){
          search_regoin =0;
      }  
      int find_close_point = -1;

      while(find_close_point<0&&search_regoin<3){
        for (int iix = max(0,x_grid-search_regoin); iix < min((int)grid_range[0],x_grid+search_regoin+1); iix++){
          for (int iiy = max(0,y_grid-search_regoin); iiy < min((int)grid_range[1],y_grid+search_regoin+1); iiy++){
            for (int iiz = max(0,z_grid-search_regoin); iiz < min((int)grid_range[2],z_grid+search_regoin+1); iiz++){
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
                }
              

            }
          }
        }
        search_regoin ++;
      }//while 

      float tsdf_x = 2.0 * delta_x;
      float tsdf_y = 2.0 * delta_y;
      float tsdf_z = 2.0 * delta_z;


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

      tsdf_x = min(tsdf_x, float(2.0 * delta_x));
      tsdf_y = min(tsdf_y, float(2.0 * delta_y));
      tsdf_z = min(tsdf_z, float(2.0 * delta_z));


      if (disTosurfaceMin > 0.999){
          tsdf_x = 2.0 * delta_x;
          tsdf_y = 2.0 * delta_y;
          tsdf_z = 2.0 * delta_z;
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
      if (encode_type == 100){
        tsdf_data[index + 0 * volume_size] = tsdf_x;
        tsdf_data[index + 1 * volume_size] = tsdf_y;
        tsdf_data[index + 2 * volume_size] = tsdf_z;
      }
      // scale feature 
      for (int i =0;i<fdim;i++){
          tsdf_data[index + i * volume_size] = scale*tsdf_data[index + i * volume_size];
      }

    }// end for each index in each box


}



void compute_TSDFGPUbox(float* tsdf_data, float* R_data, float* K_data,  float* range, RGBDpixel* RGBDimage, 
                        unsigned int* star_end_indx_data, float* XYZimage,
                        float* bb3d_data, int tsdf_size, int fdim, int im_w, int im_h,
                        const int encode_type,const float scale)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;;
    int volume_size = tsdf_size * tsdf_size * tsdf_size;
    if (index > volume_size) return;
    float delta_x = 2 * bb3d_data[12] / float(tsdf_size);  
    float delta_y = 2 * bb3d_data[13] / float(tsdf_size);  
    float delta_z = 2 * bb3d_data[14] / float(tsdf_size);  
    float surface_thick = 0.1;
    //printf("delta_x:%f,%f,%f\n",delta_x,delta_y,delta_z); 
    // caculate tsdf for this box
    //for (int index = 0; index < volume_size; index++){
    float x = float(index % tsdf_size);
    float y = float((index / tsdf_size) % tsdf_size);   
    float z = float((index / tsdf_size / tsdf_size) % tsdf_size);
     
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
    float xx = R_data[0] * x + R_data[3] * y + R_data[6] * z;
    float zz = R_data[1] * x + R_data[4] * y + R_data[7] * z;
    float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = std::round(xx * K_data[0] / zz + K_data[2]) - 1;
    int iy = std::round(yy * K_data[4] / zz + K_data[5]) - 1;
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001){
        continue;
    }
   
    // find the most nearby point 
    float disTosurfaceMin = surface_thick+100.0;
    int idx_min = 0;
    int x_grid = floor((x-range[0])/scene->grid_delta);
    int y_grid = floor((y-range[1])/scene->grid_delta);
    int z_grid = floor((z-range[2])/scene->grid_delta);
    //grid_range =  [w,d,h];  linearInd =x(i)*d*h+y(i)*h+z(i);
    if (x_grid < 0 || x_grid >= grid_range[0] || y_grid < 0 || y_grid >= grid_range[1] || z_grid < 0 || z_grid >= grid_range[2]){
        continue;
    }
    int linearInd =x_grid*grid_range[1]*grid_range[2]+y_grid*grid_range[2]+z_grid;      
    int search_regoin =1;
    if (star_end_indx_data[2*linearInd+0]>0){
        search_regoin =0;
    }  
    int find_close_point = -1;

    while(find_close_point<0&&search_regoin<3){
      for (int iix = max(0,x_grid-search_regoin); iix < min((int)grid_range[0],x_grid+search_regoin+1); iix++){
        for (int iiy = max(0,y_grid-search_regoin); iiy < min((int)grid_range[1],y_grid+search_regoin+1); iiy++){
          for (int iiz = max(0,z_grid-search_regoin); iiz < min((int)grid_range[2],z_grid+search_regoin+1); iiz++){
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
              }
            

          }
        }
      }
      search_regoin ++;
    }//while 

    float tsdf_x = 2.0 * delta_x;
    float tsdf_y = 2.0 * delta_y;
    float tsdf_z = 2.0 * delta_z;


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

    tsdf_x = min(tsdf_x, float(2.0 * delta_x));
    tsdf_y = min(tsdf_y, float(2.0 * delta_y));
    tsdf_z = min(tsdf_z, float(2.0 * delta_z));


    if (disTosurfaceMin > 0.999){
        tsdf_x = 2.0 * delta_x;
        tsdf_y = 2.0 * delta_y;
        tsdf_z = 2.0 * delta_z;
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
    if (encode_type == 100){
      tsdf_data[index + 0 * volume_size] = tsdf_x;
      tsdf_data[index + 1 * volume_size] = tsdf_y;
      tsdf_data[index + 2 * volume_size] = tsdf_z;
    }
    // scale feature 
    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = scale*tsdf_data[index + i * volume_size];
    }

    //}// end for each index in each box


}

void compute_TSDF(vector<Scene3D> * chosen_scenes_ptr, float* datamem, vector<int> grid_size, int encode_type, float scale) {
    // for each scene 
    int totalcounter = 0;
    tsdf_size = grid_size[1];
    if (grid_size[1]!=grid_size[2]||grid_size[1]!=grid_size[3]){
        cerr << "grid_size[1]!=grid_size[2]||grid_size[1]!=grid_size[3]" <<endl;
        exit(EXIT_FAILURE);
    }

    int numeltsdf = grid_size[0]*grid_size[1]*grid_size[2]*grid_size[3];
    int THREADS_NUM = 512;
    int BLOCK_NUM = int((N + size_t(THREADS_NUM) - 1) / CUDA_NUM_THREADS);
    
    unsigned long long loadtime = 0;
    //unsigned long long transformtime =0;
    for (int sceneId = 0;sceneId<(*chosen_scenes_ptr).size();sceneId++){
        unsigned long long  time1 = get_timestamp_dss();
        (*chosen_scenes_ptr)[sceneId].load();
        //unsigned long long  time2 = get_timestamp_dss();
        //cout<<"loadfile:"<<(*chosen_scenes_ptr)[sceneId].filename<<endl;
        for (int boxId = 0;boxId<(*chosen_scenes_ptr)[sceneId].objects.size();boxId++){
            float * tsdf_data = &datamem[totalcounter*numeltsdf];
            // caculate in CPU mode
            // compute_TSDFCPUbox(tsdf_data,&((*chosen_scenes_ptr)[sceneId]),boxId,grid_size,encode_type,scale);
            // caculate in GPU mode
            Scene3D* scene = &((*chosen_scenes_ptr)[sceneId];
            scene -> cpu2gpu();
            float* bb3d_data;
            cudaMalloc(bb3d_data, 15);
            cudaMemcpy(bb3d_data, scene->objects[boxId].base, sizeof(float)*15, cudaMemcpyHostToDevice));
            
            unsigned int * grid_range = (unsigned int *)scene->grid_range;
            float* R_data = scene->R;
            float* K_data = scene->K;
            float* range  = scene->begin_range;
            
            RGBDpixel* RGBDimage = scene->RGBDimage;
            unsigned int* star_end_indx_data = scene->beIndex;
            unsigned int* pc_lin_indx_data = scene->pcIndex;
            float* XYZimage  = scene->XYZimage;
           

            compute_TSDFGPUbox<<BLOCK_NUM,THREADS_NUM>>(tsdf_data, R_data, K_data, range, RGBDimage, 
                               star_end_indx_data, XYZimage, bb3d_data, grid_size[1], grid_size[0], 
                               scene->width, scene-> height, encode_type, scale)
            checkCUDA(__LINE__,cudaGetLastError());

            checkCUDA(__LINE__, cudaFree(bb3d_data));
            ++totalcounter;
            
        }
        //unsigned long long  time3 = get_timestamp_dss();

        //loadtime += time2-time1;
        //transformtime += time3-time2;
    }
    
    //cout << "compute_TSDF: readdisk " << loadtime/1000 << " ms" << "transformtime " << transformtime/1000 << " ms" <<endl;  
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
     }
     return box;
}

void test(){
    string file_list = "DSS/boxfile//boxes_NYU_trainfea_debug.list";
    string data_root = "DSS/sunrgbd_marvin_data/";
    vector<Scene3D> scenes;
    //int count = 0;
    int object_count = 0;
    float scale =100;
    float context_pad =3;
    vector<int> grid_size {3,30,30,30};
    int encode_type =100;

    cout<<"loading file "<<file_list<<"\n";
    FILE* fp = fopen(file_list.c_str(),"rb");
    if (fp==NULL) { cout<<"fail to open file: "<<file_list<<endl; exit(EXIT_FAILURE); }
    while (feof(fp)==0) {
      Scene3D scene;
      unsigned int len = 0;
      fread((void*)(&len), sizeof(unsigned int), 1, fp);    
      if (len==0) break;
      scene.filename.resize(len);
      if (len>0) fread((void*)(scene.filename.data()), sizeof(char), len, fp);
      scene.filename = data_root+scene.filename+".bin"; 
      scene.R = new float[9];
      scene.K = new float[9];
      fread((void*)(scene.R), sizeof(float), 9, fp);
      fread((void*)(scene.K), sizeof(float), 9, fp);
      fread((void*)(&scene.height), sizeof(unsigned int), 1, fp);  
      fread((void*)(&scene.width), sizeof(unsigned int), 1, fp); 
      /*
      printf("scene.R:%f,%f,%f,%f,%f,%f\n",scene.R[0],scene.R[1],scene.R[2],scene.R[3],scene.R[4],scene.R[5]);
      printf("scene.K:%f,%f,%f,%f,%f,%f\n",scene.K[0],scene.K[1],scene.K[2],scene.K[3],scene.K[4],scene.K[5]);
      printf("scene.width:%d,%d\n",scene.width,scene.height);
      */

      fread((void*)(&len),    sizeof(unsigned int),   1, fp);
      scene.objects.resize(len);
      if (len>0){
          for (int i=0;i<len;++i){
              Box3D box;
              fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
              fread((void*)(box.base),        sizeof(float), 9, fp);
              fread((void*)(box.center),      sizeof(float), 3, fp);
              fread((void*)(box.coeff),       sizeof(float), 3, fp);
              //process box pad contex oreintation 
              box = processbox (box, context_pad, grid_size[1]);
              scene.objects[i]=box;
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

    
    cout<<"object_count:" <<object_count <<endl;
    float* dataCPUmem = new float[(object_count)*3*30*30*30];
    
    compute_TSDF(&scenes,dataCPUmem,grid_size,encode_type,scale);
    string outputfile = "DSS/feature.bin";

    FILE * fid = fopen(outputfile.c_str(),"wb");
    fwrite(dataCPUmem,sizeof(float),(object_count)*3*30*30*30,fid);
    fclose(fid);
}
  
 