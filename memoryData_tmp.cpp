/*
std::vector<std::string> file_data;
std::vector<std::string> file_mean; mean can be shorter than file_data
no label
std::vector<ComputeT> scale;
std::vector<ComputeT> mean;
*/









class MemoryDataLayer : public DataLayer {
	Tensor<StorageT>* dataCPU;
	Tensor<StorageT>* labelCPU;
public:
	std::string file_data;
	std::string file_label;
	std::string file_mean;
	int batch_size;
	ComputeT scale;
	ComputeT mean;

	int numofitems(){
		return dataCPU->dim[0];
	};

	void init(){
		train_me = false;
		std::cout<<"MemoryDataLayer "<<name<<" loading data: "<<std::endl;
		dataCPU  = new Tensor<StorageT> (file_data,batch_size);
		dataCPU->print(veci(0));

		if (!file_mean.empty()){
			Tensor<StorageT>* meanCPU = new Tensor<StorageT>(file_mean);
			meanCPU->print(veci(0));

			if (meanCPU->numel() != dataCPU->sizeofitem()){
				std::cerr<<"mean tensor file size error: "<<std::endl;
				std::cerr<<"mean"; veciPrint(meanCPU->dim); std::cerr<<std::endl;
				std::cerr<<"data"; veciPrint(dataCPU->dim); std::cerr<<std::endl;
				FatalError(__LINE__);
			};

			StorageT* d  = dataCPU->CPUmem;
			StorageT* dE = dataCPU->CPUmem + dataCPU->numel();

			StorageT* m  = meanCPU->CPUmem;
			StorageT* mE = meanCPU->CPUmem + meanCPU->numel();

			while(d!=dE){
				*d = CPUCompute2StorageT( CPUStorage2ComputeT(*d) - CPUStorage2ComputeT(*m) );
				++m;
				if (m==mE) m = meanCPU->CPUmem;
				++d;
			}
			delete meanCPU;
		}

		//std::cout<<"scaling ... ";
		//tic();
		if (scale != 1){
			StorageT* dE = dataCPU->CPUmem + dataCPU->numel();
			for(StorageT* d  = dataCPU->CPUmem; d!=dE; ++d){
				*d = CPUCompute2StorageT( CPUStorage2ComputeT(*d) * scale );
			}
		}
		//toc();

		//std::cout<<"subtracting ... ";
		//tic();
		if (mean != 0){
			StorageT* dE = dataCPU->CPUmem + dataCPU->numel();
			for(StorageT* d  = dataCPU->CPUmem; d!=dE; ++d){
				*d = CPUCompute2StorageT( CPUStorage2ComputeT(*d) - mean );
			}
		}
		//toc();

		labelCPU = new Tensor<StorageT>(file_label,batch_size);
		labelCPU->print(veci(0));
		std::cout<<"    "; labelCPU->printRange();
		while (labelCPU->dim.size()<dataCPU->dim.size())
			labelCPU->dim.push_back(1);
		if (phase!=Testing) shuffle();
	};

	MemoryDataLayer(std::string name_, Phase phase_, std::string file_data_, std::string file_label_, int batch_size_): DataLayer(name_), batch_size(batch_size_), file_data(file_data_), file_label(file_label_), scale(1.0), mean(0.0){
		phase = phase_;
		init();
	};

	MemoryDataLayer(JSON* json){
		SetOrDie(json, name)
		SetValue(json, phase,		Training)
		SetOrDie(json, file_data 	)
		SetOrDie(json, file_label 	)
		SetValue(json, file_mean,	"")
		SetValue(json, batch_size,	64)
		SetValue(json, scale,		1.0)
		SetValue(json, mean,		0.0)
		init();
	};

	~MemoryDataLayer(){
		delete dataCPU;
		delete labelCPU;
	};

	size_t Malloc(Phase phase_){

		if (phase == Training && phase_==Testing) return 0;

		size_t memoryBytes = 0;

		std::cout<< (train_me? "* " : "  ");
		std::cout<<name<<std::endl;

		out[0]->need_diff = false;
		std::vector<int> data_dim = dataCPU->dim;
		data_dim[0] = batch_size;
		out[0]->receptive_field.resize(data_dim.size()-2);	fill_n(out[0]->receptive_field.begin(), data_dim.size()-2,1);
		out[0]->receptive_gap.resize(data_dim.size()-2);	fill_n(out[0]->receptive_gap.begin(),   data_dim.size()-2,1);
		out[0]->receptive_offset.resize(data_dim.size()-2);	fill_n(out[0]->receptive_offset.begin(),data_dim.size()-2,0);
		memoryBytes += out[0]->Malloc(data_dim);


		out[1]->need_diff = false;
		std::vector<int> label_dim= labelCPU->dim;
		label_dim[0] = batch_size;
		memoryBytes += out[1]->Malloc(label_dim);

		return memoryBytes;
	};

	void shuffle(){
		std::vector<size_t> v = randperm(dataCPU->numofitems(), rng);
		dataCPU->permute(v);
		labelCPU->permute(v);
	};

	void forward(Phase phase_){
		if (counter + batch_size >= dataCPU->numofitems() ){
			++epoch;
			if(phase!=Testing){
				shuffle();
				counter = 0;
			}
		}

		checkCUDA(__LINE__, cudaMemcpy(out[1]->dataGPU, labelCPU->CPUmem + (size_t(counter) * size_t(labelCPU->sizeofitem())), batch_size * labelCPU->sizeofitem() * sizeofStorageT, cudaMemcpyHostToDevice) );
		checkCUDA(__LINE__, cudaMemcpy(out[0]->dataGPU, dataCPU->CPUmem +  (size_t(counter) * size_t( dataCPU->sizeofitem())), batch_size * dataCPU->sizeofitem() * sizeofStorageT, cudaMemcpyHostToDevice) );

		counter+=batch_size;
		if (counter >= dataCPU->numofitems()) counter = 0;
	};
};





