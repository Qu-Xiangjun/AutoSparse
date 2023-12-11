#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <parallel/algorithm>
#include "time.hpp"
using namespace std;

typedef enum
{
	COMPRESSED,
	UNCOMPRESSED
} taco_mode_t;

/**
 * 存储一个tensor的元数据与值，按照COO格式
 */
typedef struct 
{
	// 张量格式元信息
	int32_t order;			 // tensor order (number of modes)
	int32_t *dimensions;	 // tensor dimensions， 各个维度的轴长
	int32_t csize;			 // component size
	int32_t *mode_ordering;	 // mode storage ordering 各个模式的顺序 暂时未用
	taco_mode_t *mode_types; // mode storage types

	// 稀疏张量的坐标信息和非零值
	uint8_t ***indices;		 // tensor index data (per mode) 一个三维数组，存储各个`压缩维度`下csr格式的信息
	uint8_t *vals;			 // tensor values 存储所有非零值的数组

	int32_t vals_size;		 // values array size    NOTE: 没看见使用
} taco_tensor_t;

struct FormatInfo 
{
	string var;
	int dimension;		// 该维度的轴长，如i维度长为1024
	taco_mode_t mode;	// 该维度格式的属性，压缩还是不压缩 
	// Lowest Rank first 的存储在 临时coo格式的压缩坐标表示buffer中
	int startbit;		// 当前格式轴坐标存储在bit数组中起始位置 
	int lenbit;			// 当前格式对应轴坐标 存储需要的bit数量
};


/**
 * 
 */
class FormatScheduler
{
protected:
	vector<FormatInfo> TensorFormat;		// 每个维度的格式信息
	vector<FormatInfo> TensorFormat_init;	// 最初init版本的TensorFormat，很干净，没有任何调度格式的应用
	vector<pair<uint64_t, float>> coo;		// 外面传入未被pack版本的coo格式存储
	vector<vector<int>> T_pos, T_crd;		// pack 内部存储按照csr格式存 的 元数据
	vector<float> T_vals;					// pack 内部存储按照csr格式存 的 非零值，注意这是有序的从低地址到高地址
	taco_tensor_t *T = NULL; // 该张量的元数据， 以下的重要函数都不使用，应该是 用于 get_taco_tensor 外面查看
	bool is_dense;	   // All ranks are Uncompressed
	bool is_dense_fix; // All ranks are Uncompressed

protected:
	inline int get_digit2(int x) { return (int)ceil(log2(x)); }
	inline int extract(uint64_t coords, int start, int len)
	{
		/**
		 * 将压缩后的值解压，对应为坐标的行号i
		 */
		uint64_t result = (coords >> start);
		result = result & ((1 << len) - 1);
		return (int)result;
	}
	inline uint64_t extractuppercoords(uint64_t coords, int start)
	{
		return coords >> start;
	}

	void init_taco_tensor_t()
	{
		int num_rank = TensorFormat.size();
		T = (taco_tensor_t *)new taco_tensor_t[1];
		T->order = num_rank;	// 维度大小
		T->dimensions = (int32_t *)new int32_t[num_rank];	// 存储shape，即各个维度的大小
		T->mode_types = (taco_mode_t *)new taco_mode_t[num_rank];	// 存储各个维度的稀疏模式 压缩 or 未压缩
		// 这里是分配一个三维数组指针, new 的一个 数据类型 为uint8_t ** 大小是num_rank的数组
		// 即 T->indices 是一个数组，数组每个元素是 uint8_t **类型，即一个uint8_t的二维数组。
		T->indices = (uint8_t ***)new uint8_t **[num_rank]; // 用于存储压缩csr格式。
		for (int32_t rank = 0; rank < num_rank; rank++)
		{
			T->dimensions[rank] = TensorFormat[rank].dimension;
			T->mode_types[rank] = TensorFormat[rank].mode;
			T->indices[rank] = (uint8_t **)new uint8_t *[2]; // new 一个 数据类型为uint8_t *的，大小为2的数组
			switch (TensorFormat[rank].mode)
			{
			case COMPRESSED: // 注意indices只需要存储压缩格式下的
				// 注意下面.data 是从vector中取出来数据，所以不需要new，vector会自动释放区域
				// 同时注意操作T也相当于是在操作T_pos T_crd T_vals
				T->indices[rank][0] = (uint8_t *)(T_pos[rank].data()); // 存储各个维度csr格式的pos数组
				T->indices[rank][1] = (uint8_t *)(T_crd[rank].data()); // 存储各个维度csr格式的crd数组
				break;
			}
		}
		T->vals = (uint8_t *)(T_vals.data());
		T->vals_size = T_vals.size();
	}

	void destroy_taco_tensor_t()
	{
		// cout << "Destroy " << flush;
		// cout << T->order << flush;
		int num_rank = T->order;
		delete[] T->dimensions;
		delete[] T->mode_types;
		for (int rank = 0; rank < num_rank; rank++)
		{
			delete[] T->indices[rank]; // 只需要手动释放前两维数组，最后一维是vector的数据，会自动释放
		}
		delete[] T->indices;
		delete[] T;
	}

public:
	vector<FormatInfo> &get_format() { return TensorFormat; } 	// 获得各个维度的格式信息
	vector<float> &get_vals() { return T_vals; }				// 获得所有的非零值
	vector<vector<int>> &get_pos() { return T_pos; }				
	vector<vector<int>> &get_crd() { return T_crd; }				
	void init() { TensorFormat = TensorFormat_init; }			// 初始化，即将格式等于最早初始化时候的格式
	void set_coo(vector<pair<uint64_t, float>> &coo_) { coo = coo_; }	// 设置coo，即pack前的临时存储buffer
	
	/**
	 * 析构函数，主要是针对delete掉张量的元数据
	 */
	~FormatScheduler() 
	{
		if (T != NULL)
			destroy_taco_tensor_t();
	}

	/**
	 * 遍历每个维度，查看其轴名称是否存在
	 */
	bool is_var_exist(string var) // 用于检测
	{
		for (int rank = 0; rank < TensorFormat.size(); rank++)
		{
			if (TensorFormat[rank].var == var)
			{
				return true;
			}
		}
		return false;
	}

	/**
	 * 按照表格形式输出各个轴的 formatInfo 结构体中存储的信息
	 */
	string print_format()
	{
		stringstream ss;
		ss << "Rank\tVar\tDim\tMode\tStartb\tLenb" << endl;
		int rank = 0;
		for (FormatInfo &format : TensorFormat)
		{
			ss << rank++ << "\t";						// 维度	
			ss << format.var << "\t";					// 轴名称
			ss << format.dimension << "\t";				// 该维度的shape大小
			ss << (format.mode ? "U" : "C") << "\t";	// 轴模式属性
			ss << format.startbit << "\t";				// 该维度坐标在coo压缩的坐标格式中的 起始位置
			ss << format.lenbit;						// 该维度坐标在coo压缩的坐标格式中的 长度
			ss << endl;
		}
		return ss.str();
	}


	/**
	 * 两种构造函数，针对稠密和稀疏张量的 TensorFormat数据结构 初始化
	 */
	// Rank order in TensorFormat starts from the highest rank.
	// e.g.) A[i,k](U,C) => TensorFormat = [{"i",U}, {"k",C}]
	//       Then COO : {"i", "k"}
	FormatScheduler(vector<pair<uint64_t, float>> &coo, vector<FormatInfo> &init) : coo(coo), TensorFormat(init)
	{
		int num_rank = TensorFormat.size();
		TensorFormat[num_rank - 1].startbit = 0; // Lowest Rank first
		TensorFormat[num_rank - 1].lenbit = get_digit2(TensorFormat[num_rank - 1].dimension);
		for (int rank = num_rank - 2; rank >= 0; rank--)
		{
			TensorFormat[rank].startbit = TensorFormat[rank + 1].startbit + TensorFormat[rank + 1].lenbit;
			TensorFormat[rank].lenbit = get_digit2(TensorFormat[rank].dimension);
		}
		TensorFormat_init = TensorFormat;
		is_dense_fix = false;
	}

	FormatScheduler(vector<float> &dense, vector<FormatInfo> &init) : TensorFormat(init), TensorFormat_init(init)
	{
		is_dense_fix = true;
		is_dense = true;
		T_vals = dense;
	}


	/**
	 * Split 调度，拆分两个轴一里一外，inner的初始化插入TensorFormat，更改老轴为outter
	 */
	// |<-----i----->|
	// |<-i1->|<-i0->|
	FormatScheduler &split(string var, string outer_var, string inner_var, int split_size)
	{
		for (int rank = 0; rank < TensorFormat.size(); rank++)
		{	// 遍历所有维度找到这个轴
			if (TensorFormat[rank].var == var) 
			{
				// 新建立一个inner的轴
				FormatInfo inner_rank;
				inner_rank.var = inner_var; // 名称
				inner_rank.dimension = min(split_size, TensorFormat[rank].dimension); // inner轴的长度
				inner_rank.mode = TensorFormat[rank].mode;	// 模式保持不变
				inner_rank.startbit = TensorFormat[rank].startbit;	  // in bit
				inner_rank.lenbit = get_digit2(inner_rank.dimension); // in bit

				TensorFormat[rank].var = outer_var;
				TensorFormat[rank].dimension = (TensorFormat[rank].dimension + inner_rank.dimension - 1) / inner_rank.dimension;
				TensorFormat[rank].mode = TensorFormat[rank].mode;
				TensorFormat[rank].startbit = TensorFormat[rank].startbit + get_digit2(split_size);
				TensorFormat[rank].lenbit = get_digit2(TensorFormat[rank].dimension);

				TensorFormat.insert(TensorFormat.begin() + rank + 1, inner_rank); // 因为rank从0开始，所以+1
				break;
			}
		}
		return *this; // 返回类
	}
 
	/**
	 * 轴顺序更换，必须输入所有轴重排后的顺序
	 * 遍历所有轴，按照新的顺序 将旧的轴格式信息加入新的轴格式数组
	 * 但是： formatInfo中的startbit lenbit并未改变顺序， 
	 * 			同时，对应临时存储区coo数据结构中位置并没变。
	 * 			因此pack的时候读取coo的每个点的坐标顺序会改变，然后再被pack为实际的压缩格式存储，
	 * 			此时pack后存储就紧凑了，如果当前是compress层，就依赖上一个层的非零元素数量+1作为pos数组的长度
	 */
	FormatScheduler &reorder(vector<string> reordered_vars)
	{
		if (reordered_vars.size() != TensorFormat.size()) // 要求必须输入所有的轴信息，否则不做更换
			return *this;
		vector<FormatInfo> ReorderedTensorFormat;
		for (string &var : reordered_vars)
		{
			for (int rank = 0; rank < TensorFormat.size(); rank++)
			{
				if (TensorFormat[rank].var == var)
				{
					ReorderedTensorFormat.push_back(TensorFormat[rank]);
				}
			}
		}
		TensorFormat = ReorderedTensorFormat;
		return *this;
	}


	/**
	 * 改变某个轴的格式属性， 注意操作是在pack之前改变，
	 * 然后 pack 会通过coo格式去生成格式属性确定后的压缩存储buffer。
	 */
	FormatScheduler &mode(string var, taco_mode_t mode)
	{
		is_dense = true;
		for (int rank = 0; rank < TensorFormat.size(); rank++)
		{
			if (TensorFormat[rank].var == var)
			{
				TensorFormat[rank].mode = mode;
			}
			is_dense &= (TensorFormat[rank].mode == UNCOMPRESSED);
		}
		return *this;
	}



	/**
	 * 将coo里面的暂存的张量数据 pack 到实际计算时使用的buffer，按照类csr格式压缩存储。
	 * 这是由于可能reorder之后，当前的轴顺序和coo中的实际存储轴顺序不同，存储并不紧凑，这里pack
	 * 使得存储是一定相邻的存储数据
	 */
	void pack()
	{
		if (is_dense && is_dense_fix) // 稠密的不需要pack
		{
			return;
		}
		// cout << "Pack1 " << flush;
		if (T != NULL) // 若pack之前T已有值，就摧毁掉，后面会初始化
			destroy_taco_tensor_t();

		
		// cout << "Pack2 " << flush;
		auto t1 = Clock::now();
		int nnz = coo.size(); // 非零值的数量
		int num_rank = TensorFormat.size(); // 维度数量
		vector<int> newstartbit(num_rank); // 每个维度的起始bit位置
		vector<pair<uint64_t, float>> pack_coo(nnz); // 转换后->存储每个非0值的多维坐标和value
		vector<uint64_t> uniq_coords(num_rank, -1); 
		T_pos = vector<vector<int>>(num_rank, vector<int>()); 
		T_crd = vector<vector<int>>(num_rank, vector<int>());
		T_vals = vector<float>();

		
		/** pack3
		 * 填充 pack_coo， 有序的存储了所有非零值的坐标和值
		 * pack_coo[i] 代表第0个非零值，这个数组按照多维坐标数组从小到大的顺序排序的
		 * pack_coo[i].first : uint64_t 是压缩拼接存储着各个维度的
		 * pack_coo[i].second : float 是存储的值
		 * 
		 * TODO: 但是疑问在于pack_coo 应该与 coo相同，为什么要重复一遍？
		 */
		// cout << "Pack3 " << flush;

		int prefixsum = 0; // 求和所有维度的 lenbit
		for (int rank = num_rank - 1; rank >= 0; rank--)
		{
			newstartbit[rank] = prefixsum;
			prefixsum += TensorFormat[rank].lenbit;
		}

		#pragma omp parallel for
		for (int i = 0; i < nnz; i++) // 遍历每个非0值
		{
			uint64_t coords = coo[i].first; // 存储
			uint64_t pack_coords = 0;
			for (int rank = num_rank - 1; rank >= 0; rank--) // 遍历每个非零值的每个压缩维度
			{
				uint64_t rank_coord = extract( // 获得 在该维度上的坐标
					coords, TensorFormat[rank].startbit, TensorFormat[rank].lenbit
				);
				pack_coords |= (rank_coord << newstartbit[rank]);
			}
			// 仅有各个维度维度非零值存储在pack_coords中，类似拼接{i,j,k}...
			// coo[i].second 代表其中存储该坐标下的非零值
			pack_coo[i] = {pack_coords, coo[i].second}; 
		}
		
		// 将该坐标进行排序
		__gnu_parallel::sort(pack_coo.begin(), pack_coo.end());


		// cout << "Pack4 " << flush;
		/**
		 * 填充T_pos, T_crd, T_val ，将各个压缩维度的pos 和 crd数组填充
		 */
		int limit = nnz * 20; // 5e8;
		for (int i = 0; i < nnz; i++) // 遍历每个非0值
		{
			uint64_t coords = pack_coo[i].first; // 坐标
			long pos_idx = 0; // 依次展开各个维度后 累计的非零值位置
			for (int rank = 0; rank < num_rank; rank++) // 遍历每个维度 取出对应的坐标
			{
				int rank_coord = extract(coords, newstartbit[rank], TensorFormat[rank].lenbit); // 取出该维度的坐标
				uint64_t upper_coords = extractuppercoords(coords, newstartbit[rank]); // 当前维度即之后维度的坐标压缩值
				switch (TensorFormat[rank].mode) // 判断当前维度是否为 压缩格式
				{
				case UNCOMPRESSED:
					pos_idx = pos_idx * TensorFormat[rank].dimension + rank_coord; // 累计的坐标偏移*当前稠密轴长 + 当前轴中偏移 
					break;
				case COMPRESSED:
					if (upper_coords != uniq_coords[rank]) // 当uniq_coords中还未存upper_coords时候，即初始化，做以下处理
					{
						uniq_coords[rank] = upper_coords;
						T_crd[rank].push_back(rank_coord); // crd中应该存储该维度的坐标
						if (T_pos[rank].size() <= pos_idx + 1) // 首先初始化或不足时候 拓展该维度下 pos 数组的大小
						{
							if (pos_idx > limit)
							{
								T = NULL;
								throw std::invalid_argument("Pack Too Large");
							}
							T_pos[rank].resize((pos_idx + 1000000), 0);
						}
						T_pos[rank][pos_idx + 1]++; // 在pos 数组pos_idx + 1位置，即当前这一‘行’非零值+1
					}
					pos_idx = T_crd[rank].size() - 1; // 展开到当前维度后 累计的非零值位置
					break;
				}
			}
			// case VALUEARRAY:
			if (T_vals.size() <= pos_idx) // 初始化或者不足时候， 拓展存储非零值数组的大小
			{
				if (pos_idx > limit)
				{
					T = NULL;
					throw std::invalid_argument("Pack Too Large");
				}
				T_vals.resize((pos_idx + 1000000), 0);
			}
			T_vals[pos_idx] = pack_coo[i].second; // 存储
		}

		
		/**
		 * 用于修正T_pos数组，pack4的pos数组只是记录了每个区间的非零元素数量
		 * 这一pack用于累加前缀和得到真正的pos数组
		 */

		// cout << "Pack5 " << flush;
		int format_size = 0; // 记录格式存储使用的大小
		long pos_size = 1; // 记录到展开到当前维度 非零元素的数量
		for (int rank = 0; rank < num_rank; rank++) // 遍历每个维度
		{
			switch (TensorFormat[rank].mode) // 判断当前维度是否压缩属性
			{
			case UNCOMPRESSED:
				format_size += 1;
				pos_size *= TensorFormat[rank].dimension;
				break;
			case COMPRESSED:
				if (T_pos[rank].size() < pos_size)
				{
					if (pos_size > limit)
					{
						T = NULL;
						throw std::invalid_argument("Pack Too Large");
					}
					T_pos[rank].resize(pos_size + 1, 0);
				}
				for (int i = 0; i < pos_size; i++)
					T_pos[rank][i + 1] += T_pos[rank][i]; // 前面pack4只记录了每个区间的非零元素数量，现在累加前缀和得到真正pos，代表区间位置
				format_size += pos_size + T_crd[rank].size();
				pos_size = T_crd[rank].size();
				break;
			}
		}
		format_size += pos_size; // value arr
		if (T_vals.size() < pos_size)
		{
			// cout << pos_size << endl;
			if (pos_size > limit)
			{
				T = NULL;
				throw std::invalid_argument("Pack Too Large");
			}
			T_vals.resize(pos_size + 1, 0);
		}

		// cout << "Format Conversion Time (ms) : " << compute_clock(Clock::now(), t1) << " ms " << endl;
		// cout << "Converted Format Size  (MB) : " << format_size*4/1e6 << " MB " << endl;
	}

	taco_tensor_t *get_taco_tensor()
	{
		init_taco_tensor_t();
		return T;
	}
};
