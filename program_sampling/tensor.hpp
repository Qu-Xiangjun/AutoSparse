#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <parallel/algorithm>

#include "utils.hpp"
#include "time.h"

using namespace std;

/* The axis property of every level in tensor.*/
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;

typedef enum{
	UNCOMPRESSED,
    COMPRESSED,
	COMPRESSED_NU,
	SINGLETON,
	SINGLETON_NU
} mode_t;

/* Tensor expression struct in taco */
typedef struct 
{
	// Tensor format meta data.
	int32_t order;			 // tensor order (number of modes)
	int32_t *dimensions;	 // tensor dimensions for every rank.
	int32_t csize;			 // component size, there will be null.
	int32_t *mode_ordering;	 // mode storage ordering point array, there will be null.
	taco_mode_t *mode_types; // mode types for every rank.

	// Tensor index info.
	uint8_t ***indices;		 // tensor index data (per mode) 3D point array.
                             // first indicate every tensor rank.
                             // sencond indicate crd and pos point for a array.
                             // third store array. 
	uint8_t *vals;			 // tensor values corresponding to coordinates from smallest 
                             // to largest.
	int32_t vals_size;		 // values array size
} taco_tensor_t;


typedef vector<pair<uint64_t, float>> Compressed_Coo;


typedef struct  
{
	string var;         // axis name
	int    dimension;	// axis length
	mode_t mode;	    // axis storage property
	// Higher rank first store in a coo format temp buffer.
    // Such as: 2D matrix, row's startbit is 0, col's startbit = row's lenbit.
    // And the indices expressed by compressed array.
	int    startbit;	// Current axis' expression in compressed bin array. 
	int    lenbit;		// Current axis' length indicates the desired binary length. 
} FormatInfo;


class Tensor
{
public:
    vector<FormatInfo> format;      // Format info in now. It can be change by schduler.
                                    // Higer axis rank  first, such as 2D matrix(i,j)
                                    // vector = [format_i, format_j]
    vector<FormatInfo> format_rst;  // Format info with origin coo formath info.
    Compressed_Coo coo;             // Store all the non zero in coo format, which can
                                    // be change by some schduler, such as fspilt, freorder.
    
    // packed tensor data and indices from coo according from schduler.
    vector<vector<int>> T_pos;      
    vector<vector<int>> T_crd;
    vector<vector<int>> T_un_pos;   // non unique pos for COMPRESSED_NU    
    vector<vector<int>> T_un_crd;   // non unique pos for COMPRESSED_NU, SINGLETON_NU 
    vector<float>       T_val;      // Store none-zero according low to high coord.
    taco_tensor_t       *T;
    
    const bool          is_dense;   // All ranks are uncompressed.
                                    // It onlu support change by construct function.


public:
    /* Construct tensor from sparse coo format. */
    Tensor(Compressed_Coo &coo_init, vector<FormatInfo> &format_init) 
    : coo(coo), format(format_init), is_dense(false)
    {
        /* Initial format compressed bit info */
        int num_rank = format.size();
        format[num_rank - 1].startbit = 0; 
        format[num_rank - 1].lenbit   = get_ceil_log2(format[num_rank - 1].dimension)
        for (int rank = num_rank - 2; rank ; rank--)
        {   // Higher rank first
            format[rank].startbit = format[rank + 1].startbit + format[rank + 1].lenbit;
            format[rank].lenbit   = get_ceil_log2(format[rank].dimension);
        }
        format_rst = format;
        /* Initial class property */
        T = NULL;
    }

    /* construct tensor from dense format. */
    Tensor(vector<float> &dense_vals, vector<FormatInfo> &format_init)
    : T_val(dense_vals), format(format_init), format_rst(format_init), is_dense(true)
    {
        /* Initial class property */
        T = NULL;
    }

    /* Destructor. */
    ~Tensor()
    {
        destroy_taco_tensor_t();
    }

    /////////////////////////
    // Class utils functions
    /////////////////////////

    vector<FormatInfo>  &get_format()   { return format; }
    vector<float>       &get_val()      { return T_val; }
    vector<vector<int>> &get_pos()      { return T_pos; }
    vector<vector<int>> &get_crd()      { return T_crd; }

    /*  Clear all the change by shedule and pack to format. */
    void reset() 
    {
        format = format_rst;
        T_pos.clear();
        T_crd.clear();
        if(!is_dense) T_crd.clear();
        destroy_taco_tensor_t()
    }

    /* Check whether axis exist by search axis name. */
    bool is_var_exist(string var)
	{
		for (int rank = 0; rank < format.size(); rank++)
		{
			if (format[rank].var == var)
			{
				return true;
			}
		}
		return false;
	}

    /* Print format infomation and coo info. */
    string print_format()
	{
		stringstream ss;
		ss << "Rank\tVar\tDim\tMode\tStartb\tLenb" << endl;
		int rank = 0;
		for (FormatInfo &f : format)
		{
			ss << rank++ << "\t";						
			ss << f.var << "\t";					
			ss << f.dimension << "\t";		
            vector<string> mode = {
                "UNCOMPRESSED", "COMPRESSED", "COMPRESSED_NU", 
                "SINGLETON", "SINGLETON_NU"
            };
			ss << mode[static_cast<int>(f.mode)] << "\t";
			ss << f.startbit << "\t";				
			ss << f.lenbit;						
			ss << endl;
		}
		return ss.str();
	}

    /* Get axis index in format info vector by axis name. */
    int get_axis_format_info(string var)
    {
        for (int rank = 0; rank < format.size(); rank++)
        {
            if(var == format[rank].var) return rank;
        }
        stringstring ss;
        ss << "[ERROR][Tensor] The axis name " << var << " don't exist";
        throw std::exception(ss);
        exit(-1);
    }


    //////////////////////////
    // Format scheduler
    //////////////////////////

    /**
     * Split storage format axis. Change format bit info.
     * Parameters
     * ----------
     * arg1 : var
     *   axis name.
     * arg2 : outer_var
     *   The outer loop, takes the position of the computed assignment statement 
     *   as the innermost layer.
     * arg3 : inner_var
     *   The inner loop is closer to innermost computed assignment statement.
     * arg4 : factor
     *   Split factor, must be power for 2.
     */
    void split(string var, string outer_var, string inner_var, int factor)
    {
        if(!is_power_of_2(spilt_size)) 
        {
            stringstring ss;
            ss << "[ERROR][Tensor] The split factor " << factor << " must be power for 2.";
            throw std::exception(ss);
            exit(-1);
        }
        int rank = get_axis_format_info(var);
        FormatInfo outer_axis;
        outer_axis.var = outer_var;
        outer_axis.dimension = min(spilt_size, format[rank].dimension);
        outer_axis.mode = format[rank].mode;
        outer_axis.startbit = format[rank].startbit;
        outer_axis.lenbit = get_ceil_log2(outer_axis.dimension);

        format[rank].var = inner_var;
        format[rank].dimension = 
            (format[rank].dimension + outer_axis.dimension - 1) / outer_axis.dimension;
        format[rank].mode = format[rank].mode;
        format[rank].startbit = format[rank].startbit + outer_axis.lenbit;
        format[rank].lenbit = get_ceil_log2(format[rank].dimension);

        format.insert(format.begin() + rank, outer_axis);
    }

    /**
     * Reorder all the axes, but coo position info will not change.
     * This schedule will affect the compacted storage order after pack.
     * Parameter
     * ---------
     * arg : reorder_vars
     *   All the axis name contained in the vector and which indicate order.
     */
    void reorder(vector<string> reorder_vars)
    {
        if (reorder_vars.size() != format.size())
        {
            throw std::exception(
                "[ERROR][Tensor] The reorder var vector must contain all the axes."
            );
            exit(-1);
        }
        

        vector<FormatInfo> reordered_format;
        for(string &var : reorder_vars)
        {
            int rank = get_axis_format_info(var);
            reordered_format.push_back(format[rank]);
        }
        format = reordered_format;
    }

    /**
     * Change the mode for a sparse tensor axis.
     * Parameters
     * ----------
     * arg1 : var
     *   axis name.
     * arg2 : mode mode_t
     *   Changed mode.
     */
    void mode(string var, mode_t mode)
    {
        if(is_dense) 
        {
            throw std::exception(
                "[ERROR][Tensor] Dense array don't support change axis mode."
            );
            exit(-1);
        }

        int rank = get_axis_format_info(var);
        format[rank].mode = mode;
    }


    //////////////////////////
    // Pack Tensor
    //////////////////////////

    /* Pack to taco_tensor_t from current format. */
    void init_taco_tensor_t()
    {
        if(T != NULL) destroy_taco_tensor_t();
        T = new taco_tensor_t;
        num_rank = format.size();
        T->order = num_rank;
        T->dimensions = new int32_t[num_rank];
        T->mode_types = new taco_mode_t[num_rank];
        T->indices    = new uint8_t**[num_rank];
        for (int rank = 0; rank < num_rank; rank++)
        {
            T->dimensions[rank] = format[rank].dimension;
            T->mode_types[rank] = format[rank].mode == UNCOMPRESSED ? 
                                    taco_mode_dense : taco_mode_sparse
            T->indices[rank] = new uint8_t*[2];
            switch (format[rank].mode)
            {
            case COMPRESSED:
                T->indices[rank][0] = (uint8_t *)(T_pos[rank].data()); 
                T->indices[rank][1] = (uint8_t *)(T_crd[rank].data()); 
                break;
            case COMPRESSED_NU:
                T->indices[rank][0] = (uint8_t *)(T_pos[rank].data()); 
                T->indices[rank][1] = (uint8_t *)(T_crd[rank].data()); 
                break;
            case SINGLETON:
                T->indices[rank][1] = (uint8_t *)(T_crd[rank].data()); 
                break;
            case SINGLETON_NU:
                T->indices[rank][1] = (uint8_t *)(T_crd[rank].data()); 
                break;
            default: 
                /* UNCOMPRESSED axis don't need indices info */
                break;
            }
        }
    }

    /* Destroy exist taco_tensor_t. */
    void destroy_taco_tensor_t()
    {
        if(T == NULL) return;
        int num_rank = T->order;
        delete[] T->dimensions;
        delete[] T->mode_types;
        for (int rank = 0; rank < num_rank; rank++)
        {
            delete[] T->indices[rank];
        }
        delete[] T->indices;
        delete   T;
    }

    void pack()
    {
        if(is_dense)
        {
            init_taco_tensor_t();
            return ;
        }

        /* Init some variable. */
        auto start_time = Clock::now();
        int nnz = coo.size();
        int num_rank = format.size();
        vector<int> newstartbit(num_rank);
        Compressed_Coo packed_coo(nnz);
        vector<uint64_t> unique_corrds(num_rank, -1);
        T_pos = vector<vector<int>>(num_rank, vector<int>());
        T_crd = vector<vector<int>>(num_rank, vector<int>());

        int limit = nnz * 20; // 5e8
        int format_size = 0; // Count the format storage int32 overhead.

        /* Pack to packed coo copied deeply from coo, and resort packed coo. */
        // #pragma omp parallel for
        for(int i = 0; i < nnz; i++) // Traverse all the none-zero value in coo.
        {
            float value = coo[i].second;
            uint64_t packed_corrds = coo[i].first;
            packed_coo[i] = {packed_corrds, value};
        }
        // __gnu_parallel::sort(packed_coo.begin(), packed_coo.end());
        sort(packed_coo.begin(), packed_coo.end());

        /* Pack to taco tensor type buffer according to current format. */
        for (int i = 0; i < nnz; i++) // Traverse all the nnz value to pack
        {
            uint64_t coords = packed_coo[i].first;
            float value = packed_coo[i].second;
            long pos_idx = 0; // accumulate none-zero value as each dimensio unfolds.
            // Traverse all the rank from high to low axis.
            for (int rank = 0; rank < num_rank; rank++) 
            {
                // current rank's coord
                int rank_coord = extract(coords, format[rank].startbit, format[rank].lenbit);
                // current rank's coord with all the upper rank coords in compressed bin.
                uint64_t upper_coords = extract_upper_coords(coords, format[rank].startbit);
                {
				case UNCOMPRESSED:
                    pos_idx = pos_idx * format[rank].dimension + coords;
                    break;
                case SINGLETON_NU:
                case COMPRESSED_NU:
                    T_un_crd[rank].push_back(rank_coord);
                    if (T_un_pos[rank].size() <= pos_idx + 1)
                    {
                        if (pos_idx > limit)
                        {
                            throw std::invalid_argument( 
                                "[ERROR][Tensor] Format storage size overleaf when pack." 
                            );
                            exit(-1);
                        }
                        T_un_pos[rank].resize((pos_idx + 1000000), 0);
                    }
                    T_un_pos[rank][pos_idx + 1]++;
                    // notice there is no break, because need to excute next code for T_pos T_crd,
                    // which can get unique crd size for pos_idx.
                case SINGLETON: // SINGLETON only need crd array, so implemente same with COMPRESSED.
                case COMPRESSED:
                    // Notice coords is sorted, so that same upper coords in this rank indicates
                    // same coord the axis. Then first initial when come in new upper rank,
                    // because T_crd array is unique in same axis coords.
                    if(upper_coords != unique_corrds[rank])
                    {   
                        unique_corrds[rank] = upper_coords;
                        T_crd[rank].push_back(rank_coord);
                        // expand T_pos array size. because pos array size always be last
                        // rank none-zero count + 1.
                        if (T_pos[rank].size() <= pos_idx + 1)
                        {
                            if (pos_idx > limit)
                            {
                                throw std::invalid_argument(
                                    "[ERROR][Tensor] Format storage size overleaf when pack." 
                                );
                                exit(-1);
                            }
                            T_pos[rank].resize((pos_idx + 1000000), 0);
                        }
                        // Notice now the pos array have not prefixsumed.
                        T_pos[rank][pos_idx + 1]++;  
                    }
                    // Up to now rank, none-zero count. nnz is same with crd size.
                    pos_idx = T_crd[rank].size() - 1; 
                    break;
                
                default: 
                    /* UNCOMPRESSED axis  */
                    pos_idx = pos_idx * format[rank].dimension + coords;
                    break;
                }
            }
            // Store value for the indice.
            if (T_val.size() <= pos_idx)
            {
                if (pos_idx > limit)
                {
                    throw std::invalid_argument( 
                        "[ERROR][Tensor] Format storage size overleaf when pack."
                    );
                    exit(-1);
                }
                T_val.resize((pos_idx + 1000000), 0);
            }
            T_val[pos_idx] = value; // Only mentioned in this position index stored none-zero value.
        }
        
        /* Do the prefixsum for T_pos, which had only count number of intervals. */
        long pos_size = 1; // Record every level pos array size, which can accumulate to format_size.
        for (int rank = 0; rank <= num_rank; rank++)
        {
            switch (format[rank].mode)
            {
            case UNCOMPRESSED:
                format_size += 1; // Only record axis size.
                pos_size *= format[rank].dimension;
                break;
            case COMPRESSED:
                for (int i = 0; i < pos_size; i++)
                    T_pos[rank][i + 1] += T_pos[rank][i]; // Prefixsum
                format_size = pos_size + 1 + T_crd[rank].size();
				pos_size = T_crd[rank].size();
                break;
            case COMPRESSED_NU:
                for (int i = 0; i < pos_size; i++)
                    T_un_pos[rank][i + 1] += T_un_pos[rank][i]; // Prefixsum
                format_size = pos_size + 1 + T_un_crd[rank].size();
				pos_size = T_crd[rank].size();
                break;
            case SINGLETON:
                /* Don't need pos array */
                break;
            case SINGLETON_NU:
                /* Don't need pos array */
                break;
            default: 
                break;
            }
        }
        
        init_taco_tensor_t();

    }

}

#endif // FORMAT_SCHEDULE_HPP