# #!/bin/bash

# SPMM_BIN="./SDDMM_ASpT_SP.x"

# DATASET_DIR="../../../dataset/mtx_demo_dataset/"

# RESULT_FILE="result"

# if [ ! -f "$SPMM_BIN" ]; then
#     echo "SpMM binary not found at $SPMM_BIN"
#     exit 1
# fi

# find "$DATASET_DIR" -type f -name "*.mtx" | while read mtx_file; do
#     echo "Processing $mtx_file"

#     ./SDDMM_ASpT_SP.x "$mtx_file" >> "$RESULT_FILE"
    
#     if [ $? -ne 0 ]; then
#         echo "SpMM failed on $mtx_file"
#     fi
# done

# echo "All .mtx files have been processed."

#!/bin/bash

for i in {1..100}
do
    ./SDDMM_ASpT_SP.x ../../../dataset/mtx_demo_dataset/bcsstk38.mtx
done
# bcsstk38
# cca
# cfd1
# mhd4800a
# msc10848
# nemspmm1_16x4_0
# pf2177

