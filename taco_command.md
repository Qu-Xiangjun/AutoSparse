
```
(torch) qxj@GPU-Server:~/Workload-Aware-Co-Optimization/code_generator$ $WACO_HOME/code_generator/taco/build/bin/taco 
Usage: taco <index expression> [options]

Examples:
  taco "a(i) = b(i) + c(i)"                            # Dense vector add
  taco "a(i) = b(i) + c(i)" -f=b:s -f=c:s -f=a:s       # Sparse vector add
  taco "a(i) = B(i,j) * c(j)" -f=B:ds                  # SpMV
  taco "A(i,l) = B(i,j,k) * C(j,l) * D(k,l)" -f=B:sss  # MTTKRP

Options:
  -d=<var/tensor>:<size>      Specify the dimension of tensor modes. This can 
                              be done by either specifying the dimension of 
                              index variables, or by specifying the dimension 
                              of tensor modes. All dimensions default to 42. 
                              Examples: i:5, j:100, b:5, A:10,10. 

  -f=<tensor>:<format>        Specify the format of a tensor in the 
                              expression. Formats are specified per dimension 
                              using d (dense), s (sparse), u (sparse, not 
                              unique), q (singleton), or c (singleton, not 
                              unique). All formats default to dense. The 
                              ordering of modes can also be optionally 
                              specified as a comma-delimited list of modes in 
                              the order they should be stored. Examples: A:ds 
                              (i.e., CSR), B:ds:1,0 (i.e., CSC), c:d (i.e., 
                              dense vector), D:sss (i.e., CSF). 

  -t=<tensor>:<data type>     Specify the data type of a tensor (defaults to 
                              double).Currently loaded tensors must be 
                              double.Available types: bool, uint8, uint16, 
                              uint32, uint64, uchar, ushort,uint, ulong, 
                              ulonglong, int8, int16, int32, int64, char, 
                              short, int,long, longlong, float, double, 
                              complexfloat, complexdoubleExamples: A:uint16, 
                              b:long and D:complexfloat. 

  -s="<command>(<params>)"    Specify a scheduling command to apply to the 
                              generated code. Parameters take the form of a 
                              comma-delimited list. See -help=scheduling for a 
                              list of scheduling commands. Examples: 
                              split(i,i0,i1,16), precompute(A(i,j)*x(j),i,i). 

  -c                          Generate compute kernel that simultaneously does 
                              assembly. 

  -i=<tensor>:<filename>      Read a tensor from a file (.tns .ttx .mtx .rb). 

  -o=<tensor>:<filename>      Write a tensor to a file (.tns .ttx .mtx .rb). 

  -O=<directory path>         Write all tensors to a directory in the .tns 
                              format (defaults to $TMPDIR) 

  -g=<tensor>:<fill>          Generate data for a vector or matrix. Vectors 
                              can be d (dense sequence), r (dense random), s 
                              (sparse) or h (hypersparse). Matrices can be d, 
                              s, h or l (slicing), f (FEM), b (Blocked). 
                              Examples: B:s, c:r. 

  -time=<repeat>              Time compilation, assembly and <repeat> times 
                              computation (defaults to 1). 

  -write-time=<filename>      Write computation times in csv format to 
                              <filename> as 
                              compileTime,assembleTime,mean,stdev,median. 

  -write-compute=<filename>   Write the compute kernel to a file. 

  -write-assembly=<filename>  Write the assembly kernel to a file. 

  -write-source=<filename>    Write the C source code of the kernel functions 
                              of the given expression to a file. 

  -read-source=<filename>     Read C kernels from the file. The argument order 
                              is inferred from the index expression. If the 
                              -time option is used then the given expression 
                              and kernels are timed. 

  -verify                     Compare results of generated and read kernels 

  -print-compute              Print the compute kernel (default). 

  -print-assembly             Print the assembly kernel. 

  -print-evaluate             Print the evaluate kernel. 

  -print-kernels              Print all kernels as a C library. 

  -print-concrete             Print the concrete index notation of this 
                              expression. 

  -print-iteration-graph      Print the iteration graph of this expression in 
                              the dot format. 

  -print-nocolor              Print without colors. 

  -cuda                       Generate CUDA code for NVIDIA GPUs 

  -schedule                   Specify parallel execution schedule 

  -nthreads                   Specify number of threads for parallel execution 

  -prefix                     Specify a prefix for generated function names 

  -help                       Print this usage information. 

  -version                    Print version and build information. 

  -help=scheduling            Print information on the scheduling directives 
                              that can be passed to '-s'. 


(torch) qxj@GPU-Server:~/Workload-Aware-Co-Optimization/code_generator$ $WACO_HOME/code_generator/taco/build/bin/taco -help=scheduling
Scheduling commands modify the execution of the index expression.
The '-s' parameter specifies one or more scheduling commands.
Schedules are additive; more commands can be passed by separating
them with commas, or passing multiple '-s' parameters.

Examples:
  -s="precompute(A(i,j)*x(j),i,i)"
  -s="split(i,i0,i1,32),parallelize(i0,CPUThread,NoRaces)"

See http://tensor-compiler.org/docs/scheduling/index.html for more examples.

Commands:
  -s=pos(i, ipos, tensor)     Takes in an index variable `i` that iterates 
                              over the coordinate space of `tensor` and 
                              replaces it with a derived index variable `ipos` 
                              that iterates over the same iteration range, but 
                              with respect to the the position space. The 
                              `pos` transformation is not valid for dense 
                              level formats. 

  -s=fuse(i, j, f)            Takes in two index variables `i` and `j`, where 
                              `j` is directly nested under `i`, and collapses 
                              them into a fused index variable `f` that 
                              iterates over the product of the coordinates `i` 
                              and `j`. 

  -s=split(i, i0, i1, factor) Splits (strip-mines) an index variable `i` into 
                              two nested index variables `i0` and `i1`. The 
                              size of the inner index variable `i1` is then 
                              held constant at `factor`, which must be a 
                              positive integer. 

  -s=precompute(expr, i, iw)  Leverages scratchpad memories and reorders 
                              computations to increase locality. Given a 
                              subexpression `expr` to precompute, an index 
                              variable `i` to precompute over, and an index 
                              variable `iw` (which can be the same or 
                              different as `i`) to precompute with, the 
                              precomputed results are stored in a temporary 
                              tensor variable. 

  -s=reorder(i1, i2, ...)     Takes in a new ordering for a set of index 
                              variables in the expression that are directly 
                              nested in the iteration order. The indexes are 
                              ordered from outermost to innermost. 

  -s=bound(i, ib, b, type)    Replaces an index variable `i` with an index 
                              variable `ib` that obeys a compile-time 
                              constraint on its iteration space, incorporating 
                              knowledge about the size or structured sparsity 
                              pattern of the corresponding input. The meaning 
                              of `b` depends on the `type`. Possible bound 
                              types are: MinExact, MinConstraint, MaxExact, 
                              MaxConstraint. 

  -s=unroll(index, factor)    Unrolls the loop corresponding to an index 
                              variable `i` by `factor` number of iterations, 
                              where `factor` is a positive integer. 

  -s=parallelize(i, u, strat) tags an index variable `i` for parallel 
                              execution on hardware type `u`. Data races are 
                              handled by an output race strategy `strat`. 
                              Since the other transformations expect serial 
                              code, parallelize must come last in a series of 
                              transformations. Possible parallel hardware 
                              units are: NotParallel, GPUBlock, GPUWarp, 
                              GPUThread, CPUThread, CPUVector. Possible output 
                              race strategies are: IgnoreRaces, NoRaces, 
                              Atomics, Temporary, ParallelReduction. 

```



查看下面两个例子
``` cpp
// taco "A(i,j)=B(i,k)*C(k,j)" -f=A:dd:0,1 -f=B:sd:0,1 -f=C:dd:0,1  -verify

int compute(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *Ad, taco_tensor_t *Bd, taco_tensor_t *Cd) {
  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  double* __restrict A_vals = (double*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  int* __restrict B1_pos = (int*)(B->indices[0][0]);
  int* __restrict B1_crd = (int*)(B->indices[0][1]);
  double* __restrict B_vals = (double*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  double* __restrict C_vals = (double*)(C->vals);

  for (int32_t iB = B1_pos[0]; iB < B1_pos[1]; iB++) {
    int32_t i = B1_crd[iB];
    for (int32_t k = 0; k < C1_dimension; k++) {
      int32_t kB = iB * B2_dimension + k;
      for (int32_t j = 0; j < C2_dimension; j++) {
        int32_t jA = i * A2_dimension + j;
        int32_t jC = k * C2_dimension + j;
        A_vals[jA] = A_vals[jA] + B_vals[kB] * C_vals[jC];
      }
    }
  }
  return 0;
}

// taco "A(i,j)=B(i,k)*C(k,j)" -f=A:dd:0,1 -f=B:ds:1,0 -f=C:dd:0,1  -verify

int compute(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *Ad, taco_tensor_t *Bd, taco_tensor_t *Cd) {
  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  double* __restrict A_vals = (double*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  int* __restrict B2_pos = (int*)(B->indices[1][0]);
  int* __restrict B2_crd = (int*)(B->indices[1][1]);
  double* __restrict B_vals = (double*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  double* __restrict C_vals = (double*)(C->vals);

  for (int32_t k = 0; k < C1_dimension; k++) {
    for (int32_t iB = B2_pos[k]; iB < B2_pos[(k + 1)]; iB++) {
      int32_t i = B2_crd[iB];
      for (int32_t j = 0; j < C2_dimension; j++) {
        int32_t jA = i * A2_dimension + j;
        int32_t jC = k * C2_dimension + j;
        A_vals[jA] = A_vals[jA] + B_vals[iB] * C_vals[jC];
      }
    }
  }
  return 0;
}
```

可以看出来 对于B(i,k) 矩阵的 -f 后的ds:1,0 与 sd:0,1 是完全不一样的顺序，
ds:1,0 中ds代表遍历此数据时候，第一个循环是 dense ，第二个循环是 compress ，同时1,0 指示第一个循环 是轴1:k, 第二个循环 是轴0:i
sd:0,1 中sd代表遍历此数据时候，第一个循环是 compress ，第二个循环是 dense ，同时0,1 指示第一个循环 是轴0:i, 第二个循环 是轴1:k

为什么前一个ds;1,0 可以代表csc，后者无法呢？
  - 因为csc遍历一定是第一个循环dense，第二个循环compress。
但为什么不能是通过reorder的调度变换这个呢？
  - 因为sd即使变换为第一个循环是dense，但这个dense的轴绑定的是i，并不是k，无法对等csc格式。
  - 即因为reorder是在已定性的代码中进行更改循环顺序，

那么在-f命令中的 rank顺序也是可变的，区别于其他两种reorder：
  - freorder 是格式上的reorder，会改变实际存储中顺序
  - lreorder 是调度上的reorder，在生成的代码循环上进行交换，即遍历访问维度的顺序改变

那么如何能通过这两个reorder实现csc格式覆盖呢？
  - 通过freorder改变的是如A(i,j)=B(i,k)\*C(k,j) 到 A(i,j)=B(k,i)\*C(k,j), 也能实现csc，然后命令为-f=B:sd:0,1
  - B(k,i)\*C(k,j) 识别还是矩阵乘

但如上这些变换，感觉不能保证代码的正确性？
- 错误代码集中在于reorder的导致的循环引用错误：



可以通过 parallel 命令支持 vectorize
- 首先只能针对非reduce 轴
- 然后要对vectorize的轴进行bound (其实这个在WACO的代码中默认做了对所有的UNCOMPRESSED轴)
- 然后再parallel 的Hardware  选项选择 CPU vector


GPU上的不支持调度原语，因为只能通过parallel命令的选项绑定Thread Block Wrap等，但现在不支持。
应该是可以生成GPU代码，但不支持调度。



### WACO 调度原语特点与可能改进地方:

**格式**
WACO 只使用dense compressed 两种其他的没什么可以用的？
- 但在Format abstract 这篇问斩各种，如COO格式就需要 Compressed(!U)
- 各种属性轴的存储方案：
  - dense 不需要辅助数组，直接存在当前位置的vals
    - 注意放在 Sparse 轴之后，计算坐标的使用的是sparse 的pos数组结果，而真实坐标却是 crd数组的结果，如果中间有全0的行这样看起来是错误的，其生成的代码对于sd组合为`for (int32_t iB = B1_pos[0]; iB < B1_pos[1]; iB++) {
    int32_t i = B1_crd[iB];
    for (int32_t k = 0; k < C1_dimension; k++) {
      int32_t kB = iB * B2_dimension + k;`
      但为什么不发生错误呢，因为存储的时候也是错位的存储，访问的时候也偏差了，刚好对齐的存储和访问，因此不发生错误，但是注意会不会影响其他的地方(如其他数组C包含了k轴，但使用的是k，而不是kB);
    - 不可以放在 no_unique_sparse 后面，因为通过pos数组的值作为前一个维度的坐标，这是错误的
    - 不可以放在 singleton  no_unique_singleton 之后，因为和他们用的crd索引值一样作为前一个维度的坐标，而且一般这两个跟在 no_unique_sparse 之后
    - 如果跟在了 ucq 后面，那么后面所有的轴长都必须是1，或者前面的ucq长度都是1
  - sparse 只存储非零值的坐标到pos和crd，注意需要排除重合的前缀坐标
    - 区分前一个轴的类型来存储crd数组：
      - 若前一个轴是sparse dense，与waco相同操作
      - 若前一个轴是 no_unique_sparse 时候，当前的pos_idx应该是上一个轴的un_crd的size大小，然后以这个pos_idx push_back crd和pos[pos_idx+1]++;
      - 若前一个轴是 singleton ，处理和sparse的一样来
      - 若前一个轴是 no_unique_singleton, 处理和 no_unique_sparse 的一样来
    - 使用规则：
      - 可以放在任何位置和任何轴之后
  - no_unique_sparse 只存储所有的非零值的pos和crd，不需要排除重合的前缀坐标；但是注意其计算出的pos_idx需要排除重合前缀坐标后的crd计算出，这是因为重复的当前轴坐标crd数组长度不等于当前轴所处的index
    - 可以放在任何位置和任何轴之后
  - singleton ，
    - 区分前一个轴的类型来存储crd数组：
      - 前一个轴不存在时候，即第0个轴是singleton，则只需要存一个数在当前数组，除非当前
      - 若前一个轴是sparse 时候，当前crd数组和前一个轴的crd数组对齐，pos_idx来源于前一个数组crd的大小
      - 若前一个轴是 no_unique_sparse 时候，当前crd数组以前一个轴计算出来的的pos_idx为接下来的pos_idx，使用unique的前缀判断是否push_back, 但是注意此时只能是在最后一个轴，否则 no_unique_sparse crd数组长度大于 singleton的crd
      - 若前一个轴是dense轴，前一个轴计算出的pos_idx 应该是当前crd数组的idnex存入，而不是push_back; 然后当前轴再计算pos_idx应该不变
      - 若前一个轴是singleton, 则和前一周的crd数组对齐即可，和sparse轴的情况处理一样，pos_idx来源于前面的一样，不变
      - 若前一个轴是no_unique_singleton, 和 non_unique_sparse的一样处理 (强调：但是注意此时只能是在最后一个轴，否则 no_unique_sparse crd数组长度大于 singleton的crd)
    - 使用singlon的规则: 
      - 不可以接在dense 轴之后，除非该长度为1，否则计算会是错误的结果 `(注意访问方式是按照index索引的)`
      - 除非该长度为1，不可以单独的作为起始的轴，否则计算会是错误的结果
      - 除非轴长度为1，不可以接在 sparse, singleton 后面，否则计算错误
      - 前面可以是 no_unique_sparse 或者 no_unique_singleton，
      - 除非轴长度为1，一般是放在最后一个维度，因为非长度为1的轴肯定不会一次只存储一个crd数
      - 但是注意如果轴长为1，又不是最后一个轴，如果后面轴长度有大于1，则会出现crd数组存储出错，即会由于重复不存有些crd，导致生成的kernel遍历时候出错（在pack可以想方法支持，但no_unique_singleton可以在这里替代，因此就不这样用）。
  - no_unique_singleton ,注意其计算出的pos_idx需要排除重合前缀坐标后的crd计算出，这是因为重复的当前轴坐标crd数组长度不等于当前轴所处的index
    - 区分前一个轴的类型来存储crd数组：
      - 前一个轴不存在时候，即第0个轴是no_unique_singleton，则只需要存一个数在当前数组
      - 若前一个轴是 dense, 
        - 注意其crd数组是以 index索引访问的，而这个index索引总是在不重复的第一个坐标，将crd扩到这个长度后 再 push_back (但是注意，这么计算也应该是错误的，因为no_unique_singleton会`while (kB < pB2_end && B2_crd[kB] == k) {B_val += B_vals[kB];kB++;}` 类似的累加了再去计算，但实际这么是错误的，每一个crd中坐标指向的元素都是独立的与其他进行计算，并不需要累加在计算。)
        - 还需要注意,pack 的时候第一步crd数组的长度可能是不够dense访问的长度，需要在后面补上reseize。因为在访问的时候dense可能会遍历这一个轴包括0值的地方坐标，而这里会超过第一步pack时候装入crd的坐标范围。
        - 另外注意到`int32_t k0A = i1A;int32_t pA3_end = i1A + 1;while (k0A < pA3_end) {int32_t k0 = A3_crd[k0A];` 这样的访问，其实是只能存下一个坐标在crd，因此当前轴及之后的轴都需要长度为1,除非遇到dsu就停止要求轴长度为一，换句话说若出现了dc，则后面连续跟着的c或q的轴长必须都是1.
      - 若前一个轴是 sparse, 需要和sparse的pos和crd一一对应，因此无法跟在sparse轴之后，除非和该crd是一一对应的，即该行长度为1,**且后面所有轴长度都为1**
      - 若前一个轴是 no_unique_sparse, 则和上一个轴的pos 与 crd是一一对应的，在pack的时候确实是如此，因为un_crd会存所有的非零元素坐标
      - 若前一个轴是 singleton, 需要和crd一一对应，因此无法跟在singleton轴之后，除非和该crd是一一对应的,即该行长度为1
      - 若前一个轴是 no_unique_singleton, 可以的
    - 使用规则：
      - 不可以接在dense 轴之后，除非该长度为1，否则计算会是错误的结果 `(注意访问方式是按照index索引的)`；若出现了dc，同时后面连续跟着的c或q的轴长必须都是1.
      - 不可以作为起始的轴，区别于`singleton` 本来就只需要存一个，而这里实际是需要存所有非零值在crd，而在生成代码里面 ```int32_t iB = 0;
        int32_t pB1_end = 1;
        while (iB < pB1_end) {
          int32_t i = B1_crd[iB];
          int32_t B1_segend = iB + 1;
          while (B1_segend < pB1_end && B1_crd[B1_segend] == i) {
            B1_segend++;
          }``` 这里值会访问crd数组的第一个位置，而crd数组其他元素并未被访问到，因此错误；但是若起始轴后面跟着dense就不会发生问题了，因为dense的轴只关注un_crd的大小来作为pos_idx，但必须保证 no_unique_singleton 长度为1才行，因为起始轴生成的代码为pB1_end,(此外可以是连续长度为1的c or q or u 跟着之后，然后才是dense，这也是没问题的；若这个区间是 与到了长度为1的s，只有长度为1的s可以利用该pos值，那么也是可以的，并且可以替代d作为结束； 区间中不能是u，因为un_crd数组一直在增加，而c给出的pos实际访问不到这么大的区间，除非长度为1，则可以充当和s一样的作用)
      - 除非长度为1，前一个轴不能是sparse 和singleton，**且后面所有轴长度都为1**,否则计算错误
      - 前一个轴可以是 no_unique_sparse，no_unique_singleton
      - 不可以作为最后一个轴，除非长度为1,因为前面的`{B_val += B_vals[kB];kB++;}`累加问题是错误的计算
  - 基于错误case的格式规则：
  - `"C(i1,i0,j1,j0)=A(i1,k1,k0,i0)*B(k1,k0,j1,j0)" -f=A:sdcd:64,128,1,1` 
    - 这是由于dcd的问题，d跟在c后面通过c的虚拟pos 值来当做前面所有轴累计的坐标值，然后在pack的时候由于使用的是crd而不是un_crd的size，因此在生成的代码中直接用了第一个d给的坐标越过c给第二个d作为坐标，就和pack的坐标不一样
    - 因此不能出现d(c,q)*n次d的情况,其中至少出现一次c，d前面必须还有轴长非1的轴（即d不能作为起始轴）


**Split**
WACO 的`split`只对存储格式的轴操作，并不会使用sch上的split操作，这是因为：
- 对于存储格式的轴进行split会使得存储时候轴进行拆分，拆分后的轴如果reorder可以更改实际存储的数据相邻性和内容，如UC变成UCUU后，存储的0值可能多一些，块内部数据相邻，原来UC时候相邻的数据现在如果在同一个块还是相邻，原来隔很远的，现在在块内可能相邻了，跨块就不相邻了。
- 那么有没有必要再加一个`fsplit`之后的`lsplit`呢？
  - 我觉得加上这个是没有语法错误的，可以单独加
  - 但加了之后可以再额外的`lreorder`， 使得`lsplit`产生的新轴 区别于 存储的`forder`进行迭代？ 但是这样是不是会降低性能
  - 注意 `lsplit` 和 `singleton` 相关的轴不兼容 

**Reorder**
WACO 里面分为 `freorder` 和 `lreorder`, 但两者共用的相同的reorder顺序，实际是变换的`lreoder`中包含所有的已split轴，然后`freorder`仅从`lreorder`的结果中提取出各个张量程序轴在其中的顺序作为格式的存储顺序
- freorder的顺序决定了存储的实际物理位置，由于和lreorder相同，因此迭代遍历的时候一定最高效的按照`存储的相邻顺序访问`。
- `freorder`的顺序会决定 生成 `kernel` 表达式中轴的顺序，如此可以达到 csr -> csc 的变换
- `lreorder` 可不可以 不和 `freorder`使用相同顺序呢？ 
  - 将reorder只视为对存储格式的改变，lreorder单独处理，就像稠密中也不考虑存储格式是行主序还是列主序，reorder直接应用循环，这里也可以
  - 但是这样顺序的打乱就不会按照存储的相邻顺序访问数据？可能性能还降低了？??
  - 不过要注意一点，由于是分为稠密与稀疏的轴，稀疏轴需要依赖一个数作为pos数组的大小，因此容易出现reorder后程序有误，要做错误排除
  - 排除错误的order: 只需要处理和sparse 数组相关的轴顺序，从freorder固定sparse的轴之间order不能改变，sparse依赖前一个轴，改变后也一定在前一个轴之后；(这里简单的视线为lreorder中设计sparse 数组相关的轴和freorder中顺序一致，因为按照存储顺序访问的一定性能最好，同时freorder也在改变存储顺序的同时改变了迭代的顺序)
- 此外，由于`lsplit` 产生了一些和 `fsplit` 不一样的轴，因此更需要 `lreorder` 达到稠密调度中split 和reorder 配合的效果，让一些轴顺序变化。


**Bound**
WACO中用于直接标注上dense的轴迭代范围。方便Parallel unroll等方法的使用。

**Parallelize**
WACO 目前只对i1轴，即i轴最外层轴进行并行化，单独选择num_thread 为当前系统核数量，chunk_size即omp_sched_dynamic的块分配大小可变。
- 注意Parallel 的轴必须提前bound。
- 注意reduce 轴不能直接Parallel， 需要添加atomics。
- **注意sparse的轴无法Parallel**。
- 是否可以改变Parallel 随机选择一个lsplit后的轴，或者每次只对最外层的轴进行变换呢？
- 只能选一个
- 注意不要是规约轴，因为无法bound，就不能使用

**Unroll**
WACO并未使用这个
- 可以和Parallel一样随机选择一个轴
- 可以尝试选多个unroll，可以设置一个 count，大于这可以不用unroll，因为收益不大？
- 某些轴需要bound，这里随机咯
- 发现`sparse`轴添加无效果，因此只考虑dense轴unroll

**Vectorize**
WACO并未使用
- 可以通过Parallel中的选项实现。注意sparse的轴无法实现
- 随机选择一个和Parallel unroll不冲突的轴，保证在unroll的inner 轴即可。
- 其实可以只对最内层的

**Precompute**
WACO 并未使用。
利用暂存存储器和重新排序计算来增加局部性。给定一个要进行预计算的子表达式`expr`、一个要预计算的索引变量`i`和一个要预计算的索引变量`iw`(可以与`i`相同或不同)，预计算的结果存储在一个临时张量变量中。`-s=parallelize(i, u, strat)`
- 随机选择一个轴，然后将该轴最近的子表达式当做目标，轴名称不需要变。
- 另外也可以只选择reduce的轴或只选择space的轴进行，对比一下看看哪个OK一些作为策略？我理解是reduce的轴更需要缓存结果进行累加。即是否优先选择输出的轴
- 注意和`sinlgeton` 属性的轴不兼容


**augmented**
 we have augmented the Suitesparse dataset using the following steps:
1. Downsampled the matrix to 256x256.
2. Upsampled the 256x256 matrix to an arbitrary size of rows and columns. (e.g. 256x256 -> 16384x2048)
3. During the upsample, we randomly selected the dense block size, then fill the matrix with the dense blocks. (e.g., 2x1, 4x4, 8x2, etc.).
So If you examine each .csr file, you'll notice that the filename format is "<original matrix name>_<blocksize>_<uniqueID>.csr."


**Trick**
- 热身次数越多，得出的时间越好，热身至少100次，甚至可以500次更好，实际跑50~100次即可。。。
- 选择vag 还是 mid 策略差不多
- 发生format overleaf, 实际是装得下的，这里只是尽可能的排除需要运行时间长的设计。 
- 注意测试c++用的编译命令要-03 且NUMCORE选择正确