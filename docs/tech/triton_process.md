
# Triton 编译流程

> 本文梳理了从 triton DSL 到执行的逐过程

## Step 1: Python 代码

我们直接以 triton 教程中的一段向量相加为例。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # 指向第一个输入向量的指针。
               y_ptr,  # 指向第二个输入向量的指针。
               output_ptr,  # 指向输出向量的指针。
               n_elements,  # 向量的大小。
               BLOCK_SIZE: tl.constexpr,  # 每个程序应处理的元素数量。
               # NOTE：`constexpr` 因此它可以用作形状值。
               ):
    # 有多个“程序”处理不同的数据。需要确定是哪一个程序：
    pid = tl.program_id(axis=0)  # 使用 1D 启动网格，因此轴为 0。
    # 该程序将处理相对初始数据偏移的输入。
    # 例如，如果有一个长度为 256, 块大小为 64 的向量，程序将各自访问 [0:64, 64:128, 128:192, 192:256] 的元素。
    # 注意 offsets 是指针列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建掩码以防止内存操作超出边界访问。
    mask = offsets < n_elements
    # 从 DRAM 加载 x 和 y，如果输入不是块大小的整数倍，则屏蔽掉任何多余的元素。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 将 x + y 写回 DRAM。
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # 需要预分配输出
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # 这里的 SPMD 启动 grid 表示并行运行的内核实例的数量
    # 类似 CUDA 启动 grid。可以是 Tuple[int]，也可以是 Callable(meta) -> Tuple[int]
    # 在这种情况下，使用 1D 网格，其中大小是块的数量
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    # NOTE:
    #  - 每个 torch.tensor 对象都会隐式转换为其第一个元素的指针
    #  - triton.jit 函数可以通过启动网格索引来获得可调用的 GPU 内核
    #  - 不要忘记以关键字参数传递元参数
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # 返回 z 的句柄，但由于 torch.cuda.synchronize() 尚未被调用，此时内核仍在异步运行
    return output

# 调用测试
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device="cuda")
y = torch.rand(size, device="cuda")
output = add(x, y)
print(output)
```

其中的 `add_kernel` 是我们这次的主角。直接运行这段代码，triton 会在 `~/.triton/cache` 中保存此次的所有中间产物。

## Step 2: Triton DSL 代码

此处的代码实际上是 Triton JIT 编译流程的输入，即 Python DSL。是编译流程的起点。

该代码中包含了最高级别的抽象：

```c
module {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("x_ptr"(#loc)), %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("y_ptr"(#loc)), %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("output_ptr"(#loc)), %n_elements: i32 {tt.divisibility = 16 : i32} loc("n_elements"(#loc))) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32 loc(#loc18)
    %block_start = arith.constant 1024 : i32 loc(#loc19)
    %block_start_0 = arith.constant 1024 : i32 loc(#loc19)
    %block_start_1 = arith.extsi %pid : i32 to i64 loc(#loc19)
    %block_start_2 = arith.extsi %block_start_0 : i32 to i64 loc(#loc19)
    %block_start_3 = arith.muli %block_start_1, %block_start_2 : i64 loc(#loc19)
    %block_start_4 = arith.constant 2147483647 : i64 loc(#loc19)
    %block_start_5 = arith.constant -2147483648 : i64 loc(#loc19)
    %block_start_6 = arith.cmpi sle, %block_start_3, %block_start_4 : i64 loc(#loc19)
    %block_start_7 = arith.cmpi sge, %block_start_3, %block_start_5 : i64 loc(#loc19)
    %block_start_8 = arith.andi %block_start_6, %block_start_7 : i1 loc(#loc19)
    %block_start_9 = arith.muli %pid, %block_start_0 : i32 loc(#loc19)
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc20)
    %offsets_10 = tt.splat %block_start_9 : i32 -> tensor<1024xi32> loc(#loc21)
    %offsets_11 = arith.extsi %offsets_10 : tensor<1024xi32> to tensor<1024xi64> loc(#loc21)
    %offsets_12 = arith.extsi %offsets : tensor<1024xi32> to tensor<1024xi64> loc(#loc21)
    %offsets_13 = arith.addi %offsets_11, %offsets_12 : tensor<1024xi64> loc(#loc21)
    %offsets_14 = arith.constant 2147483647 : i64 loc(#loc21)
    %offsets_15 = arith.constant -2147483648 : i64 loc(#loc21)
    %offsets_16 = arith.constant dense<2147483647> : tensor<1024xi64> loc(#loc21)
    %offsets_17 = arith.cmpi sle, %offsets_13, %offsets_16 : tensor<1024xi64> loc(#loc21)
    %offsets_18 = arith.constant dense<-2147483648> : tensor<1024xi64> loc(#loc21)
    %offsets_19 = arith.cmpi sge, %offsets_13, %offsets_18 : tensor<1024xi64> loc(#loc21)
    %offsets_20 = arith.andi %offsets_17, %offsets_19 : tensor<1024xi1> loc(#loc21)
    %offsets_21 = arith.addi %offsets_10, %offsets : tensor<1024xi32> loc(#loc21)
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc22)
    %mask_22 = arith.cmpi slt, %offsets_21, %mask : tensor<1024xi32> loc(#loc22)
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc23)
    %x_23 = tt.addptr %x, %offsets_21 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc23)
    %x_24 = tt.load %x_23, %mask_22 : tensor<1024x!tt.ptr<f32>> loc(#loc24)
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc25)
    %y_25 = tt.addptr %y, %offsets_21 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc25)
    %y_26 = tt.load %y_25, %mask_22 : tensor<1024x!tt.ptr<f32>> loc(#loc26)
    %output = arith.addf %x_24, %y_26 : tensor<1024xf32> loc(#loc27)
    %0 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc11)
    %1 = tt.addptr %0, %offsets_21 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc11)
    tt.store %1, %output, %mask_22 : tensor<1024x!tt.ptr<f32>> loc(#loc12)
    tt.return loc(#loc13)
  } loc(#loc)
} loc(#loc)
```

## Step 3: TTIR (Triton IR)

Triton JIT 的第一步是将 Python DSL 转换成 Triton 自己的 IR。

在这一步中，所完成的事项有三类：
1. 解析：将 Python 文本代码转换为一个 AST(抽象语法树)，层次化表示了原始代码中的计算逻辑。
2. 函数、类型、语义检查：识别 Triton 内置函数，进行类型推断和语法检查等。
3. 转换：按照 Triton 内部设定转换为自身可识别的 IR 格式。通常涉及到一些 Triton 操作的特定内容生成，例如并行模型转换、内存访问抽象化等。

```c
module {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("x_ptr"(#loc)), %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("y_ptr"(#loc)), %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("output_ptr"(#loc)), %n_elements: i32 {tt.divisibility = 16 : i32} loc("n_elements"(#loc))) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %pid = tt.get_program_id x : i32 loc(#loc19)
    %block_start = arith.muli %pid, %c1024_i32 : i32 loc(#loc20)
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc21)
    %offsets_0 = tt.splat %block_start : i32 -> tensor<1024xi32> loc(#loc22)
    %offsets_1 = arith.addi %offsets_0, %offsets : tensor<1024xi32> loc(#loc22)
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc23)
    %mask_2 = arith.cmpi slt, %offsets_1, %mask : tensor<1024xi32> loc(#loc23)
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc24)
    %x_3 = tt.addptr %x, %offsets_1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc24)
    %x_4 = tt.load %x_3, %mask_2 : tensor<1024x!tt.ptr<f32>> loc(#loc25)
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc26)
    %y_5 = tt.addptr %y, %offsets_1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc26)
    %y_6 = tt.load %y_5, %mask_2 : tensor<1024x!tt.ptr<f32>> loc(#loc27)
    %output = arith.addf %x_4, %y_6 : tensor<1024xf32> loc(#loc28)
    %0 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc12)
    %1 = tt.addptr %0, %offsets_1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc12)
    tt.store %1, %output, %mask_2 : tensor<1024x!tt.ptr<f32>> loc(#loc13)
    tt.return loc(#loc14)
  } loc(#loc)
} loc(#loc)
```

## Step 4: TTGIR (Triton Global IR)

在转换到该 IR 的过程中，Triton 主要将 Kernel 中独立、分散的计算逻辑提升为可以被整个程序全局分析和优化的结构，并且包含了更多内存相关信息。

其主要作用为：
1. 规范化格式和局部优化：采用 SSA 转换，确保每个变量只被复制一次；消除死边。
2. **识别内存访存方式**：识别全局内存区域中被频繁使用的内存区域进行重点标记；识别和简化指针计算，合并内存索引相关计算，简化表达式。
3. 优化控制流表示：识别循环中的不变量提取到外部；识别循环的计数器、增量、退出方式。

总的来说，该转换旨在更加清晰地描述数据流。

```c
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("x_ptr"(#loc)), %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("y_ptr"(#loc)), %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("output_ptr"(#loc)), %n_elements: i32 {tt.divisibility = 16 : i32} loc("n_elements"(#loc))) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %pid = tt.get_program_id x : i32 loc(#loc19)
    %block_start = arith.muli %pid, %c1024_i32 : i32 loc(#loc20)
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked> loc(#loc21)
    %offsets_0 = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked> loc(#loc22)
    %offsets_1 = arith.addi %offsets_0, %offsets : tensor<1024xi32, #blocked> loc(#loc22)
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32, #blocked> loc(#loc23)
    %mask_2 = arith.cmpi slt, %offsets_1, %mask : tensor<1024xi32, #blocked> loc(#loc23)
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc24)
    %x_3 = tt.addptr %x, %offsets_1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked> loc(#loc24)
    %x_4 = tt.load %x_3, %mask_2 : tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc25)
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc26)
    %y_5 = tt.addptr %y, %offsets_1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked> loc(#loc26)
    %y_6 = tt.load %y_5, %mask_2 : tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc27)
    %output = arith.addf %x_4, %y_6 : tensor<1024xf32, #blocked> loc(#loc28)
    %0 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc12)
    %1 = tt.addptr %0, %offsets_1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked> loc(#loc12)
    tt.store %1, %output, %mask_2 : tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc13)
    tt.return loc(#loc14)
  } loc(#loc)
} loc(#loc)
```

## Step 5: LLIR (LLVM IR)

这一步转换是 Triton 代码能执行的关键，在这一步中，Trion JIT 编译器将 Triton 中专有的面向并行计算的抽象转换成通用的 LLVM 框架能够理解和处理的机器无关指令。

在这一步中，主要涉及的是：
1. 内存抽象的降级：将 TTGIR 中标记的访问模式翻译成 LLVM IR 接受的指令；为 TTGIR 中的抽象指针赋予正确的 LLVM 地址空间 ID；将 TTGIR 中索引计算的指令转换为 LLVM 中的准确计算。
2. 并行抽象的降级：将 TTGIR 中 pid 相关的抽象翻译为对特定 CUDA 寄存器的访问；降级 TTGIR 中的同步操作，翻译为 CUDA 的同步函数。
3. 控制流抽象的降级：将循环分支转化为 LLVM 基本块结构并通过 Phi 节点处理；将判断分枝转换为 Block 和 br 指令构成的 CFG
4. 类型与读写指令抽象的降级：triton 函数降级为 LLVM 指令，相关类型降级为标准类型。

```c
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"

; Function Attrs: nounwind
define ptx_kernel void @add_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %3, ptr addrspace(1) readnone captures(none) %4, ptr addrspace(1) readnone captures(none) %5) local_unnamed_addr #0 !dbg !4 {
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !7
  %8 = shl i32 %7, 10, !dbg !8
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !9
  %10 = shl nuw nsw i32 %9, 2, !dbg !9
  %11 = and i32 %10, 508, !dbg !9
  %12 = or disjoint i32 %11, %8, !dbg !10
  %13 = or disjoint i32 %12, 512, !dbg !10
  %14 = icmp slt i32 %12, %3, !dbg !11
  %15 = icmp slt i32 %13, %3, !dbg !11
  %16 = sext i32 %12 to i64, !dbg !12
  %17 = getelementptr float, ptr addrspace(1) %0, i64 %16, !dbg !12
  %18 = sext i32 %13 to i64, !dbg !12
  %19 = getelementptr float, ptr addrspace(1) %0, i64 %18, !dbg !12
  %20 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0;\0A\09mov.u32 $1, 0x0;\0A\09mov.u32 $2, 0x0;\0A\09mov.u32 $3, 0x0;\0A\09@$5 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l,b"(ptr addrspace(1) %17, i1 %14) #2, !dbg !13
  %21 = extractvalue { i32, i32, i32, i32 } %20, 0, !dbg !13
  %22 = extractvalue { i32, i32, i32, i32 } %20, 1, !dbg !13
  %23 = extractvalue { i32, i32, i32, i32 } %20, 2, !dbg !13
  %24 = extractvalue { i32, i32, i32, i32 } %20, 3, !dbg !13
  %25 = bitcast i32 %21 to float, !dbg !13
  %26 = bitcast i32 %22 to float, !dbg !13
  %27 = bitcast i32 %23 to float, !dbg !13
  %28 = bitcast i32 %24 to float, !dbg !13
  %29 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0;\0A\09mov.u32 $1, 0x0;\0A\09mov.u32 $2, 0x0;\0A\09mov.u32 $3, 0x0;\0A\09@$5 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l,b"(ptr addrspace(1) %19, i1 %15) #2, !dbg !13
  %30 = extractvalue { i32, i32, i32, i32 } %29, 0, !dbg !13
  %31 = extractvalue { i32, i32, i32, i32 } %29, 1, !dbg !13
  %32 = extractvalue { i32, i32, i32, i32 } %29, 2, !dbg !13
  %33 = extractvalue { i32, i32, i32, i32 } %29, 3, !dbg !13
  %34 = bitcast i32 %30 to float, !dbg !13
  %35 = bitcast i32 %31 to float, !dbg !13
  %36 = bitcast i32 %32 to float, !dbg !13
  %37 = bitcast i32 %33 to float, !dbg !13
  %38 = getelementptr float, ptr addrspace(1) %1, i64 %16, !dbg !14
  %39 = getelementptr float, ptr addrspace(1) %1, i64 %18, !dbg !14
  %40 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0;\0A\09mov.u32 $1, 0x0;\0A\09mov.u32 $2, 0x0;\0A\09mov.u32 $3, 0x0;\0A\09@$5 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l,b"(ptr addrspace(1) %38, i1 %14) #2, !dbg !15
  %41 = extractvalue { i32, i32, i32, i32 } %40, 0, !dbg !15
  %42 = extractvalue { i32, i32, i32, i32 } %40, 1, !dbg !15
  %43 = extractvalue { i32, i32, i32, i32 } %40, 2, !dbg !15
  %44 = extractvalue { i32, i32, i32, i32 } %40, 3, !dbg !15
  %45 = bitcast i32 %41 to float, !dbg !15
  %46 = bitcast i32 %42 to float, !dbg !15
  %47 = bitcast i32 %43 to float, !dbg !15
  %48 = bitcast i32 %44 to float, !dbg !15
  %49 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0;\0A\09mov.u32 $1, 0x0;\0A\09mov.u32 $2, 0x0;\0A\09mov.u32 $3, 0x0;\0A\09@$5 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l,b"(ptr addrspace(1) %39, i1 %15) #2, !dbg !15
  %50 = extractvalue { i32, i32, i32, i32 } %49, 0, !dbg !15
  %51 = extractvalue { i32, i32, i32, i32 } %49, 1, !dbg !15
  %52 = extractvalue { i32, i32, i32, i32 } %49, 2, !dbg !15
  %53 = extractvalue { i32, i32, i32, i32 } %49, 3, !dbg !15
  %54 = bitcast i32 %50 to float, !dbg !15
  %55 = bitcast i32 %51 to float, !dbg !15
  %56 = bitcast i32 %52 to float, !dbg !15
  %57 = bitcast i32 %53 to float, !dbg !15
  %58 = fadd float %25, %45, !dbg !16
  %59 = fadd float %26, %46, !dbg !16
  %60 = fadd float %27, %47, !dbg !16
  %61 = fadd float %28, %48, !dbg !16
  %62 = fadd float %34, %54, !dbg !16
  %63 = fadd float %35, %55, !dbg !16
  %64 = fadd float %36, %56, !dbg !16
  %65 = fadd float %37, %57, !dbg !16
  %66 = getelementptr float, ptr addrspace(1) %2, i64 %16, !dbg !17
  %67 = getelementptr float, ptr addrspace(1) %2, i64 %18, !dbg !17
  %68 = bitcast float %58 to i32, !dbg !18
  %69 = bitcast float %59 to i32, !dbg !18
  %70 = bitcast float %60 to i32, !dbg !18
  %71 = bitcast float %61 to i32, !dbg !18
  tail call void asm sideeffect "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l,b"(i32 %68, i32 %69, i32 %70, i32 %71, ptr addrspace(1) %66, i1 %14) #2, !dbg !18
  %72 = bitcast float %62 to i32, !dbg !18
  %73 = bitcast float %63 to i32, !dbg !18
  %74 = bitcast float %64 to i32, !dbg !18
  %75 = bitcast float %65 to i32, !dbg !18
  tail call void asm sideeffect "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l,b"(i32 %72, i32 %73, i32 %74, i32 %75, ptr addrspace(1) %67, i1 %15) #2, !dbg !18
  ret void, !dbg !19
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { nounwind "nvvm.reqntid"="128" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "triton_add.py", directory: "/home/miner/code/triton")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!4 = distinct !DISubprogram(name: "add_kernel", linkageName: "add_kernel", scope: !1, file: !1, line: 6, type: !5, scopeLine: 6, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{}
!7 = !DILocation(line: 14, column: 24, scope: !4)
!8 = !DILocation(line: 18, column: 24, scope: !4)
!9 = !DILocation(line: 19, column: 41, scope: !4)
!10 = !DILocation(line: 19, column: 28, scope: !4)
!11 = !DILocation(line: 21, column: 21, scope: !4)
!12 = !DILocation(line: 23, column: 24, scope: !4)
!13 = !DILocation(line: 23, column: 16, scope: !4)
!14 = !DILocation(line: 24, column: 24, scope: !4)
!15 = !DILocation(line: 24, column: 16, scope: !4)
!16 = !DILocation(line: 25, column: 17, scope: !4)
!17 = !DILocation(line: 27, column: 26, scope: !4)
!18 = !DILocation(line: 27, column: 35, scope: !4)
!19 = !DILocation(line: 27, column: 4, scope: !4)
```

## Step 6: PTX (Parallel Thread eXecution)

转换到 PTX 的过程由 LLVM-NVPTX 后端和 NVVM (NVIDIA Virtual Machine) 共同完成，目的是将机器无关的 LLIR 转换为 GPU 虚拟指令集 PTX。

在 PTX 虚拟指令集中，使用虚拟寄存器进行编写。在实际运行前，PTX 代码会被 CUDA 转换为 SASS 代码，在这个转换过程中才会真正绑定实际的寄存器。

这一步的转换中，主要考虑 NVIDIA GPU 的特性来对代码进行优化，确保 LLVM IR 的转换是 GPU 友好的。具体来说，这一步所做的内容主要是：
1. 应用 GPU 特有优化
2. 翻译 LLVM IR 语句：每句 LLVM IR 会被翻译为一条或多条 PTX。
3. 生成 CUDA 运行时环境的元数据：生成 PTX 头部（PTX 版本，目标架构，指针位数）；生成函数和声明，明确 Kernel 入口点，指定代码所需的寄存器数量、共享内存大小、所需的最大线程数等。

```c
//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	add_kernel              // -- Begin function add_kernel
                                        // @add_kernel
.visible .entry add_kernel(
	.param .u64 .ptr .global .align 1 add_kernel_param_0,
	.param .u64 .ptr .global .align 1 add_kernel_param_1,
	.param .u64 .ptr .global .align 1 add_kernel_param_2,
	.param .u32 add_kernel_param_3,
	.param .u64 .ptr .global .align 1 add_kernel_param_4,
	.param .u64 .ptr .global .align 1 add_kernel_param_5
)
.reqntid 128
{
	.reg .pred 	%p<7>;
	.reg .b32 	%r<33>;
	.reg .b64 	%rd<11>;
	.loc	1 6 0                           // triton_add.py:6:0
$L__func_begin0:
	.loc	1 6 0                           // triton_add.py:6:0

// %bb.0:
	ld.param.b64 	%rd7, [add_kernel_param_0];
	ld.param.b64 	%rd8, [add_kernel_param_1];
$L__tmp0:
	.loc	1 14 24                         // triton_add.py:14:24
	mov.u32 	%r25, %ctaid.x;
	.loc	1 18 24                         // triton_add.py:18:24
	shl.b32 	%r26, %r25, 10;
	ld.param.b64 	%rd9, [add_kernel_param_2];
	ld.param.b32 	%r27, [add_kernel_param_3];
	.loc	1 19 41                         // triton_add.py:19:41
	mov.u32 	%r28, %tid.x;
	shl.b32 	%r29, %r28, 2;
	and.b32 	%r30, %r29, 508;
	.loc	1 19 28                         // triton_add.py:19:28
	or.b32 	%r31, %r30, %r26;
	or.b32 	%r32, %r31, 512;
	.loc	1 21 21                         // triton_add.py:21:21
	setp.lt.s32 	%p1, %r31, %r27;
	setp.lt.s32 	%p2, %r32, %r27;
	.loc	1 23 24                         // triton_add.py:23:24
	mul.wide.s32 	%rd10, %r31, 4;
	add.s64 	%rd1, %rd7, %rd10;
	add.s64 	%rd2, %rd1, 2048;
	.loc	1 23 16                         // triton_add.py:23:16
	// begin inline asm
	mov.u32 %r1, 0x0;
	mov.u32 %r2, 0x0;
	mov.u32 %r3, 0x0;
	mov.u32 %r4, 0x0;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r5, 0x0;
	mov.u32 %r6, 0x0;
	mov.u32 %r7, 0x0;
	mov.u32 %r8, 0x0;
	@%p2 ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];
	// end inline asm
	.loc	1 24 24                         // triton_add.py:24:24
	add.s64 	%rd3, %rd8, %rd10;
	add.s64 	%rd4, %rd3, 2048;
	.loc	1 24 16                         // triton_add.py:24:16
	// begin inline asm
	mov.u32 %r9, 0x0;
	mov.u32 %r10, 0x0;
	mov.u32 %r11, 0x0;
	mov.u32 %r12, 0x0;
	@%p1 ld.global.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r13, 0x0;
	mov.u32 %r14, 0x0;
	mov.u32 %r15, 0x0;
	mov.u32 %r16, 0x0;
	@%p2 ld.global.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd4 + 0 ];
	// end inline asm
	.loc	1 25 17                         // triton_add.py:25:17
	add.f32 	%r17, %r1, %r9;
	add.f32 	%r18, %r2, %r10;
	add.f32 	%r19, %r3, %r11;
	add.f32 	%r20, %r4, %r12;
	add.f32 	%r21, %r5, %r13;
	add.f32 	%r22, %r6, %r14;
	add.f32 	%r23, %r7, %r15;
	add.f32 	%r24, %r8, %r16;
	.loc	1 27 26                         // triton_add.py:27:26
	add.s64 	%rd5, %rd9, %rd10;
	add.s64 	%rd6, %rd5, 2048;
	.loc	1 27 35                         // triton_add.py:27:35
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd5 + 0 ], { %r17, %r18, %r19, %r20 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd6 + 0 ], { %r21, %r22, %r23, %r24 };
	// end inline asm
	.loc	1 27 4                          // triton_add.py:27:4
	ret;
$L__tmp1:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/home/miner/code/triton/triton_add.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 0                                   // DW_CHILDREN_no
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 59                                 // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x34 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 116                                 // DW_AT_name
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 95
.b8 97
.b8 100
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 104
.b8 111
.b8 109
.b8 101
.b8 47
.b8 109
.b8 105
.b8 110
.b8 101
.b8 114
.b8 47
.b8 99
.b8 111
.b8 100
.b8 101
.b8 47
.b8 116
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
	}
	.section	.debug_macinfo	{	}
```

## Step 7: CUBIN (Cuda Binary)

CUBIN，即 SASS(Streaming Array Specific Shader) GPU 原生指令集代码

这是最终转换，将虚拟汇编代码转换为可以直接在 GPU 上运行的 CUBIN。

主要操作为：
1. 将 PTX 指令集汇编为特定 GPU 架构的机器吗
2. 指令优化与调度：根据硬件信息进行特定优化，如指令集并行、线程束调度、延迟隐藏等。
3. 寄存器分配与溢出处理：虚拟寄存器映射到物理寄存器，物理寄存器不足时映射局部内存。
4. 生成 CUBIN 元信息：划分不同段，分为 SASS, 常量内存数据，kernel 元数据等。

```c
[二进制数据]
```

## 参考链接

- [Triton 中文站](https://triton.hyper.ai/docs/getting-started/tutorials/vector-addition)