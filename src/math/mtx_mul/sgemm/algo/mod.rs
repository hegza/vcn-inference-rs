#[cfg(test)]
mod test;

use ocl;
use ocl::{flags, Buffer, Context, Device, Kernel, OclPrm, Platform, Program, Queue, SpatialDims};

/// Naive matrix multiplication on the CPU
pub fn gemm_naive(M: usize, N: usize, K: usize, A: &[f32], B: &[f32], C: &mut [f32]) {
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                acc += A[k * M + m] * B[n * K + k]
            }
            C[n * M + m] = acc;
        }
    }
}

fn mtx_mul_1_naive_cl(M: usize, N: usize, K: usize, A: &[f32], B: &[f32], C: &mut [f32]) {
    let src = String::from_utf8_lossy(include_bytes!("cl/1_naive.cl"));

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()
        .unwrap();
    let program = Program::builder()
        .devices(device)
        .src(src)
        .build(&context)
        .unwrap();
    let queue = Queue::new(&context, device, None).unwrap();

    unsafe {
        let A_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(A)
            .flags(flags::MEM_READ_ONLY)
            .len(A.len())
            .build()
            .unwrap();
        let B_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(B)
            .flags(flags::MEM_READ_ONLY)
            .len(B.len())
            .build()
            .unwrap();
        let C_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(C)
            .flags(flags::MEM_WRITE_ONLY)
            .len(C.len())
            .build()
            .unwrap();

        // This is 32 in the original example; that would produce gws of 1024, but the maximum of
        // my desktop GPU is 256 (16x16).
        let TS: usize = 16;
        println!("SGEMM1 lws: {}x{}", TS, TS);
        println!("SGEMM1 gws: {}x{}", M, N);
        let kernel = Kernel::builder()
            .program(&program)
            .name("myGEMM1")
            .queue(queue.clone())
            .local_work_size(SpatialDims::Two(TS, TS))
            .global_work_size(SpatialDims::Two(M, N))
            .arg(M as i32)
            .arg(N as i32)
            .arg(K as i32)
            .arg(&A_buf)
            .arg(&B_buf)
            .arg(&C_buf)
            .build()
            .unwrap();

        kernel.cmd().queue(&queue).enq().unwrap();

        queue.finish().unwrap();
    }
}

// NOTE: Code almost identical to above
fn mtx_mul_4_vector_data_types_cl(
    M: usize,
    N: usize,
    K: usize,
    A: &[f32],
    B: &[f32],
    C: &mut [f32],
) {
    // The width of the OpenCL vector-type (in number of floats)
    const WIDTH: usize = 4;
    // The square-root of the 2D tile-size (== work-group dims)
    const TS: usize = 32;

    let src = String::from_utf8_lossy(include_bytes!("cl/4_wider_data_types.cl"));

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()
        .unwrap();
    let program = Program::builder()
        .devices(device)
        .cmplr_opt("-I./src/math/mtx_mul/sgemm/algo/cl")
        .cmplr_def("WIDTH", WIDTH as i32)
        .cmplr_def("TS", TS as i32)
        .src(src)
        .build(&context)
        .unwrap();
    let queue = Queue::new(&context, device, None).unwrap();

    unsafe {
        let A_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(A)
            .len(A.len())
            .build()
            .unwrap();
        let B_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(B)
            .len(B.len())
            .build()
            .unwrap();
        let C_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(C)
            .len(C.len())
            .build()
            .unwrap();

        let lws = SpatialDims::Two(TS / WIDTH, TS);
        let gws = SpatialDims::Two(M / WIDTH, N);
        println!("SGEMM4 lws: {:?}", lws);
        println!("SGEMM4 gws: {:?}", gws);
        let kernel = Kernel::builder()
            .program(&program)
            .name("myGEMM4")
            .queue(queue.clone())
            .local_work_size(lws)
            .global_work_size(gws)
            .arg(M as i32)
            .arg(N as i32)
            .arg(K as i32)
            .arg(&A_buf)
            .arg(&B_buf)
            .arg(&C_buf)
            .build()
            .unwrap();

        kernel.cmd().queue(&queue).enq().unwrap();

        queue.finish().unwrap();
    }
}

// NOTE: Code almost identical to above
fn mtx_mul_5_transpose_cl(M: usize, N: usize, K: usize, A: &[f32], B: &[f32], C: &mut [f32]) {
    // The square-root of the 2D tile-size (== work-group dims)
    const TS: usize = 32;
    // The amount of work-per-thread, i.e. the thread-coarsening factor
    const WPT: usize = 8;
    // The tile-size in dimension K. Determines number of loads per work-item.
    const TSDK: usize = 16;
    // Dimensions for local memory optimization
    const TRANSPOSEX: usize = 16;
    const TRANSPOSEY: usize = 16;

    let src_transpose = String::from_utf8_lossy(include_bytes!("cl/transpose.cl"));
    let src_mtx_mul = String::from_utf8_lossy(include_bytes!("cl/5_transpose.cl"));

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()
        .unwrap();
    let program = Program::builder()
        .devices(device)
        .cmplr_opt("-I./src/math/mtx_mul/sgemm/algo/cl")
        .cmplr_def("TS", TS as i32)
        .cmplr_def("WPT", WPT as i32)
        .cmplr_def("TSDK", TSDK as i32)
        .cmplr_def("TRANSPOSEX", TRANSPOSEX as i32)
        .cmplr_def("TRANSPOSEY", TRANSPOSEY as i32)
        .src(src_transpose)
        .src(src_mtx_mul)
        .build(&context)
        .unwrap();
    let queue = Queue::new(&context, device, None).unwrap();

    unsafe {
        let A_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(A)
            .flags(flags::MEM_READ_ONLY)
            .len(A.len())
            .build()
            .unwrap();
        let B_untransposed_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(B)
            .flags(flags::MEM_READ_ONLY)
            .len(B.len())
            .build()
            .unwrap();
        let B_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(B.len())
            .build()
            .unwrap();
        let C_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(C)
            .flags(flags::MEM_WRITE_ONLY)
            .len(C.len())
            .build()
            .unwrap();

        // Build kernel for transposing B
        let transpose_b_kernel = Kernel::builder()
            .program(&program)
            .name("transpose")
            .queue(queue.clone())
            .local_work_size(SpatialDims::Two(TRANSPOSEX, TRANSPOSEY))
            .global_work_size(SpatialDims::Two(K, N))
            .arg(K as i32)
            .arg(N as i32)
            .arg(&B_untransposed_buf)
            .arg(&B_buf)
            .build()
            .unwrap();

        let lws = SpatialDims::Two(TS, TS / WPT);
        let gws = SpatialDims::Two(M, N / WPT);
        println!("SGEMM5 lws: {:?}", lws);
        println!("SGEMM5 gws: {:?}", gws);

        // Build kernel for mtx_mul
        let kernel = Kernel::builder()
            .program(&program)
            .name("myGEMM5")
            .queue(queue.clone())
            .local_work_size(lws)
            .global_work_size(gws)
            .arg(M as i32)
            .arg(N as i32)
            .arg(K as i32)
            .arg(&A_buf)
            .arg(&B_buf)
            .arg(&C_buf)
            .build()
            .unwrap();

        transpose_b_kernel.cmd().queue(&queue).enq().unwrap();
        kernel.cmd().queue(&queue).enq().unwrap();

        queue.finish().unwrap();
    }
}

// NOTE: Code almost identical to above
fn mtx_mul_6_register_tiling_cl(M: usize, N: usize, K: usize, A: &[f32], B: &[f32], C: &mut [f32]) {
    // The tile-size in dimension M
    const TSM: usize = 32;
    // The tile-size in dimension N
    const TSN: usize = 32;
    // The tile-size in dimension K
    const TSK: usize = 16;
    // The amount of work-per-work-item in dimension M
    const WPTM: usize = 8;
    // The amount of work-per-work-item in dimension N
    const WPTN: usize = 8;
    // Dimensions for local memory optimization
    const TRANSPOSEX: usize = 16;
    const TRANSPOSEY: usize = 16;

    let src_transpose = String::from_utf8_lossy(include_bytes!("cl/transpose.cl"));
    let src_mtx_mul = String::from_utf8_lossy(include_bytes!("cl/6_register_tiling.cl"));

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()
        .unwrap();
    let program = Program::builder()
        .devices(device)
        .cmplr_opt("-I./src/math/mtx_mul/sgemm/algo/cl")
        .cmplr_def("TSM", TSM as i32)
        .cmplr_def("TSN", TSN as i32)
        .cmplr_def("TSK", TSK as i32)
        .cmplr_def("WPTM", WPTM as i32)
        .cmplr_def("WPTN", WPTN as i32)
        .cmplr_def("TRANSPOSEX", TRANSPOSEX as i32)
        .cmplr_def("TRANSPOSEY", TRANSPOSEY as i32)
        .src(src_transpose)
        .src(src_mtx_mul)
        .build(&context)
        .unwrap();
    let queue = Queue::new(&context, device, None).unwrap();

    unsafe {
        let A_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(A)
            .flags(flags::MEM_READ_ONLY)
            .len(A.len())
            .build()
            .unwrap();
        let B_untransposed_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(B)
            .flags(flags::MEM_READ_ONLY)
            .len(B.len())
            .build()
            .unwrap();
        let B_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(B.len())
            .build()
            .unwrap();
        let C_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .use_host_slice(C)
            .flags(flags::MEM_WRITE_ONLY)
            .len(C.len())
            .build()
            .unwrap();

        // Build kernel for transposing B
        let transpose_b_kernel = Kernel::builder()
            .program(&program)
            .name("transpose")
            .queue(queue.clone())
            .local_work_size(SpatialDims::Two(TRANSPOSEX, TRANSPOSEY))
            .global_work_size(SpatialDims::Two(K, N))
            .arg(K as i32)
            .arg(N as i32)
            .arg(&B_untransposed_buf)
            .arg(&B_buf)
            .build()
            .unwrap();

        let lws = SpatialDims::Two(TSM / WPTM, TSN / WPTN);
        let gws = SpatialDims::Two(M / WPTM, N / WPTN);
        println!("SGEMM6 lws: {:?}", lws);
        println!("SGEMM6 gws: {:?}", gws);

        // Build kernel for mtx_mul
        let kernel = Kernel::builder()
            .program(&program)
            .name("myGEMM6")
            .queue(queue.clone())
            .local_work_size(lws)
            .global_work_size(gws)
            .arg(M as i32)
            .arg(N as i32)
            .arg(K as i32)
            .arg(&A_buf)
            .arg(&B_buf)
            .arg(&C_buf)
            .build()
            .unwrap();

        transpose_b_kernel.cmd().queue(&queue).enq().unwrap();
        kernel.cmd().queue(&queue).enq().unwrap();

        queue.finish().unwrap();
    }
}
