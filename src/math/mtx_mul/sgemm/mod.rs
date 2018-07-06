// Matrix multiplication based on https://cnugteren.github.io/tutorial/
// NOTE: all matrices are assumed to be multiples of 32 to fit into OpenCL work-groups

#![allow(non_snake_case)]

#[cfg(test)]
mod test;

use layers::Coeff;
use ocl;
use ocl::{flags, Buffer, Context, Device, Kernel, OclPrm, Platform, Program, Queue, SpatialDims};

// Column-major into row-major would be to switch A and B and N and M
/// Naive matrix multiplication
///
/// Matrices are all column-major
///
/// * C := A * B
/// * A is K by M
/// * B is N by K
/// * C is M by N
pub fn mtx_mul_1_naive_host(M: usize, N: usize, K: usize, A: &[f32], B: &[f32], C: &mut [f32]) {
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

pub fn mtx_mul_1_naive_cl(M: usize, N: usize, K: usize, A: &[f32], B: &[f32], C: &mut [f32])
where
    f32: OclPrm,
{
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

        // This is 32 in the original example; that would produce gws of 1024, but the maximum of
        // my desktop GPU is 256 (16x16).
        let TS: usize = 16;
        println!("lws: {}x{}", TS, TS);
        println!("gws: {}x{}", M, N);
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
    }
}
