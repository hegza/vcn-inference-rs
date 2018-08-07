mod params;
mod with_transpose;

use self::params::*;
pub use self::with_transpose::*;
use super::*;

/// Cnugteren's gemm version 10. A is column-major and B is row-major. This is the more efficient
/// implementation.
pub struct Gemm10Kernel {
    kernel: Kernel,
    pad_a_kernel: Kernel,
    pad_b_kernel: Kernel,
    unpad_c_kernel: Kernel,
    queue: Queue,
    a_buf: Buffer<f32>,
    b_transposed_buf: Buffer<f32>,
    use_host_ptr: bool,
}

impl OclGemm<Gemm10Kernel> for Gemm10Kernel {
    fn uninitialized(
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        device: DeviceType,
    ) -> Gemm10Kernel {
        // Make sure enough space is reserved for the output buffer
        debug_assert_eq!(out.len(), m * n);

        // If Device uses RAM, use_host_ptr and mapping via address translation may be faster
        let use_host_ptr = device == DeviceType::CPU;

        let src_padding = String::from_utf8_lossy(include_bytes!("../cl/padding.cl"));
        let src_mtx_mul = String::from_utf8_lossy(include_bytes!("../cl/10_incomplete_tiles.cl"));

        let params = Gemm10CompileParameters::choose(m, n, k, device);
        let mut params_list: Vec<String> = params.clone().into();
        params_list.push("-I./src/math/mtx_mul/gemm/cl".to_owned());

        let (queue, program, _context) = cl_util::init_from_sources::<f32>(
            &[&src_mtx_mul, &src_padding],
            &params_list.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            Some(device),
        );

        let (a_buf, a_padded_buf) = (
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY)
                .len(k * m)
                .build()
                .unwrap(),
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .len(params.padded_k * params.padded_m)
                .build()
                .unwrap(),
        );
        let (b_transposed_buf, b_padded_buf) = (
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY)
                .len(n * k)
                .build()
                .unwrap(),
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .len(params.padded_n * params.padded_k)
                .build()
                .unwrap(),
        );
        let (c_padded_buf, c_buf) = unsafe {
            (
                Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(flags::MEM_READ_WRITE)
                    .len(params.padded_m * params.padded_n)
                    .build()
                    .unwrap(),
                Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(flags::MEM_WRITE_ONLY)
                    .len(m * n)
                    .use_host_slice(&out)
                    .build()
                    .unwrap(),
            )
        };

        let pad_a_kernel = Kernel::builder()
            .program(&program)
            .name("paddingAddZeroes")
            .queue(queue.clone())
            .local_work_size(params.pad_lws)
            .global_work_size(params.pad_a_gws)
            .arg(m as i32)
            .arg(k as i32)
            .arg_named("a", &a_buf)
            .arg(params.padded_m as i32)
            .arg(params.padded_k as i32)
            .arg_named("a_padded", &a_padded_buf)
            .build()
            .unwrap();
        let pad_b_kernel = Kernel::builder()
            .program(&program)
            .name("paddingAddZeroes")
            .queue(queue.clone())
            .local_work_size(params.pad_lws)
            .global_work_size(params.pad_b_gws)
            .arg(n as i32)
            .arg(k as i32)
            .arg_named("b_transposed", &b_transposed_buf)
            .arg(params.padded_n as i32)
            .arg(params.padded_k as i32)
            .arg_named("b_transposed_padded", &b_padded_buf)
            .build()
            .unwrap();
        let gemm10 =
                // Build kernel for mtx_mul
                Kernel::builder()
                    .program(&program)
                    .name("myGEMM10")
                    .queue(queue.clone())
                    .local_work_size(params.lws)
                    .global_work_size(params.gws)
                    .arg(m as i32)
                    .arg(n as i32)
                    .arg(k as i32)
                    .arg_named("a_padded", &a_padded_buf)
                    .arg_named("b_transposed_padded", &b_padded_buf)
                    .arg_named("c_padded", &c_padded_buf)
                    .build()
                    .unwrap();
        let unpad_c_kernel = Kernel::builder()
            .program(&program)
            .name("paddingRemoveZeroes")
            .queue(queue.clone())
            .local_work_size(params.pad_lws)
            .global_work_size(params.unpad_c_gws)
            .arg(params.padded_m as i32)
            .arg(params.padded_n as i32)
            .arg_named("c_padded", &c_padded_buf)
            .arg(m as i32)
            .arg(n as i32)
            .arg_named("c", &c_buf)
            .build()
            .unwrap();
        queue.finish().unwrap();

        Gemm10Kernel {
            kernel: gemm10,
            pad_a_kernel,
            pad_b_kernel,
            unpad_c_kernel,
            queue: queue,
            a_buf: a_buf,
            b_transposed_buf,
            use_host_ptr,
        }
    }

    /// a is column-major and b_transposed is row-major (or transposed column-major)
    fn from_slices(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b_transposed: &[f32],
        c: &mut [f32],
        device: DeviceType,
    ) -> Gemm10Kernel {
        debug_assert_eq!(a.len(), k * m);
        debug_assert_eq!(b_transposed.len(), k * n);

        let mut kernel = Gemm10Kernel::uninitialized(m, n, k, c, device);
        {
            let queue = &kernel.queue;

            // Re-create buffers as use-host-ptr if necessary
            if kernel.use_host_ptr {
                unsafe {
                    kernel.a_buf = Buffer::<f32>::builder()
                        .queue(queue.clone())
                        .flags(flags::MEM_READ_ONLY)
                        .len(k * m)
                        .use_host_slice(&a)
                        .build()
                        .unwrap();
                    kernel.b_transposed_buf = Buffer::<f32>::builder()
                        .queue(queue.clone())
                        .flags(flags::MEM_READ_ONLY)
                        .len(k * n)
                        .use_host_slice(&b_transposed)
                        .build()
                        .unwrap();
                }
                kernel.kernel.set_arg("a", &kernel.a_buf).unwrap();
                kernel
                    .kernel
                    .set_arg("b_transposed", &kernel.b_transposed_buf)
                    .unwrap();
            } else {
                kernel.set_buffers_from_slices(&a, &b_transposed);
            }
        }

        kernel
    }

    /// a is column-major and b_transposed is row-major (or transposed column-major)
    fn set_buffers_from_slices(&self, a: &[f32], b_transposed: &[f32]) {
        match self.use_host_ptr {
            true => unsafe {
                cl_util::map_to_buf(&self.a_buf, a).unwrap();
                cl_util::map_to_buf(&self.b_transposed_buf, b_transposed).unwrap();
            },
            false => {
                self.a_buf.write(a).enq().unwrap();
                self.b_transposed_buf.write(b_transposed).enq().unwrap();
            }
        }
    }

    fn calculate(&self) {
        unsafe {
            self.pad_a_kernel.cmd().queue(&self.queue).enq().unwrap();
            self.pad_b_kernel.cmd().queue(&self.queue).enq().unwrap();
            self.kernel.cmd().queue(&self.queue).enq().unwrap();
            self.unpad_c_kernel.cmd().queue(&self.queue).enq().unwrap();
        }
    }

    fn queue(&self) -> &Queue {
        &self.queue
    }
}
