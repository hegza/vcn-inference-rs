use super::*;

pub struct Transpose5GemmKernel {
    transpose_kernel: Kernel,
    main_kernel: Kernel,
    queue: Queue,
    a_buf: Buffer<f32>,
    b_untransposed_buf: Buffer<f32>,
    use_host_ptr: bool,
}

impl OclGemm<Transpose5GemmKernel> for Transpose5GemmKernel {
    fn uninitialized(
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        device: DeviceType,
    ) -> Transpose5GemmKernel {
        // Make sure enough space is reserved for the output buffer
        debug_assert_eq!(out.len(), m * n);

        // If Device uses RAM, use_host_ptr and mapping via address translation may be faster
        let use_host_ptr = device == DeviceType::CPU;

        // The square-root of the 2D tile-size (== work-group dims)
        const TS: usize = 32;
        // The amount of work-per-work-item, i.e. the work-item-coarsening factor
        const WPT: usize = 8;
        // The tile-size in dimension k. Determines number of loads per work-item.
        const TSDK: usize = 16;
        // Dimensions for local memory optimization
        const TRANSPOSEX: usize = 16;
        const TRANSPOSEY: usize = 16;

        let src_transpose = String::from_utf8_lossy(include_bytes!("cl/transpose.cl"));
        let src_mtx_mul = String::from_utf8_lossy(include_bytes!("cl/5_transpose.cl"));

        let (queue, program, _context) = cl_util::init_from_sources::<f32>(
            &[&src_transpose, &src_mtx_mul],
            &[
                "-I./src/math/mtx_mul/gemm/cl",
                &format!("-D TS={}", TS),
                &format!("-D WPT={}", WPT),
                &format!("-D TSDK={}", TSDK),
                &format!("-D TRANSPOSEX={}", TRANSPOSEX),
                &format!("-D TRANSPOSEY={}", TRANSPOSEY),
            ],
            Some(device),
        );

        let (a_buf, b_untransposed_buf, b_buf) = (
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY)
                .len(k * m)
                .build()
                .unwrap(),
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY)
                .len(n * k)
                .build()
                .unwrap(),
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .len(n * k)
                .build()
                .unwrap(),
        );
        let c_buf = unsafe {
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_WRITE_ONLY)
                .len(m * n)
                .use_host_slice(&out)
                .build()
                .unwrap()
        };

        let (transpose_kernel, main_kernel) = {
            let lws = SpatialDims::Two(TS, TS / WPT);
            let gws = SpatialDims::Two(m, n / WPT);

            // Build kernel for transposing b
            (
                Kernel::builder()
                    .program(&program)
                    .name("transpose")
                    .queue(queue.clone())
                    .local_work_size(SpatialDims::Two(TRANSPOSEX, TRANSPOSEY))
                    .global_work_size(SpatialDims::Two(k, n))
                    .arg(k as i32)
                    .arg(n as i32)
                    .arg_named("b_untransposed", &b_untransposed_buf)
                    .arg_named("b", &b_buf)
                    .build()
                    .unwrap(),
                // Build kernel for mtx_mul
                Kernel::builder()
                    .program(&program)
                    .name("myGEMM5")
                    .queue(queue.clone())
                    .local_work_size(lws)
                    .global_work_size(gws)
                    .arg(m as i32)
                    .arg(n as i32)
                    .arg(k as i32)
                    .arg_named("a", &a_buf)
                    .arg_named("b", &b_buf)
                    .arg_named("c", &c_buf)
                    .build()
                    .unwrap(),
            )
        };
        queue.finish().unwrap();

        Transpose5GemmKernel {
            transpose_kernel,
            main_kernel,
            queue,
            a_buf,
            b_untransposed_buf,
            use_host_ptr,
        }
    }
    fn from_slices(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        device: DeviceType,
    ) -> Transpose5GemmKernel {
        debug_assert_eq!(a.len(), k * m);
        debug_assert_eq!(b.len(), n * k);

        let mut kernel = Transpose5GemmKernel::uninitialized(m, n, k, c, device);
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
                    kernel.b_untransposed_buf = Buffer::<f32>::builder()
                        .queue(queue.clone())
                        .flags(flags::MEM_READ_ONLY)
                        .len(n * k)
                        .use_host_slice(&b)
                        .build()
                        .unwrap();
                }
                kernel.main_kernel.set_arg("a", &kernel.a_buf).unwrap();
                kernel
                    .transpose_kernel
                    .set_arg("b_untransposed", &kernel.b_untransposed_buf)
                    .unwrap();
            } else {
                kernel.set_buffers_from_slices(&a, &b);
            }
        }

        kernel
    }

    /// a and b are column-major (b will be transposed automatically into row-major by the algorithm)
    fn set_buffers_from_slices(&self, a: &[f32], b: &[f32]) {
        match self.use_host_ptr {
            true => unsafe {
                cl_util::map_to_buf(&self.a_buf, a).unwrap();
                cl_util::map_to_buf(&self.b_untransposed_buf, b).unwrap();
            },
            false => {
                self.a_buf.write(a).enq().unwrap();
                self.b_untransposed_buf.write(b).enq().unwrap();
            }
        }
    }

    fn calculate(&self) {
        unsafe {
            self.transpose_kernel
                .cmd()
                .queue(&self.queue)
                .enq()
                .unwrap();
            self.main_kernel.cmd().queue(&self.queue).enq().unwrap();
        }
    }

    fn queue(&self) -> &Queue {
        &self.queue
    }
}
