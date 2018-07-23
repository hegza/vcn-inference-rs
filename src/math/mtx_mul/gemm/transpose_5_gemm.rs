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
        debug_assert_eq!(c.len(), m * n);

        // Share inputs within host memory on CPU's
        let use_host_ptr = device.contains(DeviceType::CPU);

        // The square-root of the 2D tile-size (== work-group dims)
        const TS: usize = 32;
        // The amount of work-per-thread, i.e. the thread-coarsening factor
        const WPT: usize = 8;
        // The tile-size in dimension k. Determines number of loads per work-item.
        const TSDK: usize = 16;
        // Dimensions for local memory optimization
        const TRANSPOSEX: usize = 16;
        const TRANSPOSEY: usize = 16;

        let src_transpose = String::from_utf8_lossy(include_bytes!("cl/transpose.cl"));
        let src_mtx_mul = String::from_utf8_lossy(include_bytes!("cl/5_transpose.cl"));

        let platform = Platform::default();
        let device = *Device::list(platform, Some(flags::DeviceType::GPU))
            .unwrap()
            .first()
            .unwrap();
        let context = Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
            .unwrap();
        let program = Program::builder()
            .devices(device)
            .cmplr_opt("-I./src/math/mtx_mul/gemm/cl")
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

        let (mut a_buf, mut b_untransposed_buf, b_buf, c_buf) = unsafe {
            (
                Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(flags::MEM_READ_ONLY)
                    .len(a.len()),
                Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(flags::MEM_READ_ONLY)
                    .len(b.len()),
                Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(flags::MEM_READ_WRITE)
                    .len(b.len()),
                Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .use_host_slice(c)
                    .flags(flags::MEM_WRITE_ONLY)
                    .len(c.len()),
            )
        };
        if use_host_ptr {
            a_buf = unsafe { a_buf.use_host_slice(&a) };
            b_untransposed_buf = unsafe { b_untransposed_buf.use_host_slice(&b) };
        } else {
            a_buf = a_buf.copy_host_slice(&a);
            b_untransposed_buf = b_untransposed_buf.copy_host_slice(&b);
        }
        let (a_buf, b_untransposed_buf, b_buf, c_buf) = (
            a_buf.build().unwrap(),
            b_untransposed_buf.build().unwrap(),
            b_buf.build().unwrap(),
            c_buf.build().unwrap(),
        );

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
                    .arg(&b_untransposed_buf)
                    .arg(&b_buf)
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
                    .arg(&a_buf)
                    .arg(&b_buf)
                    .arg(&c_buf)
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

    /// a and b are column-major (b will be transposed automatically into row-major by the algorithm)
    fn set_buffers_from_slices(&self, a: &[f32], b: &[f32]) {
        debug_assert!(
            !self.use_host_ptr,
            "memory region for buffers has already been set via ocl::BufferBuilder::use_host_ptr"
        );
        self.a_buf.write(a).enq().unwrap();
        self.b_untransposed_buf.write(b).enq().unwrap();
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
