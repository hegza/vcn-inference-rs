use super::*;

// TODO: Implement a version for pretransposed b-matrix. Measure performance. This, because in a neural net, the weights matrices (b's) can be consistently pre-transposed.
/// Minimum matrix size: 32x32
pub struct Tiling6GemmKernel {
    transpose_kernel: Kernel,
    main_kernel: Kernel,
    queue: Queue,
    // Buffer A
    a_buf: Buffer<f32>,
    // Untransposed buffer B
    b_untransposed_buf: Buffer<f32>,
    use_host_ptr: bool,
}

impl OclGemm<Tiling6GemmKernel> for Tiling6GemmKernel {
    fn uninitialized(
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        device: DeviceType,
    ) -> Tiling6GemmKernel {
        // Make sure enough space is reserved for the output buffer
        debug_assert_eq!(out.len(), m * n);

        // If Device uses RAM, use_host_ptr and mapping via address translation may be faster
        let use_host_ptr = device.contains(DeviceType::CPU);

        // The tile-size in dimension m
        let tsm: usize = 32;
        // The tile-size in dimension n
        let tsn: usize = 32;
        // The tile-size in dimension k
        let tsk: usize = 16;
        // The amount of work-per-work-item in dimension m
        let wptm: usize = 8;
        // The amount of work-per-work-item in dimension n
        let wptn: usize = 8;
        // Dimensions for local memory optimization
        const TRANSPOSEX: usize = 16;
        const TRANSPOSEY: usize = 16;

        let src_transpose = String::from_utf8_lossy(include_bytes!("cl/transpose.cl"));
        let src_mtx_mul = String::from_utf8_lossy(include_bytes!("cl/6_register_tiling.cl"));

        let (queue, program, _context) = cl_util::init_from_sources::<f32>(
            &[&src_transpose, &src_mtx_mul],
            &[
                "-I./src/math/mtx_mul/gemm/cl",
                &format!("-D TSM={}", tsm),
                &format!("-D TSN={}", tsn),
                &format!("-D TSK={}", tsk),
                &format!("-D WPTM={}", wptm),
                &format!("-D WPTN={}", wptn),
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
            let lws = SpatialDims::Two(tsn / wptm, tsn / wptn);
            let gws = SpatialDims::Two(m / wptm, n / wptn);

            // Build kernel for transposing B
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
                    .name("myGEMM6")
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

        Tiling6GemmKernel {
            transpose_kernel: transpose_kernel,
            main_kernel: main_kernel,
            queue: queue,
            a_buf: a_buf,
            b_untransposed_buf: b_untransposed_buf,
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
    ) -> Tiling6GemmKernel {
        debug_assert_eq!(a.len(), k * m);
        debug_assert_eq!(b.len(), n * k);

        let mut kernel = Tiling6GemmKernel::uninitialized(m, n, k, c, device);
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

    // TODO: comment is misleading; is this how it should be implemented though?
    /// Sets input and output buffers from the slices a, b and c. Essentially free on CPU and
    /// devices with shared GPU/CPU memory.
    ///
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

    /*
    fn set_buffers(&self, a: GemmInput, b_untransposed: GemmInput) {
        unimplemented!();
        match a {
            GemmInput::Slice(arr) => {
                self.a_buf.write(arr).enq().unwrap();
            },
            GemmInput::OclBuffer(buffer) => {

            }
        }
    }
    */

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
