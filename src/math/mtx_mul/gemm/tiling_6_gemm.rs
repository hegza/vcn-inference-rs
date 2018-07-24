use super::*;

// TODO: Implement a version for pretransposed b-matrix. Measure performance. This, because in a neural net, the weights matrices (b's) can be consistently pre-transposed.
pub struct Tiling6GemmKernel {
    transpose_kernel: Kernel,
    main_kernel: Kernel,
    queue: Queue,
    // Buffer A
    input_a: Buffer<f32>,
    // Untransposed buffer B
    input_b: Buffer<f32>,
    use_host_ptr: bool,
}

impl OclGemm<Tiling6GemmKernel> for Tiling6GemmKernel {
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
        debug_assert_eq!(c.len(), m * n);

        // If Device uses RAM, use_host_ptr and mapping via address translation may be faster
        let use_host_ptr = device.contains(DeviceType::CPU);

        // The tile-size in dimension m
        const TSM: usize = 32;
        // The tile-size in dimension n
        const TSN: usize = 32;
        // The tile-size in dimension k
        const TSK: usize = 16;
        // The amount of work-per-work-item in dimension m
        const WPTM: usize = 8;
        // The amount of work-per-work-item in dimension n
        const WPTN: usize = 8;
        // Dimensions for local memory optimization
        const TRANSPOSEX: usize = 16;
        const TRANSPOSEY: usize = 16;

        let src_transpose = String::from_utf8_lossy(include_bytes!("cl/transpose.cl"));
        let src_mtx_mul = String::from_utf8_lossy(include_bytes!("cl/6_register_tiling.cl"));

        let (queue, program, _context) = cl_util::init_from_sources::<f32>(
            &[&src_transpose, &src_mtx_mul],
            &[
                "-I./src/math/mtx_mul/gemm/cl",
                &format!("-D TSM={}", TSM),
                &format!("-D TSN={}", TSN),
                &format!("-D TSK={}", TSK),
                &format!("-D WPTM={}", WPTM),
                &format!("-D WPTN={}", WPTN),
                &format!("-D TRANSPOSEX={}", TRANSPOSEX),
                &format!("-D TRANSPOSEY={}", TRANSPOSEY),
            ],
            Some(device),
        );

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
            let lws = SpatialDims::Two(TSM / WPTM, TSN / WPTN);
            let gws = SpatialDims::Two(m / WPTM, n / WPTN);

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
                    .arg(&b_untransposed_buf)
                    .arg(&b_buf)
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
                    .arg(&a_buf)
                    .arg(&b_buf)
                    .arg(&c_buf)
                    .build()
                    .unwrap(),
            )
        };
        queue.finish().unwrap();

        Tiling6GemmKernel {
            transpose_kernel: transpose_kernel,
            main_kernel: main_kernel,
            queue: queue,
            input_a: a_buf,
            input_b: b_untransposed_buf,
            use_host_ptr,
        }
    }

    // TODO: comment is misleading; is this how it should be implemented though?
    /// Sets input and output buffers from the slices a, b and c. Essentially free on CPU and
    /// devices with shared GPU/CPU memory.
    ///
    /// a and b are column-major (b will be transposed automatically into row-major by the algorithm)
    fn set_buffers_from_slices(&self, a: &[f32], b: &[f32]) {
        match self.use_host_ptr {
            true => unsafe {
                cl_util::map_to_buf(&self.input_a, a).unwrap();
                cl_util::map_to_buf(&self.input_b, b).unwrap();
            },
            false => {
                self.input_a.write(a).enq().unwrap();
                self.input_b.write(b).enq().unwrap();
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
