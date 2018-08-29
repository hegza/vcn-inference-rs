use super::*;

pub struct Gemm4Kernel {
    kernel: Kernel,
    queue: Queue,
    a_buf: Buffer<f32>,
    b_buf: Buffer<f32>,
    use_host_ptr: bool,
}

impl OclGemm<Gemm4Kernel> for Gemm4Kernel {
    fn uninitialized(
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        device: DeviceType,
    ) -> Gemm4Kernel {
        // Make sure enough space is reserved for the output buffer
        debug_assert_eq!(out.len(), m * n);

        // If Device uses RAM, use_host_ptr and mapping via address translation may be faster
        let use_host_ptr = device == DeviceType::CPU;

        // The width of the OpenCL vector-type (in number of floats)
        const WIDTH: usize = 4;
        // The square-root of the 2D tile-size (== work-group dims)
        const TS: usize = 32;

        let src = String::from_utf8_lossy(include_bytes!("cl/4_wider_data_types.cl"));

        let (queue, program, _context) = cl_util::init_from_sources::<f32>(
            &[&src],
            &[
                "-I./src/math/gemm/cl",
                &format!("-D WIDTH={}", WIDTH as i32),
                &format!("-D TS={}", TS as i32),
            ],
            Some(device),
        );

        let (a_buf, b_buf) = (
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

        let kernel = {
            let lws = SpatialDims::Two(TS / WIDTH, TS);
            let gws = SpatialDims::Two(m / WIDTH, n);
            Kernel::builder()
                .program(&program)
                .name("myGEMM4")
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
                .unwrap()
        };
        queue.finish().unwrap();

        Gemm4Kernel {
            kernel,
            queue,
            a_buf,
            b_buf,
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
    ) -> Gemm4Kernel {
        debug_assert_eq!(a.len(), k * m);
        debug_assert_eq!(b.len(), n * k);

        let mut kernel = Gemm4Kernel::uninitialized(m, n, k, c, device);
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
                    kernel.b_buf = Buffer::<f32>::builder()
                        .queue(queue.clone())
                        .flags(flags::MEM_READ_ONLY)
                        .len(n * k)
                        .use_host_slice(&b)
                        .build()
                        .unwrap();
                }
                kernel.kernel.set_arg("a", &kernel.a_buf).unwrap();
                kernel.kernel.set_arg("b", &kernel.b_buf).unwrap();
            } else {
                kernel.set_buffers_from_slices(&a, &b);
            }
        }

        kernel
    }

    fn set_buffers_from_slices(&self, a: &[f32], b: &[f32]) {
        match self.use_host_ptr {
            true => unsafe {
                cl_util::map_to_buf(&self.a_buf, a).unwrap();
                cl_util::map_to_buf(&self.b_buf, b).unwrap();
            },
            false => {
                self.a_buf.write(a).enq().unwrap();
                self.b_buf.write(b).enq().unwrap();
            }
        }
    }

    fn calculate(&self) {
        unsafe {
            self.kernel.cmd().queue(&self.queue).enq().unwrap();
        }
    }

    fn queue(&self) -> &Queue {
        &self.queue
    }
}
