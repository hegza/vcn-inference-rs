use super::*;

pub struct Naive1GemmKernel {
    kernel: Kernel,
    queue: Queue,
    use_host_ptr: bool,
    a_buf: Buffer<f32>,
    b_buf: Buffer<f32>,
}

impl OclGemm<Naive1GemmKernel> for Naive1GemmKernel {
    fn uninitialized(
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        device: DeviceType,
    ) -> Naive1GemmKernel {
        // Make sure enough space is reserved for the output buffer
        debug_assert_eq!(out.len(), m * n);

        // If Device uses RAM, use_host_ptr and mapping via address translation may be faster
        let use_host_ptr = device == DeviceType::CPU;

        let src = String::from_utf8_lossy(include_bytes!("cl/1_naive.cl"));

        let (queue, program, _context) =
            cl_util::init_from_sources::<f32>(&[&src], &[], Some(device));

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

        // Optimal tile-size is as close to the preferred maximum work-group-size while still
        // fitting into the max work group size on GPU and 1 on CPU because no autovectorization is
        // possible for this kernel. cnugteren uses hard-coded 32x32.
        let ts = if device == DeviceType::CPU {
            1
        } else {
            let dev_max_lws = cl_util::select_device(Some(device)).max_wg_size().unwrap();
            (dev_max_lws as f64).sqrt() as usize
        };
        let lws = SpatialDims::Two(ts, ts);

        let gws = SpatialDims::Two(m, n);
        let kernel = Kernel::builder()
            .program(&program)
            .name("myGEMM1")
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
            .unwrap();
        queue.finish().unwrap();

        Naive1GemmKernel {
            kernel,
            queue,
            use_host_ptr,
            a_buf,
            b_buf,
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
    ) -> Naive1GemmKernel {
        debug_assert_eq!(a.len(), k * m);
        debug_assert_eq!(b.len(), n * k);

        let mut kernel = Naive1GemmKernel::uninitialized(m, n, k, c, device);
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

    fn calculate_wait(&self) {
        self.calculate();
        self.queue().finish().unwrap();
    }

    fn queue(&self) -> &Queue {
        &self.queue
    }
}
