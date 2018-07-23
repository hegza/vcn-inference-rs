use super::*;

pub struct Naive1GemmKernel {
    kernel: Kernel,
    queue: Queue,
    a_buf: Buffer<f32>,
    b_buf: Buffer<f32>,
    use_host_ptr: bool,
}

impl OclGemm<Naive1GemmKernel> for Naive1GemmKernel {
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
        debug_assert_eq!(c.len(), m * n);

        // Share inputs within host memory on CPU's
        let use_host_ptr = device.contains(DeviceType::CPU);

        let src = String::from_utf8_lossy(include_bytes!("cl/1_naive.cl"));

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
            .src(src)
            .build(&context)
            .unwrap();
        let queue = Queue::new(&context, device, None).unwrap();

        let (mut a_buf, mut b_buf, c_buf) = unsafe {
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
                    .use_host_slice(c)
                    .flags(flags::MEM_WRITE_ONLY)
                    .len(c.len()),
            )
        };
        if use_host_ptr {
            a_buf = unsafe { a_buf.use_host_slice(&a) };
            b_buf = unsafe { b_buf.use_host_slice(&b) };
        } else {
            a_buf = a_buf.copy_host_slice(&a);
            b_buf = b_buf.copy_host_slice(&b);
        }
        let (a_buf, b_buf, c_buf) = (
            a_buf.build().unwrap(),
            b_buf.build().unwrap(),
            c_buf.build().unwrap(),
        );

        let kernel = {
            // This is 32 in the original example; that would produce gws of 1024, but the maximum of
            // my desktop GPU is 256 (16x16).
            const TS: usize = 16;
            let lws = SpatialDims::Two(TS, TS);
            let gws = SpatialDims::Two(m, n);
            Kernel::builder()
                .program(&program)
                .name("myGEMM1")
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
                .unwrap()
        };
        queue.finish().unwrap();

        Naive1GemmKernel {
            kernel,
            a_buf,
            b_buf,
            queue,
            use_host_ptr,
        }
    }

    fn set_buffers_from_slices(&self, a: &[f32], b: &[f32]) {
        debug_assert!(
            !self.use_host_ptr,
            "memory region for buffers has already been set via ocl::BufferBuilder::use_host_ptr"
        );
        self.a_buf.write(a).enq().unwrap();
        self.b_buf.write(b).enq().unwrap();
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
