use super::*;

pub trait ClWeightedLayer<T>: WeightedLayer<T> + ClLayer<T>
where
    T: Coeff,
{
    fn create_kernel(
        &self,
        kernel_func: &str,
        in_buf: &Buffer<T>,
        out_buf: &Buffer<T>,
        wgts_buf: &Buffer<T>,
        lws: LocalWorkSizePolicy,
        program: &Program,
        queue: &Queue,
    ) -> Kernel {
        use self::LocalWorkSizePolicy::*;
        let lws = match lws {
            Specify(dims) => Some(dims),
            Infer { dev_max_wgs } => Some(self.lws_hint(dev_max_wgs)),
            UseDefault => None,
        };
        let gws = self.gws_hint();

        debug!(
            "Create weighted kernel {}, gws: {:?} = {}.",
            kernel_func,
            gws,
            gws.to_len()
        );

        let mut builder = KernelBuilder::new();
        builder
            .program(program)
            .name(kernel_func)
            .queue(queue.clone())
            .global_work_size(gws)
            .arg(in_buf)
            .arg(out_buf)
            .arg(wgts_buf);
        if let Some(lws) = lws {
            builder.local_work_size(lws);
            debug!("\tLocal-work-size set as: {:?} = {}.", lws, lws.to_len());
        }
        builder.build().unwrap()
    }
    // TODO: generalized version over N-layers
    fn impl_standalone(
        &self,
        kernel_srcs: &[&str],
        kernel_name: &str,
        addt_cmplr_opts: Option<&[&str]>,
        device_type: Option<DeviceType>,
        lws_policy: LocalWorkSizePolicy,
    ) -> LayerImpl<T> {
        // Select device
        let platform = Platform::default();
        let device = match device_type {
            Some(dt) => Device::list(platform, Some(dt))
                .unwrap()
                .first()
                .unwrap()
                .clone(),
            None => Device::first(platform).unwrap(),
        };
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .unwrap();

        let mut program_b = Program::builder();
        // Add default compiler options
        cl_util::configure_program::<T>(&mut program_b, &device);

        // Additional compiler options
        if let Some(opts) = addt_cmplr_opts {
            for &opt in opts {
                program_b.cmplr_opt(opt);
            }
        }

        // Input the kernel source files
        for &src in kernel_srcs {
            program_b.src_file(&format!("src/cl/{}", src));
        }

        let program = program_b.build(&context).unwrap();
        let queue = Queue::new(&context, device, None).unwrap();

        // Create buffers
        let wgts_buf = self.create_wgts_buf(&queue);
        let (in_buf, out_buf) = self.create_io_bufs(
            flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
            flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        );

        let kernel = self.create_kernel(
            kernel_name,
            &in_buf,
            &out_buf,
            &wgts_buf,
            lws_policy,
            &program,
            &queue,
        );

        queue.finish().unwrap();

        LayerImpl {
            in_buf,
            out_buf,
            kernel,
            queue,
            program,
            context,
        }
    }
    /// Create a read-only buffer on-device for weights and write the weights
    fn create_wgts_buf(&self, queue: &Queue) -> Buffer<T> {
        trace!(
            "ClLayer::create_wgts_buf with {} elements. Flags: {:?}.",
            self.num_in(),
            flags::MEM_READ_ONLY
        );
        let buf = cl::create_buffer::<T>(self.num_weights(), flags::MEM_READ_ONLY, queue).unwrap();
        buf.write(self.weights()).enq().unwrap();
        buf
    }
}
