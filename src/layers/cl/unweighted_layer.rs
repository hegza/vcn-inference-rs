use super::*;

pub trait ClUnweightedLayer<T>: ClLayer<T>
where
    T: Coeff,
{
    fn create_kernel(
        &self,
        kernel_func: &str,
        in_buf: &Buffer<T>,
        out_buf: &Buffer<T>,
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
            "Create unweighted kernel {}, gws: {:?} = {}.",
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
            .arg(out_buf);
        if let Some(lws) = lws {
            builder.local_work_size(lws);
            debug!("\tLocal-work-size set as: {:?} = {}.", lws, lws.to_len());
        }
        builder.build().unwrap()
    }
    /// Passing None as device_type uses default device (usually GPU if available).
    fn impl_standalone(
        &self,
        kernel_srcs: &[&str],
        kernel_name: &str,
        addt_cmplr_opts: &[&str],
        device_type: Option<DeviceType>,
        lws_policy: LocalWorkSizePolicy,
    ) -> LayerImpl<T> {
        let (queue, program, _context) = cl::init::<T>(kernel_srcs, addt_cmplr_opts, device_type);

        // Create buffers
        let (in_buf, out_buf) = self.create_io_bufs(
            flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
            flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        );

        let kernel =
            self.create_kernel(kernel_name, &in_buf, &out_buf, lws_policy, &program, &queue);

        queue.finish().unwrap();

        LayerImpl {
            in_buf,
            out_buf,
            kernel,
            queue,
        }
    }
}
