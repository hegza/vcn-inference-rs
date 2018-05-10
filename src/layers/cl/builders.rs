use super::*;

pub struct ClKernelChainBuilder<'p, 'b, T>
where
    T: Coeff,
{
    program: &'p Program,
    queue: Queue,
    layer_idx: usize,
    wgts_idx: usize,
    io_bufs: &'b [Buffer<T>],
    wgts_bufs: &'b [Buffer<T>],
}

impl<'p, 'b, T> ClKernelChainBuilder<'p, 'b, T>
where
    T: Coeff,
{
    pub fn new(
        io_bufs: &'b [Buffer<T>],
        wgts_bufs: &'b [Buffer<T>],
        program: &'p Program,
        queue: Queue,
    ) -> ClKernelChainBuilder<'p, 'b, T> {
        ClKernelChainBuilder::<'p, 'b, T> {
            program,
            queue,
            layer_idx: 0,
            wgts_idx: 0,
            io_bufs: io_bufs.clone(),
            wgts_bufs: wgts_bufs.clone(),
        }
    }
    pub fn build_io_kernel(
        &mut self,
        layer: &ClUnweightedLayer<T>,
        kernel_func: &str,
        lws_policy: LocalWorkSizePolicy,
    ) -> Kernel {
        let kernel = layer.create_kernel(
            kernel_func,
            &self.io_bufs[self.layer_idx],     // In
            &self.io_bufs[self.layer_idx + 1], // Out
            lws_policy,
            &self.program,
            &self.queue,
        );
        self.layer_idx += 1;
        kernel
    }
    pub fn build_iow_kernel(
        &mut self,
        layer: &ClWeightedLayer<T>,
        kernel_func: &str,
        lws_policy: LocalWorkSizePolicy,
    ) -> Kernel {
        let kernel = layer.create_kernel(
            kernel_func,
            &self.io_bufs[self.layer_idx],     // In
            &self.io_bufs[self.layer_idx + 1], // Out
            &self.wgts_bufs[self.wgts_idx],    // Weights
            lws_policy,
            &self.program,
            &self.queue,
        );
        self.layer_idx += 1;
        self.wgts_idx += 1;
        kernel
    }
}
