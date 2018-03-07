use super::*;

pub const CLASSIC_HYPER_PARAMS: ClassicHyperParams = ClassicHyperParams {
    source_side: 96,
    num_source_channels: 3,
    conv_1_filter_side: 5,
    conv_2_filter_side: 5,
    num_feature_maps: 32,
    stride: 2,
    fully_connected_const: 100,
    num_output_classes: 4,
};
pub const WEIGHTS_DIR: &'static str = "input/weights";

pub struct ClassicNetwork<T>
where
    T: Coeff,
{
    pub in_buf: Buffer<T>,
    layers: Layers<T>,
    conv_relu1: Kernel,
    conv_relu2: Kernel,
    dense3_kernel: Kernel,
    dense3_out_buf: Buffer<T>,
}

impl<T> ClassicNetwork<T>
where
    T: CoeffFloat,
{
    /// Initializes the network, kernels and buffers. Returns only after all OpenCL-commands have
    /// finished running. Note that you must call upload_buffers before the network is run.
    pub fn new(program: &Program, queue: &Queue) -> ClassicNetwork<T> {
        // Create the network representation from network hyper-parameters
        let layers = create_layers(CLASSIC_HYPER_PARAMS.clone());

        // Allocate read-only memory for the weights of the 1st three layers
        let conv1_wgts_buf =
            cl::create_buffer::<T>(layers.conv1.num_weights(), flags::MEM_READ_ONLY, &queue).unwrap();
        let conv2_wgts_buf =
            cl::create_buffer::<T>(layers.conv2.num_weights(), flags::MEM_READ_ONLY, &queue).unwrap();
        let dense3_wgts_buf =
            cl::create_buffer::<T>(layers.dense3.num_weights(), flags::MEM_READ_ONLY, &queue).unwrap();

        // Allocate read-only memory for the input geometry on device with host-accessible pointer for
        // writing input from file
        let in_buf = cl::create_buffer::<T>(
            layers.conv1.num_in(),
            flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        ).unwrap();
        // Allocate read-write memory for the 1st feature map on device
        let fm1_buf =
            cl::create_buffer::<T>(layers.conv1.num_out(), flags::MEM_READ_WRITE, &queue).unwrap();
        // Allocate read-write memory for the 2nd feature map on device
        let fm2_buf =
            cl::create_buffer::<T>(layers.conv2.num_out(), flags::MEM_READ_WRITE, &queue).unwrap();
        // Allocate read-write memory for the dense (3rd) layer output on device with host pointer for reading
        let dense3_out_buf = cl::create_buffer::<T>(
            layers.dense3.num_out(),
            flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        ).unwrap();

        // Create the kernel for the 1st layer (Convolution + ReLU)
        let conv_relu1 = Kernel::new("conv_relu_1", &program).unwrap()
            .queue(queue.clone())
            .gws(layers.conv1.gws())
            // Input
            .arg_buf(&in_buf)
            // Output
            .arg_buf(&fm1_buf)
            .arg_buf(&conv1_wgts_buf);

        // Create the kernel for the 2nd layer (Convolution + ReLU)
        let conv_relu2 = Kernel::new("conv_relu_2", &program).unwrap()
            .queue(queue.clone())
            .gws(layers.conv2.gws())
            // Input
            .arg_buf(&fm1_buf)
            // Output
            .arg_buf(&fm2_buf)
            .arg_buf(&conv2_wgts_buf);

        // Create the kernel for the 3rd layer (Dense layer matrix multiplication)
        let dense3_kernel = Kernel::new("mtx_mulf", &program).unwrap()
            .queue(queue.clone())
            .gws(layers.dense3.gws())
            // Input
            .arg_buf(&fm2_buf)
            // Output
            .arg_buf(&dense3_out_buf)
            .arg_buf(&dense3_wgts_buf);

        // Write the weights of the 1st three layers to the global memory of the device
        conv1_wgts_buf.write(layers.conv1.weights()).enq().unwrap();
        conv2_wgts_buf.write(layers.conv2.weights()).enq().unwrap();
        dense3_wgts_buf.write(layers.dense3.weights()).enq().unwrap();

        // Wait until all commands have finished running before returning.
        queue.finish().unwrap();

        ClassicNetwork {
            layers: layers,
            conv_relu1,
            conv_relu2,
            dense3_kernel,
            dense3_out_buf,
            in_buf,
        }
    }
    /// Maps the input buffer, and runs the network, returning the result.
    pub fn predict(&self, input_data: &[T], queue: &Queue) -> Vec<T> {
        unsafe {
            cl::map_to_buf(&self.in_buf, &input_data).unwrap();

            // TODO: is the queue() optional here.unwrap()
            // Enqueue the kernel for the 1st layer (Convolution + ReLU)
            self.conv_relu1.cmd().queue(&queue).enq().unwrap();
            // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
            self.conv_relu2.cmd().queue(&queue).enq().unwrap();
            // Enqueue the 3rd layer (fully-connected)
            self.dense3_kernel.cmd().queue(&queue).enq().unwrap();
        }
        // Wait for all on-device calculations to finish
        queue.finish().unwrap();

        let dense3_out = relu(&unsafe { cl::read_buf(&self.dense3_out_buf).unwrap() });

        // Run the 4th layer (fully-connected)
        let dense4_out = mtxmul_relu(&dense3_out, &self.dense4);

        // Run the 5th layer (fully-connected)
        mtxmul_softmax(&dense4_out, &self.dense5)
    }
    pub fn input_shape(&self) -> &ImageGeometry {
        self.conv1.input_shape()
    }
}

pub fn create_layers<T>(params: ClassicHyperParams) -> Layers<T>
where
    T: Coeff,
{
    let params = NetworkParams::new(params);
    // Create a representation of the 1st convolutional layer with weights from a file
    let conv1 = params.create_conv(
        1,
        T::read_bin_from_file(&format!("{}/conv1-f32-le.bin", WEIGHTS_DIR)),
    );
    // Create a representation of the 2nd convolutional layer with weights from a file
    let conv2 = params.create_conv(
        2,
        T::read_bin_from_file(&format!("{}/conv2-f32-le.bin", WEIGHTS_DIR)),
    );
    // Create the representations of the fully-connected layers
    let dense3 = params.create_dense(
        3,
        T::read_bin_from_file(&format!("{}/fc3-f32-le.bin", WEIGHTS_DIR)),
    );
    let dense4 = params.create_dense(
        4,
        T::read_bin_from_file(&format!("{}/fc4-f32-le.bin", WEIGHTS_DIR)),
    );
    let dense5 = params.create_dense(
        5,
        T::read_bin_from_file(&format!("{}/fc5-f32-le.bin", WEIGHTS_DIR)),
    );

    // Verify that I/O dimensions match between layers
    verify_network_dimensions(&[&conv1, &conv2, &dense3, &dense4, &dense5]);

    Layers {
        conv1,
        conv2,
        dense3,
        dense4,
        dense5,
    }
}

/// Runs the kernel but returns only after it has finished.
pub fn run_kernel_wait(kernel: &Kernel, queue: &Queue) -> ocl::Result<()> {
    unsafe {
        kernel.cmd().queue(&queue).enq()?;
    }
    queue.finish()
}

/// Creates a standalone kernel for benchmarking. Returns only after all commands have finished.
pub fn create_standalone_kernel<L: Layer<T>, T: Num + OclPrm>(
    layer: &L,
    kernel_func: &str,
    input_data: &[T],
) -> ocl::Result<(Kernel, Buffer<T>, Queue)> {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init("original_kernels.cl").unwrap();

    let wgts_buf = cl::create_buffer::<T>(layer.num_weights(), flags::MEM_READ_ONLY, &queue).unwrap();
    let in_buf = cl::create_buffer::<T>(
        layer.num_in(),
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    ).unwrap();
    let out_buf = cl::create_buffer::<T>(
        layer.num_out(),
        flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    ).unwrap();

    let kernel = Kernel::new(kernel_func, &program).unwrap()
        .queue(queue.clone())
        .gws(layer.gws())
        // Input
        .arg_buf(&in_buf)
        // Output
        .arg_buf(&out_buf)
        .arg_buf(&wgts_buf);

    // Write the weights and input to the global memory of the device
    wgts_buf.write(layer.weights()).enq().unwrap();
    // TODO: It's maybe not fair to do this inside the kernel initialization
    unsafe {
        cl::map_to_buf(&in_buf, &input_data).unwrap();
    }
    queue.finish().unwrap();

    Ok((kernel, out_buf, queue))
}

impl<T> Deref for ClassicNetwork<T>
where
    T: Coeff,
{
    type Target = Layers<T>;

    fn deref(&self) -> &Self::Target {
        &self.layers
    }
}
