use super::*;
use geometry::*;

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
    input_shape: ImageGeometry,
    conv_relu1: Kernel,
    conv_relu2: Kernel,
    dense3_kernel: Kernel,
    dense3_out_buf: Buffer<T>,
    dense4: DenseLayer<T>,
    dense5: DenseLayer<T>,
}

impl<T> ClassicNetwork<T>
where
    T: CoeffFloat + ReadBinFromFile,
{
    /// Initializes the network, kernels and buffers. Returns only after all OpenCL-commands have
    /// finished running. Note that you must call upload_buffers before the network is run.
    pub fn new(program: &Program, queue: &Queue) -> ClassicNetwork<T> {
        // Create the network representation from network hyper-parameters
        let layers = create_layers(CLASSIC_HYPER_PARAMS.clone());
        let (conv1, conv2, dense3, dense4, dense5) = (
            layers.conv1,
            layers.conv2,
            layers.dense3,
            layers.dense4,
            layers.dense5,
        );

        // Allocate read-only memory for the weights of the 1st three layers
        let conv1_wgts_buf = conv1.create_wgts_buf(&queue).unwrap();
        let conv2_wgts_buf = conv2.create_wgts_buf(&queue).unwrap();
        let dense3_wgts_buf = dense3.create_wgts_buf(&queue).unwrap();

        // Allocate read-only memory for the input geometry on device with host-accessible pointer for
        // writing input from file
        let in_buf = conv1
            .create_in_buf(flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR, &queue)
            .unwrap();
        // Allocate read-write memory for the 1st feature map on device
        let fm1_buf = conv2.create_in_buf(flags::MEM_READ_WRITE, &queue).unwrap();
        // Allocate read-write memory for the 2nd feature map on device
        let fm2_buf = conv2.create_out_buf(flags::MEM_READ_WRITE, &queue).unwrap();
        // Allocate write-only memory for the dense (3rd) layer output on device with host pointer for reading
        let dense3_out_buf = dense3
            .create_out_buf(flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR, &queue)
            .unwrap();

        // Create the kernel for the 1st layer (Convolution + ReLU)
        let conv_relu1 = Kernel::builder().program(&program).name("conv_relu_1")
            .queue(queue.clone())
            .global_work_size(conv1.gws_hint())
            // Input
            .arg(&in_buf)
            // Output
            .arg(&fm1_buf)
            .arg(&conv1_wgts_buf).build().unwrap();

        // Create the kernel for the 2nd layer (Convolution + ReLU)
        let conv_relu2 = Kernel::builder().program(&program).name("conv_relu_2")
            .queue(queue.clone())
            .global_work_size(conv2.gws_hint())
            // Input
            .arg(&fm1_buf)
            // Output
            .arg(&fm2_buf)
            .arg(&conv2_wgts_buf).build().unwrap();

        // Create the kernel for the 3rd layer (Dense layer matrix multiplication)
        let dense3_kernel = Kernel::builder().program(&program).name("mtx_mulf")
            .queue(queue.clone())
            .global_work_size(dense3.gws_hint())
            // Input
            .arg(&fm2_buf)
            // Output
            .arg(&dense3_out_buf)
            .arg(&dense3_wgts_buf).build().unwrap();

        // Write the weights of the 1st three layers to the global memory of the device
        conv1_wgts_buf.write(conv1.weights()).enq().unwrap();
        conv2_wgts_buf.write(conv2.weights()).enq().unwrap();
        dense3_wgts_buf.write(dense3.weights()).enq().unwrap();

        // Wait until all commands have finished running before returning.
        queue.finish().unwrap();
        ClassicNetwork {
            conv_relu1,
            conv_relu2,
            dense3_kernel,
            dense3_out_buf,
            in_buf,
            input_shape: conv1.input_shape().clone(),
            dense4: dense4,
            dense5: dense5,
        }
    }
    pub fn input_shape(&self) -> &ImageGeometry {
        &self.input_shape
    }
}

impl<T> Predict<T> for ClassicNetwork<T>
where
    T: CoeffFloat,
{
    /// Maps the input buffer, and runs the network, returning the result.
    fn predict(&self, input_data: &[T], queue: &Queue) -> Vec<f32> {
        unsafe {
            cl::map_to_buf(&self.in_buf, &input_data).unwrap();

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
        let dense4_out = relu(&self.dense4.mtx_mul(&dense3_out));

        // Run the 5th layer (fully-connected)
        softmax(&self.dense5.mtx_mul(&dense4_out))
    }
}

pub fn create_layers<T>(params: ClassicHyperParams) -> Layers<T>
where
    T: Coeff + ReadBinFromFile,
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

/// Creates a standalone kernel for benchmarking. Uploads input data. Returns only after all commands have finished.
pub fn create_standalone_kernel<L: ClWeightedLayer<T>, T: Coeff>(
    layer: &L,
    kernel_func: &str,
    input_data: &[T],
) -> ocl::Result<(Kernel, Buffer<T>, Queue)> {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init(&["conv_relu.cl", "mtx_mulf.cl"]).unwrap();

    let wgts_buf = layer.create_wgts_buf(&queue)?;
    let (in_buf, out_buf) = layer.create_io_bufs(
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    )?;

    let kernel = Kernel::builder().program(&program).name(kernel_func)
        .queue(queue.clone())
        .global_work_size(layer.gws_hint())
        // Input
        .arg(&in_buf)
        // Output
        .arg(&out_buf)
        .arg(&wgts_buf).build().unwrap();

    // Write the weights and input to the global memory of the device
    wgts_buf.write(layer.weights()).enq().unwrap();
    // TODO: It's maybe not fair to do this inside the kernel initialization
    unsafe {
        cl::map_to_buf(&in_buf, &input_data).unwrap();
    }
    queue.finish().unwrap();

    Ok((kernel, out_buf, queue))
}

#[derive(Clone, Debug)]
pub struct ClassicHyperParams {
    pub source_side: usize,
    // channels for each rgb color
    pub num_source_channels: usize,
    // the size of the filter/kernels
    pub conv_1_filter_side: usize,
    pub conv_2_filter_side: usize,
    // the number of feature maps
    pub num_feature_maps: usize,
    pub stride: usize,
    // ???: what is this, what does it do? was originally magic in jani's code.
    pub fully_connected_const: usize,
    pub num_output_classes: usize,
}

pub struct NetworkParams {
    hyper_params: ClassicHyperParams,
    conv1_filter_shape: PaddedSquare,
    conv2_filter_shape: PaddedSquare,
    padded_input_shape: ImageGeometry,
    padded_fm1_shape: ImageGeometry,
    fm2_shape: ImageGeometry,
}

impl NetworkParams {
    pub fn new(hyper_params: ClassicHyperParams) -> NetworkParams {
        let conv1_filter_shape = PaddedSquare::from_side(hyper_params.conv_1_filter_side);
        let conv2_filter_shape = PaddedSquare::from_side(hyper_params.conv_2_filter_side);

        // Create descriptor for input geometry with the shape and properties of an image
        let input_shape =
            ImageGeometry::new(hyper_params.source_side, hyper_params.num_source_channels);
        let padded_input_shape = input_shape.with_filter_padding(&conv1_filter_shape);
        // Feature map 1 is a fraction of the side of initial image geometry due to stride
        let fm1_shape = ImageGeometry::new(
            input_shape.side() / hyper_params.stride,
            hyper_params.num_feature_maps,
        );
        let padded_fm1_shape = fm1_shape.with_filter_padding(&conv2_filter_shape);
        // Feature map 2 is a fraction of the side of the tier 1 feature map due to stride
        let fm2_shape = ImageGeometry::new(
            fm1_shape.side() / hyper_params.stride,
            hyper_params.num_feature_maps,
        );

        NetworkParams {
            hyper_params,
            conv1_filter_shape,
            conv2_filter_shape,
            padded_input_shape,
            padded_fm1_shape,
            fm2_shape,
        }
    }
    pub fn create_conv<T>(&self, idx: usize, weights: Vec<T>) -> ConvLayer<T>
    where
        T: Coeff,
    {
        let (filter_elems, in_shape, out_shape) = match idx {
            1 => (
                self.conv1_filter_shape.num_elems(),
                self.padded_input_shape,
                self.padded_fm1_shape,
            ),
            2 => (
                self.conv2_filter_shape.num_elems(),
                self.padded_fm1_shape,
                self.fm2_shape,
            ),
            _ => panic!(format!("no conv layer for idx {}", idx)),
        };
        ConvLayer::from_shapes(filter_elems, &in_shape, &out_shape, weights)
    }
    pub fn create_dense<T>(&self, idx: usize, weights: Vec<T>) -> DenseLayer<T>
    where
        T: Coeff,
    {
        let (num_in, num_out) = match idx {
            3 => (self.fm2_shape.num_elems(), self.fully_connected_const),
            4 => (self.fully_connected_const, self.fully_connected_const),
            5 => (self.fully_connected_const, self.num_output_classes),
            _ => panic!(format!("no dense layer for idx {}", idx)),
        };
        DenseLayer::new(num_in, num_out, weights)
    }
}

pub struct Layers<T>
where
    T: Coeff,
{
    pub conv1: ConvLayer<T>,
    pub conv2: ConvLayer<T>,
    pub dense3: DenseLayer<T>,
    pub dense4: DenseLayer<T>,
    pub dense5: DenseLayer<T>,
}

impl Deref for NetworkParams {
    type Target = ClassicHyperParams;

    fn deref(&self) -> &Self::Target {
        &self.hyper_params
    }
}
