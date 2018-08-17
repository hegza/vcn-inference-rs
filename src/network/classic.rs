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
pub const WEIGHTS_DIR: &str = "input/weights";

pub struct ClassicNetwork<T>
where
    T: Coeff,
{
    queue: Queue,
    pub in_buf: Buffer<T>,
    input_shape: ImageGeometry,
    conv_relu1: Kernel,
    conv_relu2: Kernel,
    krn_dense3: Kernel,
    dense3_out_buf: Buffer<T>,
    dense4: DenseLayer<T>,
    dense5: DenseLayer<T>,
}

impl<T> ClassicNetwork<T>
where
    T: Coeff + ReadBinFromFile,
{
    pub fn create_layers(params: &ClassicHyperParams) -> Layers<T> {
        let params = NetworkParams::new(params.clone());
        let wgts = Weights::default();
        // Create a representation of the 1st convolutional layer with weights from a file
        let conv1 = params.create_conv(1, wgts.0);
        // Create a representation of the 2nd convolutional layer with weights from a file
        let conv2 = params.create_conv(2, wgts.1);
        // Create the representations of the fully-connected layers
        let dense3 = params.create_dense(3, wgts.2);
        let dense4 = params.create_dense(4, wgts.3);
        let dense5 = params.create_dense(5, wgts.4);

        // Verify that I/O dimensions match between layers
        verify_network_dimensions(&[&conv1, &conv2, &dense3, &dense4, &dense5]);

        (conv1, conv2, dense3, dense4, dense5)
    }

    /// Initializes the network, kernels and buffers. Returns only after all OpenCL-commands have
    /// finished running. Note that you must call upload_buffers before the network is run.
    pub fn new() -> ClassicNetwork<T> {
        // Create the network representation from network hyper-parameters
        let layers = ClassicNetwork::create_layers(&CLASSIC_HYPER_PARAMS);

        // Initialize OpenCL
        let (queue, program, _context) =
            cl::init::<T>(&["src/cl/conv_mxp_relu.cl", "src/cl/mtx_mul.cl"], &[], None);

        // Create shorthands (and move)
        let (conv1, conv2, dense3, dense4, dense5) = layers;

        // Allocate read-only memory for the weights of the 1st three layers
        let wgts_bufs = create_weights_bufs(&[&conv1, &conv2, &dense3], &queue);

        // Allocate read-only memory for the input geometry on device with host-accessible pointer for
        // writing input from file
        let mut bufs = create_buffer_chain(&[&conv1, &conv2, &dense3], &queue);

        // Create the kernel for the 1st layer (Convolution + ReLU)
        let conv_relu1 = conv1.create_kernel(
            "conv_relu_1",
            &bufs[0],
            &bufs[1],
            &wgts_bufs[0],
            LocalWorkSizePolicy::UseDefault,
            &program,
            &queue,
        );

        // Create the kernel for the 2nd layer (Convolution + ReLU)
        let conv_relu2 = conv2.create_kernel(
            "conv_relu_2",
            &bufs[1],
            &bufs[2],
            &wgts_bufs[1],
            LocalWorkSizePolicy::UseDefault,
            &program,
            &queue,
        );

        // Create the kernel for the 3rd layer (Dense layer matrix multiplication)
        let krn_dense3 = dense3.create_kernel(
            "mtx_mul",
            &bufs[2],
            &bufs[3],
            &wgts_bufs[2],
            LocalWorkSizePolicy::UseDefault,
            &program,
            &queue,
        );

        // Wait until all commands have finished running before returning.
        queue.finish().unwrap();

        // Move and store the first and last buffer
        let mut buf_drain = bufs.drain(..);
        let in_buf = buf_drain.next().unwrap();
        let dense3_out_buf = buf_drain.next_back().unwrap();

        ClassicNetwork {
            queue,
            conv_relu1,
            conv_relu2,
            krn_dense3,
            in_buf,
            dense3_out_buf,
            input_shape: *conv1.input_shape(),
            dense4,
            dense5,
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
    fn predict(&self, input_data: &[T]) -> Vec<f32> {
        let q = &self.queue;
        unsafe {
            cl::map_to_buf(&self.in_buf, input_data).unwrap();

            // Enqueue the kernel for the 1st layer (Convolution + ReLU)
            self.conv_relu1.cmd().queue(q).enq().unwrap();
            // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
            self.conv_relu2.cmd().queue(q).enq().unwrap();
            // Enqueue the 3rd layer (fully-connected)
            self.krn_dense3.cmd().queue(q).enq().unwrap();
        }
        // Wait for all on-device calculations to finish
        q.finish().unwrap();

        let dense3_out = &unsafe { cl::read_buf(&self.dense3_out_buf).unwrap() };

        // Run the 4th layer (fully-connected)
        let dense4_out = relu(self.dense4.compute(&dense3_out));

        // Run the 5th layer (fully-connected)
        softmax(&self.dense5.compute(&dense4_out))
    }
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

pub struct Weights<T>(pub Vec<T>, pub Vec<T>, pub Vec<T>, pub Vec<T>, pub Vec<T>);

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

pub type Layers<T> = (
    ConvLayer<T>,
    ConvLayer<T>,
    DenseLayer<T>,
    DenseLayer<T>,
    DenseLayer<T>,
);

impl Deref for NetworkParams {
    type Target = ClassicHyperParams;

    fn deref(&self) -> &Self::Target {
        &self.hyper_params
    }
}

pub trait ClassicWeights<T>
where
    T: Coeff,
{
}

impl<T> ClassicWeights<T> for Weights<T> where T: Coeff {}

impl<T> Default for Weights<T>
where
    T: Coeff + ReadBinFromFile,
{
    fn default() -> Weights<T> {
        Weights(
            T::read_bin_from_file(&format!("{}/conv1-f32-le.bin", WEIGHTS_DIR)),
            T::read_bin_from_file(&format!("{}/conv2-f32-le.bin", WEIGHTS_DIR)),
            T::read_bin_from_file(&format!("{}/fc3-f32-le.bin", WEIGHTS_DIR)),
            T::read_bin_from_file(&format!("{}/fc4-f32-le.bin", WEIGHTS_DIR)),
            T::read_bin_from_file(&format!("{}/fc5-f32-le.bin", WEIGHTS_DIR)),
        )
    }
}
