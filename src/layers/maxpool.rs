use super::*;
use crate::cl_util as cl;
use crate::geometry::*;
use crate::util::*;
use ocl::SpatialDims;
use std::ops::Deref;

#[derive(Clone)]
pub struct MaxpoolLayer {
    in_shape: ImageGeometry,
    out_shape: ImageGeometry,
}

impl MaxpoolLayer {
    pub fn new(in_shape: &ImageGeometry, stride: usize) -> MaxpoolLayer {
        let out_shape = ImageGeometry::new(in_shape.side() / stride, in_shape.channels());
        let layer = MaxpoolLayer {
            in_shape: *in_shape,
            out_shape,
        };
        debug!(
            "Create max-pool ({}) with input: {}, output: {}.",
            stride,
            layer.num_in(),
            layer.num_out()
        );
        layer
    }

    pub fn input_shape(&self) -> &ImageGeometry {
        &self.in_shape
    }
    pub fn output_shape(&self) -> &ImageGeometry {
        &self.out_shape
    }
}

impl Layer for MaxpoolLayer {
    fn num_out(&self) -> usize {
        self.out_shape.num_elems()
    }
    fn num_in(&self) -> usize {
        self.in_shape.num_elems()
    }
    fn gws_hint(&self) -> SpatialDims {
        SpatialDims::Three(
            self.in_shape.side(),
            self.in_shape.side(),
            self.in_shape.channels(),
        )
    }
    /// Finds the largest possible local-work-size that fits with the data and the GPU max work-group-size
    fn lws_hint(&self, device_max_wgs: usize) -> SpatialDims {
        // Verify that the device max wgs is a power of two
        if (device_max_wgs & (device_max_wgs - 1)) != 0 {
            unimplemented!("device_max_wgs is not a power of two")
        }

        // Largest possible local-work-size-side on the device is the sqrt of the device_max_wgs
        let mut lws_side = (device_max_wgs as f64).sqrt() as usize;

        // Find a local-work-size-side as large as possible that it fits with the data
        while (self.in_shape.side() % lws_side) != 0 {
            lws_side >>= 1;

            // Fail if the local-work-size-side becomes too small
            if lws_side == 1 {
                panic!("unable to find a power of two for max-pool layer local work group size and it cannot be 1x1")
            }
        }

        SpatialDims::Two(lws_side, lws_side)
    }
    fn name(&self) -> &'static str {
        "maxpool"
    }
}

#[test]
fn test_lws_hint() {
    let mxp1 = MaxpoolLayer::new(&ImageGeometry::new(96, 32), 2);

    let lws_hint_1024 = mxp1.lws_hint(1024);
    assert_eq!(lws_hint_1024, SpatialDims::Two(32, 32));
    let lws_hint_256 = mxp1.lws_hint(256);
    assert_eq!(lws_hint_256, SpatialDims::Two(16, 16));

    let mxp2 = MaxpoolLayer::new(&ImageGeometry::new(48, 32), 2);

    let lws_hint_1024 = mxp2.lws_hint(1024);
    assert_eq!(lws_hint_1024, SpatialDims::Two(16, 16));
    let lws_hint_256 = mxp2.lws_hint(256);
    assert_eq!(lws_hint_256, SpatialDims::Two(16, 16));
}
