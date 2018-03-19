use util::*;
use geometry::*;
use super::*;
use std::ops::Deref;
use ocl::SpatialDims;

pub struct MaxpoolLayer {
    in_shape: ImageGeometry,
    out_shape: ImageGeometry,
}

impl MaxpoolLayer {
    pub fn new(in_shape: ImageGeometry, stride: usize) -> MaxpoolLayer {
        trace!(
            "Create maxpool({})-layer with input-shape: {:?}.",
            stride,
            in_shape
        );
        let out_shape = ImageGeometry::new(in_shape.side() / 2, in_shape.channels());
        MaxpoolLayer {
            in_shape,
            out_shape,
        }
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
}
