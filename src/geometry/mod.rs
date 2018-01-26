/// A descriptor for the filter-geometry
#[derive(Copy, Clone)]
pub struct PaddedSquare {
    side: usize,
    padding: usize,
}

/// A descriptor for input and intermediary image geometry
#[derive(Copy, Clone)]
pub struct ImageGeometry {
    side: usize,
    padding: usize,
    channels: usize,
}

impl ImageGeometry {
    pub fn new(side: usize, padding: usize, channels: usize) -> ImageGeometry {
        ImageGeometry {
            side,
            padding,
            channels,
        }
    }
    /// Returns a clone with the padding extended to the amount required to fit
    /// the filter into the image properly. Discards previous padding.
    pub fn with_filter_padding(&self, filter_shape: &PaddedSquare) -> ImageGeometry {
        ImageGeometry::new(self.side, filter_shape.side() - 1, self.channels)
    }
}

pub trait Square {
    fn side(&self) -> usize;
    fn num_elements(&self) -> usize;
}

impl PaddedSquare {
    pub fn new(side: usize, padding: usize) -> PaddedSquare {
        PaddedSquare { side, padding }
    }
    pub fn from_side(side: usize) -> PaddedSquare {
        PaddedSquare::new(side, 0)
    }
}

impl Square for PaddedSquare {
    fn side(&self) -> usize {
        self.side + self.padding
    }
    fn num_elements(&self) -> usize {
        self.side() * self.side()
    }
}

impl Square for ImageGeometry {
    fn side(&self) -> usize {
        self.side + self.padding
    }
    fn num_elements(&self) -> usize {
        self.side() * self.side() * self.channels
    }
}
