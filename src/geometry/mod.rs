/// A descriptor for the filter-geometry
#[derive(Copy, Clone, Debug)]
pub struct PaddedSquare {
    side: usize,
    padding: usize,
}

// TODO: split into PaddedImageGeometry and ImageGeometry
/// A descriptor for input and intermediary image geometry
#[derive(Copy, Clone, Debug)]
pub struct ImageGeometry {
    side: usize,
    padding: usize,
    channels: usize,
}

impl ImageGeometry {
    pub fn new(side: usize, channels: usize) -> ImageGeometry {
        ImageGeometry {
            side,
            padding: 0,
            channels,
        }
    }
    // ???: What is 'properly' here.
    /// Returns a clone with padding set to the amount required to fit the filter into the image properly.
    pub fn with_filter_padding(&self, filter_shape: &PaddedSquare) -> ImageGeometry {
        ImageGeometry {
            side: self.side,
            padding: filter_shape.side() - 1,
            channels: self.channels,
        }
    }
    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn unpadded(&self) -> ImageGeometry {
        ImageGeometry {
            side: self.side,
            padding: 0,
            channels: self.channels,
        }
    }
}

pub trait Square {
    fn side(&self) -> usize;
    fn num_elems(&self) -> usize;
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
    fn num_elems(&self) -> usize {
        self.side() * self.side()
    }
}

impl Square for ImageGeometry {
    fn side(&self) -> usize {
        self.side + self.padding
    }
    fn num_elems(&self) -> usize {
        self.side() * self.side() * self.channels
    }
}
