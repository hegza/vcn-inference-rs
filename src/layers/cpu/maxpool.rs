use super::*;
use math::*;

impl<T> ComputeOnCpu<T> for MaxpoolLayer
where
    T: CoeffFloat,
{
    fn compute(&self, in_buf: &[T]) -> Vec<T> {
        let side = (self.num_in() as f32).sqrt() as usize;
        let stride = ((self.num_in() / self.num_out()) as f32).sqrt() as usize;
        let mut out = Vec::with_capacity(self.num_out());
        for y in 0..side / stride {
            for x in 0..side / stride {
                let lt_idx = y * stride * side + x * stride;
                let mut vals = Vec::with_capacity(stride * stride);
                for ix in 0..stride {
                    for iy in 0..stride {
                        vals.push(in_buf[lt_idx + iy * side + ix]);
                    }
                }
                out.push(vals.iter().fold(T::zero(), T::generic_max));
            }
        }
        out
    }
}
