use super::*;
use ocl::builders::KernelBuilder;
use ocl::{Buffer, Kernel, Program, Queue, SpatialDims};

pub struct BinOp<T>
where
    T: OclPrm,
{
    kernel: Kernel,
    q: Queue,
    out_buf: Buffer<T>,
}

impl<T> BinOp<T>
where
    T: OclPrm,
{
    pub fn new(
        kernel_name: &str,
        lhs: &[T],
        rhs: &[T],
        out_len: usize,
        gws: SpatialDims,
        q: Queue,
        prog: &Program,
    ) -> BinOp<T> {
        // Create input buffers
        let lhs_buf = create_buffer::<T>(lhs.len(), MEM_READ_ONLY, &q).unwrap();
        lhs_buf.write(lhs).enq().unwrap();
        let rhs_buf = create_buffer::<T>(rhs.len(), MEM_READ_ONLY, &q).unwrap();
        rhs_buf.write(rhs).enq().unwrap();

        // Create output buffer
        let out_buf = create_buffer::<T>(out_len, MEM_WRITE_ONLY, &q).unwrap();

        // Create kernel
        let kernel = {
            let mut kb = ocl::builders::KernelBuilder::new();
            kb.program(prog)
                .name(kernel_name)
                .queue(q.clone())
                .global_work_size(gws)
                .arg(&lhs_buf)
                .arg(&rhs_buf)
                .arg(&out_buf);
            kb.build().unwrap()
        };
        q.finish().unwrap();

        BinOp::<T> { kernel, q, out_buf }
    }

    pub fn result(&self) -> Vec<T> {
        unsafe {
            self.kernel.enq().unwrap();
            self.q.finish().unwrap();
            read_buf(&self.out_buf).unwrap()
        }
    }
}
