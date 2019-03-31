use ocl::{Context, Program, Queue};

pub struct ClContext {
    pub cpu_queue: Queue,
    pub gpu_queue: Option<Queue>,
    pub program: Program,
    pub _context: Context,
}

impl ClContext {
    pub fn is_dual_device(&self) -> bool {
        self.gpu_queue.is_some()
    }
    pub fn conv_queue(&self) -> Queue {
        if self.is_dual_device() {
            let gpu_queue = self.gpu_queue.clone().unwrap();
            gpu_queue.clone()
        } else {
            self.cpu_queue.clone()
        }
    }
}
