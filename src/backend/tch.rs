use tch::{Kind, Tensor};

use crate::backend::{Backend, Operation};

impl Backend for Tensor {
    fn shape(&self) -> Vec<isize> {
        self.size().iter().map(|&x| x as isize).collect::<Vec<_>>()
    }

    fn reshape(&self, shape: &[isize]) -> Self {
        self.reshape(&shape.iter().map(|&x| x as i64).collect::<Vec<_>>())
    }

    fn transpose(&self, dim0: isize, dim1: isize) -> Self {
        self.transpose(dim0 as i64, dim1 as i64)
    }

    fn reduce(&self, operation: Operation, axes: &[isize]) -> Self {
        let mut output = self.shallow_clone();

        let mut axes = axes.to_vec();
        axes.sort();

        for &axis in axes.iter().rev() {
            output = match operation {
                Operation::Min => output.min2(axis as i64, false).0,
                Operation::Max => output.max2(axis as i64, false).0,
                Operation::Sum => output.sum1(&[axis as i64], false, Kind::Float),
                Operation::Mean => output.mean1(&[axis as i64], false, Kind::Float),
                Operation::Prod => output.prod1(axis as i64, false, Kind::Float),
            };
        }

        output
    }

    fn add_axes(&self, naxes: usize, pos2len: &[(usize, isize)]) -> Self {
        let mut output = self.shallow_clone();

        let mut repeats = vec![1; naxes];

        for &(axis_pos, axis_len) in pos2len {
            output = output.unsqueeze(axis_pos as i64);
            repeats.insert(axis_pos, axis_len as i64);
        }

        output.repeat(&repeats)
    }
}
