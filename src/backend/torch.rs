use tch::Tensor;

use crate::backend::{Backend, Operation};

impl Backend for &Tensor {
    type Output = Tensor;

    fn shape(&self) -> Vec<usize> {
        self.size().iter().map(|&x| x as usize).collect::<Vec<_>>()
    }

    fn reshape(self, shape: &[usize]) -> Self::Output {
        self.reshape(&shape.iter().map(|&x| x as i64).collect::<Vec<_>>())
    }

    fn transpose(self, axes: &[usize]) -> Self::Output {
        self.permute(&axes.iter().map(|&x| x as i64).collect::<Vec<_>>())
    }

    fn reduce_axes_v2(self, axes_operations: &mut [(usize, Operation)]) -> Self::Output {
        let mut output = self.shallow_clone();

        axes_operations.sort_by_key(|(axis, _)| *axis);

        for (axis, operation) in axes_operations.iter().rev() {
            output = match operation {
                Operation::Min => output.min_dim(*axis as i64, false).0,
                Operation::Max => output.max_dim(*axis as i64, false).0,
                Operation::Sum => output.sum_dim_intlist(&[*axis as i64], false, output.kind()),
                Operation::Mean => output.mean_dim(&[*axis as i64], false, output.kind()),
                Operation::Prod => output.prod_dim_int(*axis as i64, false, output.kind()),
            };
        }

        output
    }

    fn add_axes(self, naxes: usize, pos2len: &[(usize, usize)]) -> Self::Output {
        let mut output = self.shallow_clone();

        let mut repeats = vec![1; naxes];

        for &(axis_pos, axis_len) in pos2len {
            output = output.unsqueeze(axis_pos as i64);
            repeats[axis_pos] = axis_len as i64;
        }

        output.repeat(&repeats)
    }
}

impl Backend for Tensor {
    type Output = Tensor;

    fn shape(&self) -> Vec<usize> {
        Backend::shape(&self)
    }

    fn reshape(self, shape: &[usize]) -> Self::Output {
        Backend::reshape(&self, shape)
    }

    fn transpose(self, axes: &[usize]) -> Self::Output {
        Backend::transpose(&self, axes)
    }

    fn reduce_axes_v2(self, axes_operations: &mut [(usize, Operation)]) -> Self::Output {
        Backend::reduce_axes_v2(&self, axes_operations)
    }

    fn add_axes(self, naxes: usize, pos2len: &[(usize, usize)]) -> Self::Output {
        Backend::add_axes(&self, naxes, pos2len)
    }
}
