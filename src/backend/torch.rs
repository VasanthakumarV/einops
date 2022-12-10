use tch::Tensor;

use crate::backend::{Backend, Operation};

impl<T: AsRef<Tensor>> Backend for T {
    type Output = Tensor;

    fn shape(self) -> Vec<usize> {
        self.as_ref()
            .size()
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
    }

    fn reshape(self, shape: &[usize]) -> Self::Output {
        self.as_ref()
            .reshape(&shape.iter().map(|&x| x as i64).collect::<Vec<_>>())
    }

    fn transpose(self, axes: &[usize]) -> Self::Output {
        self.as_ref()
            .permute(&axes.iter().map(|&x| x as i64).collect::<Vec<_>>())
    }

    fn reduce_axes(self, axes_operations: &mut [(usize, Operation)]) -> Self::Output {
        let mut output = self.as_ref().shallow_clone();

        axes_operations.sort_by_key(|(axis, _)| *axis);

        for (axis, operation) in axes_operations.iter().rev() {
            output = match operation {
                Operation::Min => output.min_dim(*axis as i64, false).0,
                Operation::Max => output.max_dim(*axis as i64, false).0,
                Operation::Sum => output.sum_dim_intlist(&[*axis as i64][..], false, output.kind()),
                Operation::Mean => output.mean_dim(&[*axis as i64][..], false, output.kind()),
                Operation::Prod => output.prod_dim_int(*axis as i64, false, output.kind()),
            };
        }

        output
    }

    fn add_axes(self, naxes: usize, pos2len: &[(usize, usize)]) -> Self::Output {
        let mut output = self.as_ref().shallow_clone();

        let mut repeats = vec![1; naxes];

        for &(axis_pos, axis_len) in pos2len {
            output = output.unsqueeze(axis_pos as i64);
            repeats[axis_pos] = axis_len as i64;
        }

        output.repeat(&repeats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn tch_reduce() {
        let tests = vec![(
            Tensor::of_slice(&[
                0.66984287, 0.52894678, 0.85415958, 0.17721198, 0.81804799, 0.80991797, 0.64868822,
                0.96697902, 0.08047191, 0.46024353, 0.21955009, 0.31731976, 0.05446258, 0.39454557,
                0.40949016, 0.21366165, 0.2357463, 0.93699481, 0.64522596, 0.4383618, 0.54871827,
                0.87823442, 0.01261184, 0.90636503,
            ])
            .reshape(&[4, 2, 3]),
            [(0, Operation::Min)],
            Tensor::of_slice(&[
                0.05446258, 0.39454557, 0.08047191, 0.17721198, 0.01261184, 0.31731976,
            ])
            .reshape(&[2, 3]),
        )];

        for (tensor, mut axes_operations, expected) in tests {
            assert_eq!(tensor.reduce_axes(&mut axes_operations), expected);
        }
    }

    #[test]
    fn tch_transpose() {
        let tests = vec![(
            Tensor::arange(2 * 3 * 4 * 5, (Kind::Float, Device::Cpu)).reshape(&[2, 3, 4, 5]),
            &[3, 0, 2, 1],
            Tensor::of_slice(&[
                0, 20, 40, 5, 25, 45, 10, 30, 50, 15, 35, 55, 60, 80, 100, 65, 85, 105, 70, 90,
                110, 75, 95, 115, 1, 21, 41, 6, 26, 46, 11, 31, 51, 16, 36, 56, 61, 81, 101, 66,
                86, 106, 71, 91, 111, 76, 96, 116, 2, 22, 42, 7, 27, 47, 12, 32, 52, 17, 37, 57,
                62, 82, 102, 67, 87, 107, 72, 92, 112, 77, 97, 117, 3, 23, 43, 8, 28, 48, 13, 33,
                53, 18, 38, 58, 63, 83, 103, 68, 88, 108, 73, 93, 113, 78, 98, 118, 4, 24, 44, 9,
                29, 49, 14, 34, 54, 19, 39, 59, 64, 84, 104, 69, 89, 109, 74, 94, 114, 79, 99, 119,
            ])
            .reshape(&[5, 2, 4, 3]),
        )];

        for (tensor, axes, expected) in tests {
            assert_eq!(Backend::transpose(&tensor, axes), expected);
        }
    }

    #[test]
    fn tch_add_axes() {
        let tests = vec![(
            Tensor::arange(1 * 2 * 3, (Kind::Float, Device::Cpu)).reshape(&[1, 2, 3]),
            5,
            &[(0, 5), (3, 3)],
            Tensor::of_slice(&[
                0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3,
                4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1,
                2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5,
                3, 4, 5, 3, 4, 5,
            ])
            .reshape(&[5, 1, 2, 3, 3]),
        )];

        for (tensor, naxes, pos2len, expected) in tests {
            assert_eq!(tensor.add_axes(naxes, pos2len), expected);
        }
    }
}
