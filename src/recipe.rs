mod parse;

use std::collections::{HashMap, HashSet};

use crate::backend::Backend;
use crate::error::EinopsError;
use crate::Operation;
use parse::{Axis, ParsedExpression, ELLIPSIS};

pub enum Function {
    Rearrange,
    Repeat,
    Reduce(Operation),
}

pub struct TransformRecipe {
    elementary_axes_lengths: Vec<Option<usize>>,
    input_composite_axes: Vec<(Vec<usize>, Vec<usize>)>,
    reduced_elementary_axes: Vec<usize>,
    axes_permutation: Vec<usize>,
    added_axes: HashMap<usize, usize>,
    output_composite_axes: Vec<Vec<isize>>,
    reduction_type: Function,
    ellipsis_position_in_lhs: Option<usize>,
}

impl TransformRecipe {
    pub fn new(
        pattern: &str,
        operation: Function,
        axes_lengths: Option<&[(&str, usize)]>,
    ) -> Result<Self, EinopsError> {
        let expressions: Vec<&str> = pattern.split("->").collect();

        let left = ParsedExpression::new(expressions[0])?;
        let right = ParsedExpression::new(expressions[1])?;

        if !left.has_ellipsis && right.has_ellipsis {
            return Err(EinopsError::Pattern(format!(
                "ellipsis found in right side, but not on left side of the pattern {}",
                pattern
            )));
        }

        if left.has_ellipsis_parenthesized {
            return Err(EinopsError::Pattern(format!(
                "ellipsis in parenthesis on the left side is not allowed: {}",
                pattern
            )));
        }

        match operation {
            Function::Rearrange => {
                if left.has_non_unitary_anonymous_axes || right.has_non_unitary_anonymous_axes {
                    return Err(EinopsError::Pattern("non-unitary anonymous axes are not supported in rearrange (exception is length 1)".to_string()));
                }

                let difference: Vec<_> = left
                    .identifiers_named
                    .symmetric_difference(&right.identifiers_named)
                    .collect();

                if !difference.is_empty() {
                    return Err(EinopsError::Pattern(format!(
                        "identifiers only on one side of the expression (should be on both): {:?}",
                        difference
                    )));
                }
            }
            Function::Repeat => {
                let difference: HashSet<_> = left
                    .identifiers_named
                    .difference(&right.identifiers_named)
                    .collect();
                if !difference.is_empty() {
                    return Err(EinopsError::InvalidInput(format!(
                        "unexpected identifiers on the left side of repeat: {:?}",
                        difference
                    )));
                }

                let mut right_side = left.identifiers_named.clone();
                if let Some(axes) = axes_lengths {
                    let temp: HashSet<String> =
                        axes.iter().map(|(name, _)| name.to_string()).collect();
                    right_side = right_side.union(&temp).cloned().collect();
                }
                let axes_without_size: HashSet<_> =
                    right.identifiers_named.difference(&right_side).collect();
                if !axes_without_size.is_empty() {
                    return Err(EinopsError::InvalidInput(format!(
                        "specify sizes for new axes in repeat: {:?}",
                        axes_without_size
                    )));
                }
            }
            Function::Reduce(_) => {
                let difference: HashSet<_> = right
                    .identifiers_named
                    .difference(&left.identifiers_named)
                    .collect();
                if !difference.is_empty() {
                    return Err(EinopsError::InvalidInput(format!(
                        "unexpected identifiers on the right side of reduce: {:?}",
                        difference
                    )));
                }
            }
        }

        let mut axes_len_pos: HashMap<String, (Option<usize>, usize)> = HashMap::new();
        for (pos, axis) in left.composition.iter().flatten().enumerate() {
            let _ = match axis {
                Axis::Named(name) => axes_len_pos.insert(name.clone(), (None, pos)),
                Axis::Anonymous(size) => axes_len_pos.insert(size.to_string(), (Some(*size), pos)),
            };
        }
        for axis in right.composition.iter().flatten() {
            match axis {
                Axis::Named(name) => {
                    if !axes_len_pos.contains_key(name) {
                        let _ = axes_len_pos.insert(name.clone(), (None, axes_len_pos.len()));
                    }
                }
                Axis::Anonymous(size) => {
                    let name = size.to_string();
                    let mut len = 0;

                    axes_len_pos.entry(name).or_insert_with(|| {
                        len += 1;
                        (Some(*size), len - 1)
                    });
                }
            }
        }
        if let Some(axes) = axes_lengths {
            for &(axis, size) in axes {
                if let Some(axis) = axes_len_pos.get_mut(axis) {
                    axis.0 = Some(size);
                } else {
                    return Err(EinopsError::InvalidInput(format!(
                        "axis {} is not used in pattern",
                        axis
                    )));
                }
            }
        }

        let mut reduced_axes: Vec<_> = axes_len_pos
            .iter()
            .filter_map(|(axis, &(_, pos))| {
                // NOTE Anonymous axis is not considered
                if !right.identifiers_named.contains(axis) {
                    return Some(pos);
                }
                None
            })
            .collect();
        reduced_axes.sort_unstable();

        let axes_known_unknown: Vec<(Vec<usize>, Vec<usize>)> = left
            .composition
            .iter()
            .map(|composite_axis| {
                let mut known = vec![];
                let mut unknown = vec![];

                composite_axis.iter().for_each(|axis| match axis {
                    Axis::Named(name) => {
                        let entry = axes_len_pos.get(name).unwrap();
                        if entry.0.is_some() {
                            known.push(entry.1);
                        } else {
                            unknown.push(entry.1);
                        }
                    }
                    Axis::Anonymous(size) => {
                        known.push(axes_len_pos.get(&size.to_string()).unwrap().1);
                    }
                });

                (known, unknown)
            })
            .collect();

        // NOTE -1 indicates ellipsis
        let result_axes_grouping: Vec<Vec<isize>> = right
            .composition
            .iter()
            .map(|composite_axis| {
                composite_axis
                    .iter()
                    .map(|axis| match axis {
                        Axis::Named(name) => {
                            if name.as_str() == ELLIPSIS {
                                return -1;
                            }
                            axes_len_pos.get(name).unwrap().1 as isize
                        }
                        Axis::Anonymous(size) => {
                            axes_len_pos.get(&size.to_string()).unwrap().1 as isize
                        }
                    })
                    .collect()
            })
            .collect();

        let mut axis_pos_after_reduction: HashMap<String, usize> = HashMap::new();
        left.composition
            .iter()
            .flatten()
            .for_each(|axis| match axis {
                Axis::Named(name) => {
                    if right.identifiers_named.contains(name) {
                        axis_pos_after_reduction
                            .insert(name.clone(), axis_pos_after_reduction.len());
                    }
                }
                Axis::Anonymous(_) => {}
            });
        let axes_permutation: Vec<usize> = right
            .composition
            .iter()
            .flatten()
            .filter_map(|axis| match axis {
                Axis::Named(name) => {
                    if left.identifiers_named.contains(name) {
                        return Some(*axis_pos_after_reduction.get(name).unwrap());
                    }
                    None
                }
                Axis::Anonymous(_) => None,
            })
            .collect();

        let added_axes: HashMap<usize, usize> = right
            .composition
            .iter()
            .flatten()
            .enumerate()
            .filter_map(|(i, axis)| {
                let pos = match axis {
                    Axis::Named(name) => {
                        if left.identifiers_named.contains(name) {
                            return None;
                        }
                        axes_len_pos.get(name).unwrap().1
                    }
                    Axis::Anonymous(size) => axes_len_pos.get(&size.to_string()).unwrap().1,
                };

                Some((i, pos))
            })
            .collect();

        let mut elementary_axes_lengths: Vec<(Option<usize>, usize)> =
            axes_len_pos.values().cloned().collect();
        elementary_axes_lengths.sort_by_key(|(_, pos)| *pos);
        let (elementary_axes_lengths, _): (Vec<Option<usize>>, Vec<_>) =
            elementary_axes_lengths.into_iter().unzip();

        let ellipsis_left = if left.has_ellipsis {
            left.composition
                .iter()
                .position(|composite_axis| composite_axis[0] == Axis::Named(ELLIPSIS.to_string()))
        } else {
            None
        };

        Ok(TransformRecipe {
            elementary_axes_lengths,
            input_composite_axes: axes_known_unknown,
            reduced_elementary_axes: reduced_axes,
            axes_permutation,
            added_axes,
            output_composite_axes: result_axes_grouping,
            reduction_type: operation,
            ellipsis_position_in_lhs: ellipsis_left,
        })
    }

    pub fn apply<T: Backend>(&self, tensor: T) -> Result<T, EinopsError> {
        let (init_shapes, added_axes, final_shapes) =
            self.reconstruct_from_shape(tensor.shape())?;

        let mut tensor = tensor.reshape(&init_shapes);

        if !self.reduced_elementary_axes.is_empty() {
            if let Function::Reduce(operation) = self.reduction_type {
                tensor = tensor.reduce(operation, &self.reduced_elementary_axes);
            }
        }

        tensor = tensor.transpose(&self.axes_permutation);

        if !self.added_axes.is_empty() {
            tensor = tensor.add_axes(
                self.axes_permutation.len() + self.added_axes.len(),
                &added_axes,
            );
        }

        Ok(tensor.reshape(&final_shapes))
    }

    fn reconstruct_from_shape(
        &self,
        shape: Vec<usize>,
    ) -> Result<(Vec<usize>, Vec<(usize, usize)>, Vec<usize>), EinopsError> {
        let mut axes_lengths = self.elementary_axes_lengths.clone();

        if self.ellipsis_position_in_lhs.is_some() {
            if shape.len() < self.input_composite_axes.len() - 1 {
                return Err(EinopsError::InvalidInput(format!(
                    "expected atleast {} dimensions, got {}",
                    self.input_composite_axes.len() - 1,
                    shape.len()
                )));
            }
        } else if shape.len() != self.input_composite_axes.len() {
            return Err(EinopsError::InvalidInput(format!(
                "expected {} dimensions, got {}",
                self.input_composite_axes.len(),
                shape.len()
            )));
        }
        let mut ellipsis_shape = 0;

        for (input_axis, (known_axes, unknown_axes)) in
            self.input_composite_axes.iter().enumerate()
        {
            let before_ellipsis = input_axis;
            let after_ellipsis = input_axis + shape.len() - self.input_composite_axes.len();

            if input_axis == self.ellipsis_position_in_lhs.unwrap() {
                ellipsis_shape = shape[before_ellipsis..after_ellipsis + 1].iter().product();

                axes_lengths[unknown_axes[0]] = Some(input_axis);
            } else {
                let length;
                if input_axis < self.ellipsis_position_in_lhs.unwrap() {
                    length = shape[before_ellipsis];
                } else {
                    length = shape[after_ellipsis];
                }
                let mut known_product = 1;
                for axis in known_axes {
                    known_product *= axes_lengths[*axis].unwrap();
                }

                if unknown_axes.is_empty() {
                    if length as usize != known_product {
                        return Err(EinopsError::InvalidInput(format!(
                            "shape mismatch, {} != {}",
                            length, known_product,
                        )));
                    }
                } else {
                    if length as usize % known_product != 0 {
                        return Err(EinopsError::InvalidInput(format!(
                            "shape mismatch, cannot divide axis of length {} into chunks of {}",
                            length, known_product
                        )));
                    }

                    axes_lengths[unknown_axes[0]] = Some(length as usize / known_product);
                }
            }
        }

        let init_shapes = axes_lengths[..(axes_lengths.len() - self.added_axes.len())]
            .iter()
            .map(|size| size.unwrap())
            .collect();

        let final_shapes = self
            .output_composite_axes
            .iter()
            .enumerate()
            .map(|(_, grouping)| {
                if grouping[0] == -1 {
                    ellipsis_shape as usize
                } else {
                    grouping
                        .iter()
                        .fold(1, |acc, pos| acc * axes_lengths[*pos as usize].unwrap())
                }
            })
            .collect();

        let added_axes = self
            .added_axes
            .iter()
            .map(|(pos, pos_in_elementary)| (*pos, axes_lengths[*pos_in_elementary].unwrap()))
            .collect();

        Ok((init_shapes, added_axes, final_shapes))
    }
}
