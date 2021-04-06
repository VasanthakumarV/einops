mod parse;

use std::collections::{HashMap, HashSet};

use crate::backend::Backend;
use crate::error::EinopsError;
use crate::Operation;
use parse::{Axis, ParsedExpression, ELLIPSIS};

#[derive(Debug)]
pub enum Function {
    Rearrange,
    Repeat,
    Reduce(Operation),
}

#[derive(Debug)]
pub struct TransformRecipe {
    // List of available size info
    elementary_axes_lengths: Vec<Option<usize>>,
    // Separates known and unknown portions of the input axis
    input_composite_axes: Vec<(Vec<usize>, Vec<usize>)>,
    // Positions of axis that get reduced
    reduced_elementary_axes: Vec<usize>,
    // Permutation info of reduced tensor before adding any new axis
    axes_permutation: Vec<usize>,
    // Position where the new axes should appear and their size
    //added_axes: HashMap<usize, usize>,
    added_axes: Vec<(usize, usize)>,
    // Positions of output axis as per `elementary_axes_lengths`
    output_composite_axes: Vec<Vec<usize>>,
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

        let mut left = ParsedExpression::new(expressions[0])?;
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

        // Vec to store list of axis positions that get reduced
        let mut reduced_axes: Vec<usize> = vec![];

        // Helper hashmap that stores the axes names that don't get reduced
        let mut axis_pos_after_reduction: HashMap<String, usize> = HashMap::new();

        // We convert the slice into hashmap for easy search
        let mut axes_lengths_hash: HashMap<String, usize> = HashMap::new();
        if let Some(axes) = axes_lengths {
            axes.iter().for_each(|(name, size)| {
                let _ = axes_lengths_hash.insert(name.to_string(), *size);
            });
        }

        left.composition
            .iter_mut()
            .flatten()
            .enumerate()
            .for_each(|(pos, axis)| {
                // If an axis length is provided, we update its size in the
                // `left.composition`
                if let Some(size) = axes_lengths_hash.get(&axis.name) {
                    axis.size = Some(*size);
                }

                // Intially all `pos` values will be 0, we rectify that by
                // updating them
                axis.pos = pos;

                if !right.identifiers_named.contains(&axis.name) {
                    // If an axis is not available, we note down its position
                    reduced_axes.push(pos);
                } else {
                    axis_pos_after_reduction
                        .insert(axis.name.clone(), axis_pos_after_reduction.len());
                }
            });

        // We create a hashmap out of `left.composition`, this will help us later
        // on, when we search for position using the axis name
        let mut axes_pos: HashMap<String, usize> = left
            .composition
            .iter()
            .flatten()
            .map(|axis| (axis.name.clone(), axis.pos))
            .collect();

        // Variable to store the position of ellipsis, if available
        let mut ellipsis_left: Option<usize> = None;

        // List of lists to store the positions of known and unknown components
        // of each input axis
        let axes_known_unknown: Vec<(Vec<usize>, Vec<usize>)> = left
            .composition
            .iter()
            .enumerate()
            .map(|(i, composite_axis)| {
                let mut known = vec![];
                let mut unknown = vec![];

                // Update `ellipsis_left`
                if !composite_axis.is_empty() && composite_axis[0].name == *ELLIPSIS {
                    ellipsis_left = Some(i);
                }

                composite_axis.iter().for_each(|axis| {
                    if axis.size.is_some() {
                        known.push(axis.pos);
                    } else {
                        unknown.push(axis.pos);
                    }
                });

                (known, unknown)
            })
            .collect();

        // Variable that tells us how to permute the input tensor after reduction
        // but before adding new axis
        let mut axes_permutation: Vec<usize> = vec![];

        // Stores the position and size of new axis that needs to be added
        let mut added_axes: Vec<(usize, usize)> = vec![];

        right
            .composition
            .iter()
            .flatten()
            .enumerate()
            .for_each(|(i, axis)| {
                if left.identifiers_named.contains(&axis.name) {
                    if let Some(value) = axis_pos_after_reduction.get(&axis.name) {
                        // Stores index values in the order they appear in the output
                        // tensor
                        axes_permutation.push(*value);
                    }
                } else {
                    let pos = axes_pos.len();

                    // We update `axes_pos` with new axis info
                    axes_pos.insert(axis.name.clone(), pos);

                    added_axes.push((i, pos));

                    // We update `left.composition` with the information about
                    // the new axis that needs to be added in the output
                    let size: Option<usize>;
                    if let Some(value) = axis.size {
                        size = Some(value)
                    } else {
                        size = axes_lengths_hash.get(&axis.name).copied();
                    }
                    left.composition.push(vec![Axis {
                        name: axis.name.clone(),
                        size,
                        pos,
                    }]);
                }
            });

        let result_axes_grouping: Vec<Vec<usize>> = right
            .composition
            .iter()
            .map(|composite_axis| {
                composite_axis
                    .iter()
                    .map(|axis| {
                        if axis.name.as_str() == ELLIPSIS && !right.has_ellipsis_parenthesized {
                            return usize::MAX;
                        }
                        // `unwrap` is safe, because `axes_pos` should have all the
                        // unique identifiers from both left and right side of the pattern
                        *axes_pos.get(&axis.name).unwrap()
                    })
                    .collect()
            })
            .collect();

        // We store all known and unknown size info we have into a vector,
        // to preserve the order
        let elementary_axes_lengths: Vec<Option<usize>> = left
            .composition
            .iter()
            .flatten()
            .map(|axis| axis.size)
            .collect();

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

    pub fn apply<T: Backend>(&self, tensor: &T) -> Result<T, EinopsError> {
        let (init_shapes, added_axes, final_shapes) =
            self.reconstruct_from_shape(tensor.shape())?;

        let mut tensor = tensor.reshape(&init_shapes);

        // We reduce the necessary axes
        if !self.reduced_elementary_axes.is_empty() {
            if let Function::Reduce(operation) = self.reduction_type {
                tensor = tensor.reduce_axes(operation, &self.reduced_elementary_axes);
            }
        }

        // We permute the tensor
        tensor = tensor.transpose(&self.axes_permutation);

        // We add new axes
        if !self.added_axes.is_empty() {
            tensor = tensor.add_axes(
                self.axes_permutation.len() + self.added_axes.len(),
                &added_axes,
            );
        }

        Ok(tensor.reshape(&final_shapes))
    }

    #[allow(clippy::type_complexity)]
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

        // In this section, we track all axes sizes covered by the ellipsis entry,
        // and we fill any missing size information in `axes_lengths` list
        let mut ellipsis_shape: Vec<usize> = vec![];
        for (input_axis, (known_axes, unknown_axes)) in self.input_composite_axes.iter().enumerate()
        {
            let before_ellipsis = input_axis;
            let after_ellipsis =
                (input_axis + shape.len()).saturating_sub(self.input_composite_axes.len());

            if Some(input_axis) == self.ellipsis_position_in_lhs {
                if (before_ellipsis == after_ellipsis)
                    && (self.input_composite_axes.len() > shape.len())
                {
                    ellipsis_shape.extend(shape[before_ellipsis..after_ellipsis].iter().copied());
                } else {
                    ellipsis_shape
                        .extend(shape[before_ellipsis..after_ellipsis + 1].iter().copied());
                }

                axes_lengths[unknown_axes[0]] = Some(ellipsis_shape.iter().product());
            } else {
                let length;
                if Some(input_axis) < self.ellipsis_position_in_lhs {
                    length = shape[before_ellipsis];
                } else {
                    length = shape[after_ellipsis];
                }

                let mut known_product = 1;
                for axis in known_axes {
                    // `unwrap` is safe, because `known_axes` entries are created
                    // by checking for existence of size
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
            // `unwrap` is safe, because at the this, we should have already
            // figured all the size values
            .map(|size| size.unwrap())
            .collect();

        let final_shapes = self
            .output_composite_axes
            .iter()
            .fold(vec![], |mut acc, grouping| {
                if grouping.is_empty() {
                    acc.push(1);
                } else if grouping[0] == usize::MAX {
                    acc.extend(ellipsis_shape.iter().copied());
                } else {
                    acc.push(
                        grouping
                            .iter()
                            // `unwrap` is safe, because at the this, we should have already
                            // figured all the size values
                            .fold(1, |acc, pos| acc * axes_lengths[*pos].unwrap()),
                    );
                }
                acc
            });

        let added_axes: Vec<(usize, usize)> = self
            .added_axes
            .iter()
            // `unwrap` is safe, because at the this, we should have already
            // figured all the size values
            .map(|(pos, pos_in_elementary)| (*pos, axes_lengths[*pos_in_elementary].unwrap()))
            .collect();

        Ok((init_shapes, added_axes, final_shapes))
    }
}
