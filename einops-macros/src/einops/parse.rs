use std::collections::HashMap;

use syn::{parse::ParseStream, token};

// Custom keywords to represent reduce operations
mod kw {
    syn::custom_keyword!(min);
    syn::custom_keyword!(max);
    syn::custom_keyword!(sum);
    syn::custom_keyword!(mean);
    syn::custom_keyword!(prod);
}

#[derive(Debug, Clone)]
pub enum Decomposition {
    // New decomposed dimension with no shape provided
    // by the user
    Derived {
        name: String,
        index: Index,
        operation: Option<Operation>,
        shape_calc: usize,
    },
    Named {
        name: String,
        index: Index,
        operation: Option<Operation>,
        shape: Option<usize>,
    },
}

#[derive(Debug)]
pub enum Composition {
    Individual(Index),
    // Start and end index of dimesions to be combined
    // into one
    Combined { from: Index, to: Option<Index> },
}

#[derive(Debug, Clone)]
pub enum Index {
    Known(usize),
    // Position of dimension after '..' symbol
    Unknown(usize),
    // Position of the '..' symbol
    Range(usize),
}

impl PartialOrd for Index {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (
                Index::Known(i) | Index::Unknown(i) | Index::Range(i),
                Index::Known(j) | Index::Unknown(j) | Index::Range(j),
            ) => Some(i.cmp(&j)),
        }
    }
}

impl PartialEq for Index {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Index::Known(i) | Index::Unknown(i) | Index::Range(i),
                Index::Known(j) | Index::Unknown(j) | Index::Range(j),
            ) => i == j,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Operation {
    Min,
    Max,
    Sum,
    Mean,
    Prod,
}

pub fn parse_decomposition(input: ParseStream) -> syn::Result<(Vec<Decomposition>, bool)> {
    // Length of dimensions inside parenthesis,
    // we need it to account for the dimensions skipped during the iteration
    let mut parenthesized_len = 0;
    let (decomposition, requires_decomposition, _) = (0..)
        .into_iter()
        .take_while(|_| {
            // We parse till we reach '->'
            if input.peek(syn::Token![->]) {
                input.parse::<syn::Token![->]>().unwrap();
                return false;
            }
            true
        })
        .try_fold(
            (
                Vec::new(),
                // Boolean to indicate if,
                // - dimensions are decomposed
                // - squeezing is required
                false,
                // Closure that helps construct `Index`,
                // it is updated once we hit '..' to construct `Unknown` indices
                Box::new(Index::Known) as Box<dyn Fn(usize) -> Index>,
            ),
            |(mut decomposition, mut requires_decomposition, mut index_fn), mut i| {
                // We account for dimensions skipped when we parse
                // contents of parenthesized expression
                i += parenthesized_len;
                if input.peek(syn::token::Paren) {
                    let content_expression = parse_left_parenthesized(input, index_fn(i))?;
                    decomposition.extend(content_expression);
                    requires_decomposition = true;
                } else if peek_reduce_kw(input) {
                    let identifiers = parse_reduce_fn(input)?;
                    // We account for dimensions skipped during reduction operations
                    // like `sum(a b c)`, where three dimensions are reduced
                    parenthesized_len += identifiers.len().saturating_sub(1);
                    identifiers.into_iter().enumerate().for_each(
                        |(inner_index, (name, shape, operation))| {
                            // If we encounter '..' inside reduce, we will have to update the closure
                            if name == ".." {
                                index_fn = Box::new(Index::Unknown);
                                decomposition.push(Decomposition::Named {
                                    name,
                                    index: Index::Range(i + inner_index),
                                    shape,
                                    operation: Some(operation),
                                });
                            } else {
                                decomposition.push(Decomposition::Named {
                                    name,
                                    index: index_fn(i + inner_index),
                                    shape,
                                    operation: Some(operation),
                                });
                            }
                        },
                    );
                } else if input.peek(syn::Ident) {
                    let (name, shape) = parse_identifier(input)?;
                    decomposition.push(Decomposition::Named {
                        name,
                        shape,
                        index: index_fn(i),
                        operation: None,
                    });
                } else if input.peek(syn::LitInt) {
                    let lit_int = input.parse::<syn::LitInt>()?;
                    if lit_int.base10_parse::<usize>()? != 1 {
                        // '1' is the only literal int allowed on the left,
                        // this indicates a squeeze operation
                        return Err(input.error(format!(
                            "Literal Int {} not allowed on the left side",
                            lit_int.to_string()
                        )));
                    }
                    // We have to reshape to squeeze the 1 sized dimension
                    requires_decomposition = true;
                } else if input.peek(syn::Token![..]) {
                    input.parse::<syn::Token![..]>()?;
                    decomposition.push(Decomposition::Named {
                        name: "..".to_string(),
                        index: Index::Range(i),
                        shape: None,
                        operation: None,
                    });
                    // We update the closure as we have encountered '..'
                    index_fn = Box::new(Index::Unknown);
                } else {
                    return Err(input
                        .error("Unrecognized charater found in the left side of the expression"));
                }
                Ok((decomposition, requires_decomposition, index_fn))
            },
        )?;

    Ok((decomposition, requires_decomposition))
}

fn parse_left_parenthesized(input: ParseStream, index: Index) -> syn::Result<Vec<Decomposition>> {
    let content;
    syn::parenthesized!(content in input);

    let mut content_expression = Vec::new();

    let (derived_name, derived_index, running_mul) = (0..)
        .into_iter()
        // We continue till we parse everything inside the parenthesis
        .take_while(|_| !content.is_empty())
        .try_fold(
            (None, None, 1),
            |(mut derived_name, mut derived_index, mut running_mul), i| {
                // Closure to keep a running multiple of the shapes,
                // and updating the list with new dimensions with known shape
                let mut update_values = |name, shape, operation| {
                    if let Some(size) = shape {
                        running_mul *= size;
                        content_expression.push(Decomposition::Named {
                            name,
                            index: index.clone(),
                            operation,
                            shape: Some(size),
                        });
                    } else {
                        if derived_name.is_some() {
                            // We cannot have more than one dimension with unknown shape
                            return Err(content
                                .error("Shape information required to complete decomposition"));
                        }
                        derived_name = Some(name.clone());
                        derived_index = Some(i);
                    }
                    Ok(())
                };
                if peek_reduce_kw(&content) {
                    parse_reduce_fn(&content)?.into_iter().try_for_each(
                        |(name, shape, operation)| {
                            if name == ".." {
                                return Err(content.error(
                                    "Ignore symbol '..' not allowed inside brackets on the left",
                                ));
                            }
                            update_values(name, shape, Some(operation))
                        },
                    )?;
                } else if content.peek(syn::Ident) {
                    let (name, shape) = parse_identifier(&content)?;
                    update_values(name, shape, None)?
                } else if content.peek(syn::Token![..]) {
                    return Err(
                        content.error("Ignore symbol '..' not allowed inside brackets on the left")
                    );
                } else if content.peek(syn::LitInt) {
                    let lit_int = content.parse::<syn::LitInt>()?;
                    return Err(content.error(format!(
                        "Anonymous integer {} is not allowed inside brackets on the left",
                        lit_int.to_string()
                    )));
                } else {
                    return Err(content.error(
                        "Unknown character found inside the brackets of the left expression",
                    ));
                };
                Ok((derived_name, derived_index, running_mul))
            },
        )?;

    // We add the `Derived` dimension at the end at its index,
    // once we have the running multiple of all the other shapes
    // inside the parenthesis
    if let Some(derived_index) = derived_index {
        content_expression.insert(
            derived_index,
            Decomposition::Derived {
                name: derived_name.unwrap(),
                index,
                operation: None,
                shape_calc: running_mul,
            },
        );
    }

    Ok(content_expression)
}

pub fn parse_reduce(decomposition: &Vec<Decomposition>) -> Vec<(Index, Operation)> {
    // We filter for only the dimensions that have some type of
    // operation associated with them
    decomposition
        .iter()
        .cloned()
        .enumerate()
        .filter_map(|(i, expression)| match expression {
            Decomposition::Named {
                index: Index::Known(_),
                operation: Some(operation),
                ..
            } => Some((Index::Known(i), operation)),
            Decomposition::Named {
                index: Index::Unknown(_),
                operation: Some(operation),
                ..
            } => Some((Index::Unknown(i), operation)),
            Decomposition::Named {
                index: Index::Range(_),
                operation: Some(operation),
                ..
            } => Some((Index::Range(i), operation)),
            _ => None,
        })
        .collect::<Vec<_>>()
}

pub fn parse_composition_permute_repeat(
    input: ParseStream,
    decomposition: &Vec<Decomposition>,
) -> syn::Result<(Vec<Composition>, Vec<Index>, Vec<(Index, usize)>)> {
    // We calculate the span to report errors later
    let input_span = input.span();
    // We check if ignored dimensions are reduced
    let is_ignore_reduced = decomposition.iter().any(|expression| {
        matches!(expression, Decomposition::Named {name, operation: Some(_), ..} if name.as_str() == "..")
    });
    // If '..' is reduced, previously unknown indices could change
    // to known
    let unknown_index_fn = |i| {
        if is_ignore_reduced {
            return Index::Known(i);
        }
        Index::Unknown(i)
    };
    // We create a hashmap of identifiers and their positions
    // in the left expression
    let positions = decomposition
        .iter()
        // We ignore all the reduced dimensions
        .filter(|expression| {
            !matches!(
                expression,
                Decomposition::Named {
                    operation: Some(_),
                    ..
                }
            )
        })
        .enumerate()
        .try_fold(HashMap::new(), |mut map, (i, expression)| {
            let old_value = match expression {
                Decomposition::Named {
                    name,
                    index: Index::Known(_),
                    ..
                }
                | Decomposition::Derived {
                    name,
                    index: Index::Known(_),
                    ..
                } => map.insert(name.clone(), Index::Known(i)),
                Decomposition::Named {
                    name,
                    index: Index::Unknown(_),
                    ..
                }
                | Decomposition::Derived {
                    name,
                    index: Index::Unknown(_),
                    ..
                } => map.insert(name.clone(), unknown_index_fn(i)),
                Decomposition::Named {
                    name,
                    index: Index::Range(_),
                    ..
                } => map.insert(name.clone(), Index::Range(i)),
                _ => unreachable!(),
            };
            if let Some(_) = old_value {
                return Err(input.error("Names are not unique in the left expression"));
            }
            Ok(map)
        })?;

    // To keep track of the dimensions skipped inside parenthesis
    let mut parenthesized_len: usize = 0;
    let (composition, permute, repeat, _) = (0..)
        .into_iter()
        .take_while(|_| !input.is_empty())
        .try_fold::<_, _, syn::Result<(_, _, _, _)>>(
        (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            // Closure to construct `Index`, once we encounter '..',
            // `Known` index becomes `Unknown`
            Box::new(Index::Known) as Box<dyn Fn(usize) -> Index>,
        ),
        |(mut composition, mut permute, mut repeat, mut index_fn), mut i| {
            // We update the iterator's index to account for dimensions skipped
            // inside parenthesis
            i += parenthesized_len;
            if input.peek(token::Paren) {
                let (combined, combined_permute, combined_repeat, combined_len) =
                    parse_right_parenthesized(input, i, &mut index_fn, &positions)?;
                parenthesized_len += combined_len.saturating_sub(1);
                permute.extend(combined_permute);
                repeat.extend(combined_repeat);
                composition.push(combined);
            } else if input.peek(syn::Ident) {
                let (name, shape) = parse_identifier(input)?;
                if let Some(index) = positions.get(&name) {
                    permute.push(index.clone());
                } else {
                    // New identifiers represents repetition
                    repeat.push((
                        index_fn(i),
                        shape.expect("New identifier on the right should have a shape"),
                    ));
                }
                composition.push(Composition::Individual(index_fn(i)))
            } else if input.peek(syn::LitInt) {
                // Literal ints represent repetition
                repeat.push((index_fn(i), parse_usize(input)?));
                composition.push(Composition::Individual(index_fn(i)));
            } else if input.peek(syn::Token![..]) {
                input.parse::<syn::Token![..]>()?;
                composition.push(Composition::Individual(Index::Range(i)));
                permute.push(
                    positions
                        .get("..")
                        .expect("Ignore should be on both sides of the expression")
                        .clone(),
                );
                // We update the closure
                index_fn = Box::new(Index::Unknown);
            } else {
                return Err(
                    input.error("Unrecognized character on the right side of the expression")
                );
            }
            Ok((composition, permute, repeat, index_fn))
        },
    )?;

    // We raise an error if the right side of the expression
    // misses a indicator from the left side
    if positions.len() != permute.len() {
        return Err(syn::Error::new(
            input_span,
            "Identifiers missing on the right side of the expression",
        ));
    }

    Ok((composition, permute, repeat))
}

fn parse_right_parenthesized(
    input: ParseStream,
    start_index: usize,
    index_fn: &mut Box<dyn Fn(usize) -> Index>,
    positions: &HashMap<String, Index>,
) -> syn::Result<(Composition, Vec<Index>, Vec<(Index, usize)>, usize)> {
    let content;
    syn::parenthesized!(content in input);

    let mut permute = Vec::new();
    let mut repeat = Vec::new();

    // Closure to parse one entry in the expression and update
    // the relevant lists
    let mut parse_content = |content: ParseStream, index: usize| -> syn::Result<Index> {
        if content.peek(syn::Token![..]) {
            content.parse::<syn::Token![..]>()?;
            permute.push(
                positions
                    .get("..")
                    .expect("Ignore should be on both sides of the expressions")
                    .clone(),
            );
            *index_fn = Box::new(Index::Unknown);
            Ok(Index::Range(index))
        } else if content.peek(syn::Ident) {
            let (name, shape) = parse_identifier(&content)?;
            if let Some(index) = positions.get(&name) {
                permute.push(index.clone());
            } else {
                repeat.push((
                    index_fn(index),
                    shape.expect("New identifier with no shape specified on the right side"),
                ));
            }
            Ok(index_fn(index))
        } else if content.peek(syn::LitInt) {
            repeat.push((index_fn(index), parse_usize(content)?));
            Ok(index_fn(index))
        } else {
            return Err(input.error("Unrecognized character on the right side of the expression"));
        }
    };

    // The starting index of the combined dimensions
    let from = parse_content(&content, start_index)?;

    // The ending index of the combined dimension,
    // we iterate through the entire expression to update relevant lists
    let to = ((start_index + 1)..)
        .into_iter()
        .take_while(|_| !content.is_empty())
        .fold(None, |_, i| Some(parse_content(&content, i)))
        .transpose()?;

    // We calculate the length of dimesions inside this parenthesis,
    // this helps the main loop keep track of skipped dimensions
    let len = if let Some(
        Index::Known(end_index) | Index::Unknown(end_index) | Index::Range(end_index),
    ) = to
    {
        (end_index - start_index) + 1
    } else {
        0
    };

    Ok((Composition::Combined { from, to }, permute, repeat, len))
}

fn peek_reduce_kw(input: ParseStream) -> bool {
    input.peek(kw::min)
        | input.peek(kw::max)
        | input.peek(kw::sum)
        | input.peek(kw::mean)
        | input.peek(kw::prod)
}

fn parse_reduce_fn(input: ParseStream) -> syn::Result<Vec<(String, Option<usize>, Operation)>> {
    let operation = if input.peek(kw::min) {
        input.parse::<kw::min>()?;
        Operation::Min
    } else if input.peek(kw::max) {
        input.parse::<kw::max>()?;
        Operation::Max
    } else if input.peek(kw::sum) {
        input.parse::<kw::sum>()?;
        Operation::Sum
    } else if input.peek(kw::mean) {
        input.parse::<kw::mean>()?;
        Operation::Mean
    } else if input.peek(kw::prod) {
        input.parse::<kw::prod>()?;
        Operation::Prod
    } else {
        unreachable!();
    };

    let content;
    syn::parenthesized!(content in input);

    Ok(content
        // A single operation call can have multiple dimensions,
        .call(parse_identifiers)?
        .into_iter()
        .map(|(name, shape)| (name, shape, operation.clone()))
        .collect())
}

fn parse_identifiers(content: ParseStream) -> syn::Result<Vec<(String, Option<usize>)>> {
    let mut identifiers = Vec::new();
    while !content.is_empty() {
        if content.peek(syn::Ident) {
            let (name, shape) = content.call(parse_identifier)?;
            identifiers.push((name, shape));
        } else if content.peek(syn::Token![..]) {
            content.parse::<syn::Token![..]>()?;
            identifiers.push(("..".to_string(), None));
        } else if content.peek(syn::LitInt) {
            let lit_int = parse_usize(content)?;
            identifiers.push((lit_int.to_string(), Some(lit_int)));
        } else {
            return Err(content.error("Unknown character introduced in the reduce operation"));
        }
    }
    Ok(identifiers)
}

fn parse_identifier(input: ParseStream) -> syn::Result<(String, Option<usize>)> {
    let name = input.parse::<syn::Ident>()?.to_string();

    let shape = if input.peek(syn::Token![:]) {
        input.parse::<syn::Token![:]>()?;
        Some(parse_usize(input)?)
    } else {
        None
    };

    Ok((name, shape))
}

fn parse_usize(input: ParseStream) -> syn::Result<usize> {
    let len = input.parse::<syn::LitInt>()?;
    Ok(len.base10_parse::<usize>()?)
}
