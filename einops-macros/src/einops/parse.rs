use std::collections::HashMap;

use syn::{parse::ParseStream, token};

mod kw {
    syn::custom_keyword!(min);
    syn::custom_keyword!(max);
    syn::custom_keyword!(sum);
    syn::custom_keyword!(mean);
    syn::custom_keyword!(prod);
}

#[derive(Debug, Clone)]
pub enum Decomposition {
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
    Combined { from: Index, to: Option<Index> },
}

#[derive(Debug, Clone, Eq, PartialOrd)]
pub enum Index {
    Known(usize),
    Unknown(usize),
    Range(usize),
}

impl Ord for Index {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (
                Index::Known(i) | Index::Unknown(i) | Index::Range(i),
                Index::Known(j) | Index::Unknown(j) | Index::Range(j),
            ) => i.cmp(&j),
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

pub fn parse_decomposition(input: ParseStream) -> Vec<Decomposition> {
    let (decomposition, _) = (0..)
        .into_iter()
        .take_while(|_| {
            if input.peek(syn::Token![->]) {
                input.parse::<syn::Token![->]>().unwrap();
                return false;
            }
            true
        })
        .fold(
            (
                Vec::new(),
                Box::new(Index::Known) as Box<dyn Fn(usize) -> Index>,
            ),
            |(mut decomposition, mut index_fn), i| {
                if input.peek(syn::token::Paren) {
                    let content_expression = parse_left_parenthesized(input, index_fn(i)).unwrap();
                    decomposition.extend(content_expression);
                } else if peek_reduce_kw(input) {
                    let (name, shape, operation) = parse_reduce_fn(input).unwrap();
                    decomposition.push(Decomposition::Named {
                        name,
                        index: index_fn(i),
                        shape,
                        operation: Some(operation),
                    });
                } else if input.peek(syn::Ident) {
                    let (name, shape) = parse_identifier(input).unwrap();
                    decomposition.push(Decomposition::Named {
                        name,
                        shape,
                        index: index_fn(i),
                        operation: None,
                    });
                } else if input.peek(syn::LitInt) {
                    todo!("parse_int");
                } else if input.peek(syn::Token![..]) {
                    input.parse::<syn::Token![..]>().unwrap();
                    decomposition.push(Decomposition::Named {
                        name: "..".to_string(),
                        index: Index::Range(i),
                        shape: None,
                        operation: None,
                    });
                    index_fn = Box::new(Index::Unknown);
                }
                (decomposition, index_fn)
            },
        );

    decomposition
}

fn parse_left_parenthesized(input: ParseStream, index: Index) -> syn::Result<Vec<Decomposition>> {
    let content;
    syn::parenthesized!(content in input);

    let mut content_expression = Vec::new();

    let (derived_name, derived_index, running_mul) =
        (0..).into_iter().take_while(|_| !content.is_empty()).fold(
            (None, None, 1),
            |(mut derived_name, mut derived_index, mut running_mul), i| {
                let (name, shape, operation) = if peek_reduce_kw(&content) {
                    let (name, shape, operation) = parse_reduce_fn(&content).unwrap();
                    (name, shape, Some(operation))
                } else if content.peek(syn::Ident) {
                    let (name, shape) = parse_identifier(&content).unwrap();
                    (name, shape, None)
                } else {
                    todo!();
                };
                if let Some(size) = shape {
                    running_mul *= size;
                    content_expression.push(Decomposition::Named {
                        name,
                        index: index.clone(),
                        operation,
                        shape: Some(size),
                    });
                } else {
                    derived_name = Some(name.clone());
                    derived_index = Some(i);
                }
                (derived_name, derived_index, running_mul)
            },
        );

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
            _ => None,
        })
        .collect::<Vec<_>>()
}

pub fn parse_composition_permute_repeat(
    input: ParseStream,
    decomposition: &Vec<Decomposition>,
) -> (Vec<Composition>, Vec<Index>, Vec<(Index, usize)>) {
    let positions = decomposition
        .iter()
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
        .fold(HashMap::new(), |mut map, (i, expression)| {
            match expression {
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
                } => map.insert(name.clone(), Index::Unknown(i)),
                Decomposition::Named {
                    name,
                    index: Index::Range(_),
                    ..
                } => map.insert(name.clone(), Index::Range(i)),
                _ => todo!(),
            };
            map
        });

    let mut parenthesized_len: usize = 0;
    let (composition, permute, repeat, _) =
        (0..).into_iter().take_while(|_| !input.is_empty()).fold(
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Box::new(Index::Known) as Box<dyn Fn(usize) -> Index>,
            ),
            |(mut composition, mut permute, mut repeat, mut index_fn), mut i| {
                i += parenthesized_len.saturating_sub(1);
                if input.peek(token::Paren) {
                    let (combined, combined_permute, combined_repeat, combined_len) =
                        parse_right_parenthesized(input, i, &mut index_fn, &positions).unwrap();
                    parenthesized_len += combined_len;
                    permute.extend(combined_permute);
                    repeat.extend(combined_repeat);
                    composition.push(combined);
                } else if input.peek(syn::Ident) {
                    let (name, shape) = parse_identifier(input).unwrap();
                    if let Some(index) = positions.get(&name) {
                        permute.push(index.clone());
                    } else {
                        repeat.push((index_fn(i), shape.unwrap()));
                    }
                    composition.push(Composition::Individual(index_fn(i)))
                } else if input.peek(syn::LitInt) {
                    repeat.push((index_fn(i), parse_usize(input).unwrap()));
                    composition.push(Composition::Individual(index_fn(i)));
                } else if input.peek(syn::Token![..]) {
                    input.parse::<syn::Token![..]>().unwrap();
                    composition.push(Composition::Individual(Index::Range(i)));
                    permute.push(positions.get("..").unwrap().clone());
                    index_fn = Box::new(Index::Unknown);
                }
                (composition, permute, repeat, index_fn)
            },
        );
    (composition, permute, repeat)
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

    let mut parse_content = |content: ParseStream, index: usize| -> Index {
        if content.peek(syn::Token![..]) {
            content.parse::<syn::Token![..]>().unwrap();
            permute.push(positions.get("..").unwrap().clone());
            *index_fn = Box::new(Index::Unknown);
            Index::Range(index)
        } else if content.peek(syn::Ident) {
            let (name, shape) = parse_identifier(&content).unwrap();
            if let Some(index) = positions.get(&name) {
                permute.push(index.clone());
            } else {
                repeat.push((index_fn(index), shape.unwrap()));
            }
            index_fn(index)
        } else if content.peek(syn::LitInt) {
            repeat.push((index_fn(index), parse_usize(content).unwrap()));
            index_fn(index)
        } else {
            todo!();
        }
    };

    let from = parse_content(&content, start_index);

    let to = ((start_index + 1)..)
        .into_iter()
        .take_while(|_| !content.is_empty())
        .fold(None, |_, i| Some(parse_content(&content, i)));

    let len = if let Some(Index::Known(end_index) | Index::Unknown(end_index)) = to {
        end_index - (start_index - 1)
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

fn parse_reduce_fn(input: ParseStream) -> syn::Result<(String, Option<usize>, Operation)> {
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
        todo!();
    };

    let content;
    syn::parenthesized!(content in input);

    let (name, shape) = parse_identifier(&content)?;

    Ok((name, shape, operation))
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
