#[proc_macro]
pub fn rearrange(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    rearrange::rearrange(input.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

mod rearrange {
    mod kw {
        syn::custom_keyword!(min);
        syn::custom_keyword!(max);
        syn::custom_keyword!(sum);
        syn::custom_keyword!(mean);
        syn::custom_keyword!(prod);
    }

    use std::collections::HashMap;

    use quote::{format_ident, quote};
    use syn::{parse::ParseStream, token};

    pub fn rearrange(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
        let parsed_expression: ParsedExpression = syn::parse2(input)?;
        let code = quote! { #parsed_expression};
        Ok(code)
    }

    #[derive(Debug)]
    struct ParsedExpression {
        tensor: syn::Ident,
        expression: Expression,
    }

    impl syn::parse::Parse for ParsedExpression {
        fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
            let expression: Expression = input.parse::<syn::LitStr>()?.parse()?;
            input.parse::<syn::Token![,]>()?;

            Ok(Self {
                tensor: input.parse::<syn::Ident>()?,
                expression,
            })
        }
    }

    #[derive(Debug)]
    struct Expression {
        left_expression: Vec<LeftExpression>,
        reduce: Vec<(Index, Operation)>,
        permute: Vec<Index>,
        repeat: Vec<(Index, usize)>,
        right_expression: Vec<RightExpression>,
    }

    #[derive(Debug, Clone)]
    enum LeftExpression {
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
    enum RightExpression {
        Individual(Index),
        Combined { from: Index, to: Option<Index> },
    }

    #[derive(Debug, Clone)]
    enum Index {
        Known(usize),
        Unknown(usize),
        Range(usize),
    }

    #[derive(Debug, Clone)]
    enum Operation {
        Min,
        Max,
        Sum,
        Mean,
        Prod,
    }

    impl syn::parse::Parse for Expression {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let (left_expression, _) = (0..)
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
                    |(mut left_expression, mut index_fn), i| {
                        if input.peek(syn::token::Paren) {
                            let content_expression =
                                parse_left_parenthesized(input, index_fn(i)).unwrap();
                            left_expression.extend(content_expression);
                        } else if peek_reduce_kw(input) {
                            let (name, shape, operation) = parse_reduce_fn(input).unwrap();
                            left_expression.push(LeftExpression::Named {
                                name,
                                index: index_fn(i),
                                shape,
                                operation: Some(operation),
                            });
                        } else if input.peek(syn::Ident) {
                            let (name, shape) = parse_identifier(input).unwrap();
                            left_expression.push(LeftExpression::Named {
                                name,
                                shape,
                                index: index_fn(i),
                                operation: None,
                            });
                        } else if input.peek(syn::LitInt) {
                            todo!("parse_int");
                        } else if input.peek(syn::Token![..]) {
                            input.parse::<syn::Token![..]>().unwrap();
                            left_expression.push(LeftExpression::Named {
                                name: "..".to_string(),
                                index: Index::Range(i),
                                shape: None,
                                operation: None,
                            });
                            index_fn = Box::new(Index::Unknown);
                        }
                        (left_expression, index_fn)
                    },
                );

            let reduce = left_expression
                .iter()
                .cloned()
                .enumerate()
                .filter_map(|(i, expression)| match expression {
                    LeftExpression::Named {
                        index: Index::Known(_),
                        operation: Some(operation),
                        ..
                    } => Some((Index::Known(i), operation)),
                    LeftExpression::Named {
                        index: Index::Unknown(_),
                        operation: Some(operation),
                        ..
                    } => Some((Index::Unknown(i), operation)),
                    _ => None,
                })
                .collect::<Vec<_>>();

            let positions = left_expression
                .iter()
                .filter(|expression| {
                    !matches!(
                        expression,
                        LeftExpression::Named {
                            operation: Some(_),
                            ..
                        }
                    )
                })
                .enumerate()
                .fold(HashMap::new(), |mut map, (i, expression)| {
                    match expression {
                        LeftExpression::Named {
                            name,
                            index: Index::Known(_),
                            ..
                        }
                        | LeftExpression::Derived {
                            name,
                            index: Index::Known(_),
                            ..
                        } => map.insert(name.clone(), Index::Known(i)),
                        LeftExpression::Named {
                            name,
                            index: Index::Unknown(_),
                            ..
                        }
                        | LeftExpression::Derived {
                            name,
                            index: Index::Unknown(_),
                            ..
                        } => map.insert(name.clone(), Index::Unknown(i)),
                        LeftExpression::Named {
                            name,
                            index: Index::Range(_),
                            ..
                        } => map.insert(name.clone(), Index::Range(i)),
                        _ => todo!(),
                    };
                    map
                });

            let mut parenthesized_len: usize = 0;
            let (right_expression, permute, repeat, _) =
                (0..).into_iter().take_while(|_| !input.is_empty()).fold(
                    (
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Box::new(Index::Known) as Box<dyn Fn(usize) -> Index>,
                    ),
                    |(mut right_expression, mut permute, mut repeat, mut index_fn), mut i| {
                        i += parenthesized_len.saturating_sub(1);
                        if input.peek(token::Paren) {
                            let (combined, combined_permute, combined_repeat, combined_len) =
                                parse_right_parenthesized(input, i, &mut index_fn, &positions)
                                    .unwrap();
                            parenthesized_len += combined_len;
                            permute.extend(combined_permute);
                            repeat.extend(combined_repeat);
                            right_expression.push(combined);
                        } else if input.peek(syn::Ident) {
                            let (name, shape) = parse_identifier(input).unwrap();
                            if let Some(index) = positions.get(&name) {
                                permute.push(index.clone());
                            } else {
                                repeat.push((index_fn(i), shape.unwrap()));
                            }
                            right_expression.push(RightExpression::Individual(index_fn(i)))
                        } else if input.peek(syn::LitInt) {
                            repeat.push((index_fn(i), parse_usize(input).unwrap()));
                            right_expression.push(RightExpression::Individual(index_fn(i)));
                        } else if input.peek(syn::Token![..]) {
                            input.parse::<syn::Token![..]>().unwrap();
                            right_expression.push(RightExpression::Individual(Index::Range(i)));
                            permute.push(positions.get("..").unwrap().clone());
                            index_fn = Box::new(Index::Unknown);
                        }
                        (right_expression, permute, repeat, index_fn)
                    },
                );

            Ok(Expression {
                left_expression,
                reduce,
                permute,
                repeat,
                right_expression,
            })
        }
    }

    fn parse_left_parenthesized(
        input: ParseStream,
        index: Index,
    ) -> syn::Result<Vec<LeftExpression>> {
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
                        content_expression.push(LeftExpression::Named {
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
                LeftExpression::Derived {
                    name: derived_name.unwrap(),
                    index,
                    operation: None,
                    shape_calc: running_mul,
                },
            );
        }

        Ok(content_expression)
    }

    fn parse_right_parenthesized(
        input: ParseStream,
        start_index: usize,
        index_fn: &mut Box<dyn Fn(usize) -> Index>,
        positions: &HashMap<String, Index>,
    ) -> syn::Result<(RightExpression, Vec<Index>, Vec<(Index, usize)>, usize)> {
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

        Ok((RightExpression::Combined { from, to }, permute, repeat, len))
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

    impl quote::ToTokens for ParsedExpression {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            let ParsedExpression {
                tensor: ref tensor_ident,
                ref expression,
            } = self;
            let Expression {
                ref left_expression,
                ref reduce,
                ref permute,
                ref repeat,
                ref right_expression,
            } = expression;

            let shape_ident = format_ident!("{}_{}", tensor_ident, "shape");
            let ignored_len_ident = format_ident!("{}", "ignored_len");

            let ignored_len = match left_expression.last().unwrap() {
                LeftExpression::Named {
                    index: Index::Unknown(i),
                    ..
                }
                | LeftExpression::Derived {
                    index: Index::Unknown(i),
                    ..
                }
                | LeftExpression::Named {
                    index: Index::Range(i),
                    ..
                } => {
                    quote!(let #ignored_len_ident = #shape_ident.len() - #i;)
                }
                _ => proc_macro2::TokenStream::new(),
            };

            let (known_indices, ignored_indices, unknown_indices) = left_expression.iter().fold(
                (Vec::new(), proc_macro2::TokenStream::new(), Vec::new()),
                |(mut known_indices, mut ignored_indices, mut unknown_indices), expression| {
                    match expression {
                        LeftExpression::Named {
                            index: Index::Known(_),
                            shape: Some(size),
                            ..
                        } => known_indices.push(quote!(#size)),
                        LeftExpression::Named {
                            index: Index::Known(i),
                            ..
                        } => known_indices.push(quote!(#shape_ident[#i])),
                        LeftExpression::Derived {
                            index: Index::Known(i),
                            shape_calc,
                            ..
                        } => known_indices.push(quote!(#shape_ident[#i] / #shape_calc)),
                        LeftExpression::Named {
                            index: Index::Range(i),
                            ..
                        } => {
                            ignored_indices = quote!(
                                (#i..(#i + #ignored_len_ident)).into_iter().map(|i| #shape_ident[i])
                            );
                        }
                        LeftExpression::Named {
                            index: Index::Unknown(i),
                            ..
                        } => {
                            unknown_indices.push(quote!(#shape_ident[#i + #ignored_len_ident - 1]))
                        }
                        LeftExpression::Derived {
                            index: Index::Unknown(i),
                            shape_calc,
                            ..
                        } => unknown_indices
                            .push(quote!(#shape_ident[#i + #ignored_len_ident - 1] / #shape_calc)),
                        _ => todo!(),
                    }
                    (known_indices, ignored_indices, unknown_indices)
                },
            );

            let decomposition_shape = match (
                known_indices.is_empty(),
                ignored_indices.is_empty(),
                unknown_indices.is_empty(),
            ) {
                (false, true, true) => {
                    quote!([#(#known_indices),*])
                }
                (false, false, true) => quote!(
                    [#(#known_indices),*]
                        .into_iter()
                        .chain(#ignored_indices)
                        .into_iter()
                        .collect::<Vec<_>>()
                ),
                (false, false, false) => quote!(
                    [#(#known_indices),*]
                        .into_iter()
                        .chain(#ignored_indices)
                        .chain([#(#unknown_indices),*].into_iter())
                        .into_iter()
                        .collect::<Vec<_>>()
                ),
                (true, false, false) => quote!(
                    #ignored_indices
                        .chain([#(#unknown_indices),*].into_iter())
                        .into_iter()
                        .collect::<Vec<_>>()
                ),
                _ => todo!(),
            };

            let (reduce_indices, reduce_operations) = reduce.iter().fold(
                (Vec::new(), Vec::new()),
                |(mut reduce_indices, mut reduce_operations), expression| {
                    let (index, operation) = expression;
                    let index = match index {
                        Index::Known(i) => quote!(#i),
                        Index::Unknown(i) => quote!(#i + #ignored_len_ident - 1),
                        _ => todo!(),
                    };
                    let operation = match operation {
                        Operation::Min => quote!(einops::Operation::Min),
                        Operation::Max => quote!(einops::Operation::Max),
                        Operation::Sum => quote!(einops::Operation::Sum),
                        Operation::Mean => quote!(einops::Operation::Mean),
                        Operation::Prod => quote!(einops::Operation::Prod),
                    };
                    reduce_indices.push(index);
                    reduce_operations.push(operation);
                    (reduce_indices, reduce_operations)
                },
            );

            let (before_ignored, ignored_permute, after_ignored, _) = permute.iter().fold(
                (
                    Vec::new(),
                    proc_macro2::TokenStream::new(),
                    Vec::new(),
                    false,
                ),
                |(
                    mut before_ignored,
                    mut ignored_permute,
                    mut after_ignored,
                    mut is_after_ignored,
                ),
                 p| {
                    let mut insert_index = |index| {
                        if is_after_ignored {
                            after_ignored.push(index);
                        } else {
                            before_ignored.push(index);
                        }
                    };
                    match p {
                        Index::Known(index) => {
                            insert_index(quote!(#index));
                        }
                        Index::Range(index) => {
                            is_after_ignored = true;
                            ignored_permute = quote!(
                                (#index..(#index + #ignored_len_ident)).into_iter()
                            )
                        }
                        Index::Unknown(index) => {
                            insert_index(quote!(#index + #ignored_len_ident - 1));
                        }
                    };
                    (
                        before_ignored,
                        ignored_permute,
                        after_ignored,
                        is_after_ignored,
                    )
                },
            );

            let permute_indices = match (
                before_ignored.is_empty(),
                ignored_permute.is_empty(),
                after_ignored.is_empty(),
            ) {
                (false, true, true) => quote!([#(#before_ignored),*]),
                (false, false, true) => quote!(
                    [#(#before_ignored),*]
                        .into_iter()
                        .chain(#ignored_permute)
                        .into_iter()
                        .collect::<Vec<_>>()
                ),
                (false, false, false) => quote!(
                    [#(#before_ignored),*]
                        .into_iter()
                        .chain(#ignored_permute)
                        .chain([#(#after_ignored),*].into_iter())
                        .into_iter()
                        .collect::<Vec<_>>()

                ),
                (true, false, false) => quote!(
                    #ignored_permute
                        .chain([#(#after_ignored),*].into_iter())
                        .into_iter()
                        .collect::<Vec<_>>()
                ),
                _ => todo!(),
            };

            let n_repeats = repeat.len();
            let repeat_pos_len = repeat.iter().map(|expression| match expression {
                (Index::Known(index), len) => quote!((#index, #len)),
                (Index::Unknown(index), len) => quote!((#index + #ignored_len_ident - 1, #len)),
                _ => todo!(),
            });

            let (before_ignored, ignored, after_ignored, _) = right_expression.iter().fold(
                (
                    Vec::new(),
                    proc_macro2::TokenStream::new(),
                    Vec::new(),
                    false,
                ),
                |(mut before_ignored, mut ignored, mut after_ignored, mut is_after_ignored),
                 expression| {
                    let mut insert_shape = |shape| {
                        if is_after_ignored {
                            after_ignored.push(shape);
                        } else {
                            before_ignored.push(shape);
                        }
                    };
                    match expression {
                        RightExpression::Individual(Index::Known(index))
                        | RightExpression::Combined {
                            from: Index::Known(index),
                            to: None,
                        } => {
                            let shape = quote!(#shape_ident[#index]);
                            insert_shape(shape);
                        }
                        RightExpression::Individual(Index::Unknown(index))
                        | RightExpression::Combined {
                            from: Index::Unknown(index),
                            to: None,
                        } => {
                            let shape = quote!(
                                #shape_ident[#index + #ignored_len_ident - 1]
                            );
                            insert_shape(shape);
                        }
                        RightExpression::Individual(Index::Range(index)) => {
                            ignored = quote!(
                                (#index..(#index + #ignored_len_ident))
                                    .into_iter().map(|i| #shape_ident[i])
                            );
                            is_after_ignored = true;
                        }
                        RightExpression::Combined {
                            from: Index::Range(index),
                            to: None,
                        } => {
                            let shape = quote!(
                                (#index..(#index + #ignored_len_ident))
                                    .into_iter().map(|i| #shape_ident[i]).product()
                            );
                            insert_shape(shape);
                        }
                        RightExpression::Combined {
                            from: Index::Known(from_index),
                            to: Some(Index::Known(to_index)),
                        } => {
                            let shape = quote!(
                                (#from_index..=#to_index)
                                    .into_iter().map(|i| #shape_ident[i]).product()
                            );
                            insert_shape(shape);
                        }
                        RightExpression::Combined {
                            from: Index::Known(from_index),
                            to: Some(Index::Unknown(to_index)),
                        }
                        | RightExpression::Combined {
                            from: Index::Known(from_index),
                            to: Some(Index::Range(to_index)),
                        } => {
                            let shape = quote!(
                                (#from_index..(#to_index + #ignored_len_ident))
                                    .into_iter().map(|i| #shape_ident[i]).product()
                            );
                            insert_shape(shape);
                        }
                        RightExpression::Combined {
                            from: Index::Range(from_index),
                            to: Some(Index::Unknown(to_index)),
                        } => {
                            let shape = quote!(
                                (#from_index..=(#to_index + #ignored_len_ident))
                                    .into_iter().map(|i| #shape_ident[i]).product()
                            );
                            insert_shape(shape);
                        }
                        RightExpression::Combined {
                            from: Index::Unknown(from_index),
                            to: Some(Index::Unknown(to_index)),
                        } => {
                            let shape = quote!(
                                ((#from_index + #ignored_len_ident - 1)..(#to_index + #ignored_len_ident))
                                    .into_iter().map(|i| #shape_ident[i]).product()
                            );
                            insert_shape(shape);
                        }
                        _ => todo!("No"),
                    }
                    (before_ignored, ignored, after_ignored, is_after_ignored)
                },
            );

            let composition_shape = match (
                before_ignored.is_empty(),
                ignored.is_empty(),
                after_ignored.is_empty(),
            ) {
                (false, true, true) => quote!([#(#before_ignored),*]),
                (false, false, true) => quote!(
                    [#(#before_ignored),*]
                        .into_iter()
                        .chain(#ignored)
                        .into_iter()
                        .collect::<Vec<_>>()
                ),
                (false, false, false) => quote!(
                    [#(#before_ignored),*]
                        .into_iter()
                        .chain(#ignored)
                        .chain([#(#after_ignored),*].into_iter())
                        .into_iter()
                        .collect::<Vec<_>>()

                ),
                (true, false, false) => quote!(
                    #ignored
                        .chain([#(#after_ignored),*].into_iter())
                        .into_iter()
                        .collect::<Vec<_>>()
                ),
                _ => todo!(),
            };

            let code = quote! {{
                use einops::Backend;

                let #shape_ident = Backend::shape(&#tensor_ident);

                #ignored_len

                let #tensor_ident = Backend::reshape(&#tensor_ident, &#decomposition_shape);

                let #tensor_ident = Backend::reduce_axes_v2(
                    &#tensor_ident, &mut [#((#reduce_indices, #reduce_operations)),*]
                );

                let #tensor_ident = Backend::transpose(&#tensor_ident, &#permute_indices);

                let #shape_ident = Backend::shape(&#tensor_ident);

                let #tensor_ident = Backend::add_axes(
                    &#tensor_ident, #shape_ident.len() + #n_repeats, &[#(#repeat_pos_len),*]
                );

                let #shape_ident = Backend::shape(&#tensor_ident);

                let #tensor_ident = Backend::reshape(&#tensor_ident, &#composition_shape);

                #tensor_ident

                //let #tensor_ident = Backend::reshape(&#tensor_ident, &[#(#decompose_iter),*]);
            }};

            code.to_tokens(tokens);

            //todo!();

            //let shapes = &self.permute;
            //let reshape = &self.reshape;
            //let explode = &self.explode;

            //let v: Vec<proc_macro2::TokenStream> = explode
            //.iter()
            //.map(|e| match e {
            //Explode::Derived { index, shape } => {
            //let shape = shape.unwrap();
            //return quote!(Backend::shape(&#tensor)[#index] / #shape);
            //}
            //Explode::Shape(shape) => quote!(#shape),
            //Explode::Index(index) => quote!(Backend::shape(&#tensor)[#index]),
            //_ => todo!(),
            //})
            //.collect();

            //let code = quote! {{
            //use einops::Backend;

            //let #tensor = Backend::reshape(&#tensor, &[#(#v),*]);

            //let #tensor = Backend::transpose(&#tensor, &[
            //#(#shapes),*
            //]);

            //let shape = Backend::shape(&#tensor);
            //let #tensor = Backend::reshape(&#tensor, &[
            //#([#(shape[#reshape]),*].iter().product()),*
            //]);

            //#tensor
            //}};

            //code.to_tokens(tokens);
        }
    }
}
