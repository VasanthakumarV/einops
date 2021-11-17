#[proc_macro]
pub fn rearrange(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    rearrange::rearrange(input.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

mod rearrange {
    use std::collections::HashMap;

    use quote::quote;
    use syn::{parse::ParseStream, token};

    pub fn rearrange(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
        let parsed_expression: ParsedExpression = syn::parse2(input)?;
        dbg!(&parsed_expression);
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
        permute: Vec<Index>,
        decomposition: Vec<LeftExpression>,
        composition: Vec<RightExpression>,
    }

    #[derive(Debug)]
    enum LeftExpression {
        Derived {
            name: String,
            index: Index,
            shape_calc: usize,
        },
        Named {
            name: String,
            index: Index,
            shape: Option<usize>,
        },
        Ignore(usize),
    }

    #[derive(Debug)]
    enum RightExpression {
        Individual(Index),
        Combined { from: Index, to: Option<Index> },
        Ignore(usize),
    }

    #[derive(Debug, Clone)]
    enum Index {
        Known(usize),
        Unknown(usize),
        Range(usize),
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
                        } else if input.peek(syn::Ident) {
                            let (name, shape) = parse_identifier(input).unwrap();
                            left_expression.push(LeftExpression::Named {
                                name,
                                shape,
                                index: index_fn(i),
                            });
                        } else if input.peek(syn::LitInt) {
                            todo!("parse_int");
                        } else if input.peek(syn::Token![..]) {
                            input.parse::<syn::Token![..]>().unwrap();
                            left_expression.push(LeftExpression::Ignore(i));
                            index_fn = Box::new(Index::Unknown);
                        }
                        (left_expression, index_fn)
                    },
                );

            let positions = left_expression.iter().enumerate().fold(
                HashMap::new(),
                |mut map, (i, expression)| {
                    match expression {
                        LeftExpression::Ignore(_) => map.insert("..".to_string(), Index::Range(i)),
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
                        _ => todo!(),
                    };
                    map
                },
            );

            let (right_expression, permute, _) =
                (0..).into_iter().take_while(|_| !input.is_empty()).fold(
                    (
                        Vec::new(),
                        Vec::new(),
                        Box::new(Index::Known) as Box<dyn Fn(usize) -> Index>,
                    ),
                    |(mut right_expression, mut permute, mut index_fn), i| {
                        if input.peek(token::Paren) {
                            let (combined, combined_permute) =
                                parse_right_parenthesized(input, i, &mut index_fn, &positions)
                                    .unwrap();
                            permute.extend(combined_permute);
                            right_expression.push(combined);
                        } else if input.peek(syn::Ident) {
                            let (name, _) = parse_identifier(input).unwrap();
                            permute.push(positions.get(&name).unwrap().clone());
                            right_expression.push(RightExpression::Individual(index_fn(i)))
                        } else if input.peek(syn::LitInt) {
                            todo!();
                        } else if input.peek(syn::Token![..]) {
                            input.parse::<syn::Token![..]>().unwrap();
                            right_expression.push(RightExpression::Ignore(i));
                            permute.push(positions.get("..").unwrap().clone());
                            index_fn = Box::new(Index::Unknown);
                        }
                        (right_expression, permute, index_fn)
                    },
                );

            Ok(Expression {
                permute,
                decomposition: left_expression,
                composition: right_expression,
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
                    if content.peek(syn::Ident) {
                        let (name, shape) = parse_identifier(&content).unwrap();
                        if let Some(size) = shape {
                            running_mul *= size;
                            content_expression.push(LeftExpression::Named {
                                name,
                                index: index.clone(),
                                shape: Some(size),
                            });
                        } else {
                            derived_name = Some(name.clone());
                            derived_index = Some(i);
                        }
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
    ) -> syn::Result<(RightExpression, Vec<Index>)> {
        let content;
        syn::parenthesized!(content in input);

        let mut permute = Vec::new();

        let from = if content.peek(syn::Token![..]) {
            content.parse::<syn::Token![..]>().unwrap();
            *index_fn = Box::new(Index::Unknown);
            permute.push(positions.get("..").unwrap().clone());
            Index::Range(start_index)
        } else if content.peek(syn::Ident) {
            let (name, _) = parse_identifier(&content).unwrap();
            permute.push(positions.get(&name).unwrap().clone());
            index_fn(start_index)
        } else {
            todo!();
        };

        let to = (1..)
            .into_iter()
            .take_while(|_| !content.is_empty())
            .fold(None, |mut to, i| {
                if content.peek(syn::Ident) {
                    let (name, _) = parse_identifier(&content).unwrap();
                    permute.push(positions.get(&name).unwrap().clone());
                    to = Some(index_fn(i + start_index));
                } else if content.peek(syn::Token![..]) {
                    content.parse::<syn::Token![..]>().unwrap();
                    permute.push(positions.get("..").unwrap().clone());
                    to = Some(Index::Range(i + start_index));
                    *index_fn = Box::new(Index::Unknown);
                }
                to
            });

        Ok((RightExpression::Combined { from, to }, permute))
    }


    fn parse_identifier(input: ParseStream) -> syn::Result<(String, Option<usize>)> {
        let name = input.parse::<syn::Ident>()?.to_string();

        let shape = if input.peek(syn::Token![:]) {
            input.parse::<syn::Token![:]>()?;
            let shape = input.parse::<syn::LitInt>()?;
            Some(shape.base10_parse::<usize>()?)
        } else {
            None
        };

        Ok((name, shape))
    }

    impl quote::ToTokens for ParsedExpression {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            todo!();
            //let tensor = &self.tensor;
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
