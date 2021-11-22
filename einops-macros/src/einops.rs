mod parse;
mod tokens;

use quote::{format_ident, quote};
use syn::parse::ParseStream;

use parse::{
    parse_composition_permute_repeat, parse_decomposition, parse_reduce, Composition,
    Decomposition, Index, Operation,
};
use tokens::{
    to_tokens_composition, to_tokens_decomposition, to_tokens_permute, to_tokens_reduce,
    to_tokens_repeat,
};

pub fn einops(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
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
    decomposition: Vec<Decomposition>,
    reduce: Vec<(Index, Operation)>,
    permute: Vec<Index>,
    repeat: Vec<(Index, usize)>,
    composition: Vec<Composition>,
}

impl syn::parse::Parse for Expression {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let decomposition = parse_decomposition(&input);

        let reduce = parse_reduce(&decomposition);

        let (composition, permute, repeat) =
            parse_composition_permute_repeat(&input, &decomposition);

        Ok(Expression {
            decomposition,
            reduce,
            permute,
            repeat,
            composition,
        })
    }
}

impl quote::ToTokens for ParsedExpression {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ParsedExpression {
            tensor: ref tensor_ident,
            ref expression,
        } = self;
        let Expression {
            decomposition: ref left_expression,
            ref reduce,
            ref permute,
            ref repeat,
            composition: ref right_expression,
        } = expression;

        let shape_ident = format_ident!("{}_{}", tensor_ident, "shape");
        let ignored_len_ident = format_ident!("{}", "ignored_len");

        let shape_tokens = quote!(
            let #shape_ident = Backend::shape(&#tensor_ident);
        );

        let ignored_len_tokens = match left_expression.last().unwrap() {
            Decomposition::Named {
                index: Index::Unknown(i),
                ..
            }
            | Decomposition::Derived {
                index: Index::Unknown(i),
                ..
            }
            | Decomposition::Named {
                index: Index::Range(i),
                ..
            } => {
                quote!(let #ignored_len_ident = #shape_ident.len() - #i;)
            }
            _ => proc_macro2::TokenStream::new(),
        };

        let decomposition_tokens = if left_expression
            .iter()
            .any(|expression| matches!(expression, Decomposition::Derived { .. }))
        {
            to_tokens_decomposition(
                left_expression,
                &tensor_ident,
                &ignored_len_ident,
                &shape_ident,
            )
        } else {
            proc_macro2::TokenStream::new()
        };

        let reduce_tokens = if !reduce.is_empty() {
            to_tokens_reduce(reduce, &tensor_ident, &ignored_len_ident)
        } else {
            proc_macro2::TokenStream::new()
        };

        let permute_tokens = if permute.windows(2).any(|w| w[0] > w[1]) {
            to_tokens_permute(permute, &tensor_ident, &ignored_len_ident)
        } else {
            proc_macro2::TokenStream::new()
        };

        let repeat_tokens = if !repeat.is_empty() {
            to_tokens_repeat(repeat, &tensor_ident, &ignored_len_ident, &shape_ident)
        } else {
            proc_macro2::TokenStream::new()
        };

        let composition_tokens = if right_expression
            .iter()
            .any(|expression| matches!(expression, Composition::Combined { .. }))
        {
            to_tokens_composition(
                right_expression,
                &tensor_ident,
                &ignored_len_ident,
                &shape_ident,
            )
        } else {
            proc_macro2::TokenStream::new()
        };

        let code = quote! {{
            use einops::Backend;

            #shape_tokens

            #ignored_len_tokens

            #decomposition_tokens

            #reduce_tokens

            #permute_tokens

            #shape_tokens

            #repeat_tokens

            #shape_tokens

            #composition_tokens

            #tensor_ident
        }};

        code.to_tokens(tokens);
    }
}
