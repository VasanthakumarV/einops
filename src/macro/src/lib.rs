#[proc_macro]
pub fn rearrange(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    rearrange::rearrange(input.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

mod rearrange {
    use quote::quote;

    pub fn rearrange(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
        let element: Element = syn::parse2(input)?;

        let code = quote! { #element };

        Ok(code)
    }

    #[derive(Debug)]
    struct Element {
        tensor: syn::Ident,
        permute: Vec<usize>,
        reshape: Vec<Vec<usize>>,
    }

    impl syn::parse::Parse for Element {
        fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
            let expression: Expression = input.parse::<syn::LitStr>()?.parse()?;
            input.parse::<syn::Token![,]>()?;

            Ok(Element {
                permute: expression.permute,
                tensor: input.parse::<syn::Ident>()?,
                reshape: expression.reshape,
            })
        }
    }

    struct Expression {
        permute: Vec<usize>,
        reshape: Vec<Vec<usize>>,
    }

    impl syn::parse::Parse for Expression {
        fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
            let mut left = Vec::new();
            while !input.peek(syn::Token![->]) {
                left.push(input.parse::<syn::Ident>()?.to_string());
            }

            input.parse::<syn::Token![->]>()?;

            let mut permute = Vec::new();
            let mut reshape: Vec<Vec<usize>> = Vec::new();
            let mut index: usize = 0;
            while !input.is_empty() {
                if input.peek(syn::token::Paren) {
                    let content;
                    syn::parenthesized!(content in input);

                    let mut reshape_inner = Vec::new();

                    while !content.is_empty() {
                        let ident = content.parse::<syn::Ident>()?.to_string();
                        permute.push(left.binary_search(&ident).unwrap());
                        reshape_inner.push(index);

                        index += 1;
                    }

                    reshape.push(reshape_inner);
                } else {
                    let ident = input.parse::<syn::Ident>()?.to_string();
                    permute.push(left.binary_search(&ident).unwrap());
                    reshape.push(vec![index]);

                    index += 1;
                }
            }

            Ok(Expression { permute, reshape })
        }
    }

    impl quote::ToTokens for Element {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            let tensor = &self.tensor;
            let shapes = &self.permute;
            let reshape = &self.reshape;

            let code = quote! {{
                use einops::Backend;

                let #tensor = Backend::transpose(&#tensor, &[
                    #(#shapes),*
                ]);

                let shape = Backend::shape(&#tensor);
                let #tensor = Backend::reshape(&#tensor, &[
                    #([#(shape[#reshape]),*].iter().product()),*
                ]);

                #tensor
            }};

            code.to_tokens(tokens);
        }
    }
}
