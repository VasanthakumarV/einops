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
    }

    impl syn::parse::Parse for Element {
        fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
            let expression: Expression = input.parse::<syn::LitStr>()?.parse()?;
            input.parse::<syn::Token![,]>()?;

            Ok(Element {
                permute: expression.permute,
                tensor: input.parse::<syn::Ident>()?,
            })
        }
    }

    struct Expression {
        permute: Vec<usize>,
    }

    impl syn::parse::Parse for Expression {
        fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
            let mut left = Vec::new();
            while !input.peek(syn::Token![->]) {
                left.push(input.parse::<syn::Ident>()?.to_string());
            }
            input.parse::<syn::Token![->]>()?;
            let mut permute = Vec::new();
            while !input.is_empty() {
                let ident = input.parse::<syn::Ident>()?.to_string();
                permute.push(left.binary_search(&ident).unwrap());
            }

            Ok(Expression { permute })
        }
    }

    impl quote::ToTokens for Element {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            let tensor = &self.tensor;
            let shapes = self.permute.iter();

            let code = quote! {
                crate::backend::Backend::transpose(&#tensor, &[
                    #(#shapes),*
                ])
            };
            code.to_tokens(tokens);
        }
    }
}
