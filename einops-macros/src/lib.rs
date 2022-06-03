mod einops;

/// Macro to perform tensor transformations using simple expressions
///
/// # Example
///
/// ```no_run
/// let output = einops!("h w c -> c h w", &input);
/// ```
#[proc_macro]
pub fn einops(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    einops::einops(input.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
