use std::collections::HashSet;
use std::str::FromStr;

use crate::error::EinopsError;

const ELLIPSIS: &str = "…";

#[derive(Debug, PartialEq)]
enum Axis {
    Anonymous(usize),
    Named(String),
}

#[derive(Default, Debug, PartialEq)]
struct ParsedExpression {
    has_ellipsis: bool,
    has_ellipsis_parenthesized: bool,
    has_non_unitary_anonymous_axes: bool,
    identifiers: HashSet<String>,
    composition: Vec<Vec<Axis>>,
}

impl ParsedExpression {
    fn add_axis_name(
        &mut self,
        current_ident: &Option<String>,
        bracket_group: &mut Option<Vec<Axis>>,
    ) -> Result<(), EinopsError> {
        let current_ident = match current_ident.as_ref() {
            Some(value) => {
                if self.identifiers.contains(value.as_str()) {
                    return Err(EinopsError::Parse(
                        "indexing expression contains duplicate dimension".to_string(),
                    ));
                }
                value
            }
            None => return Ok(()),
        };

        if current_ident == &ELLIPSIS {
            self.identifiers.insert(ELLIPSIS.to_string());
            match bracket_group.as_mut() {
                Some(value) => {
                    value.push(Axis::Named(ELLIPSIS.to_string()));
                    self.has_ellipsis_parenthesized = true;
                }
                None => {
                    self.composition
                        .push(vec![Axis::Named(ELLIPSIS.to_string())]);
                    self.has_ellipsis_parenthesized = false;
                }
            }
        } else {
            let size = usize::from_str(&current_ident);
            match size {
                Ok(1) => {
                    if bracket_group.is_none() {
                        self.composition.push(vec![]);
                    }
                    return Ok(());
                }
                Ok(size) => {
                    self.identifiers.insert(current_ident.clone());
                    self.has_non_unitary_anonymous_axes = true;
                    match bracket_group.as_mut() {
                        Some(value) => value.push(Axis::Anonymous(size)),
                        None => self.composition.push(vec![Axis::Anonymous(size)]),
                    }
                }
                _ => {
                    let (is_axis_name, reason) = ParsedExpression::check_axis_name(&current_ident);
                    if !is_axis_name {
                        return Err(EinopsError::Parse(format!(
                            "invalid axis identifier: {}",
                            reason.unwrap()
                        )));
                    }

                    self.identifiers.insert(current_ident.clone());
                    match bracket_group.as_mut() {
                        Some(value) => value.push(Axis::Named(current_ident.clone())),
                        None => self
                            .composition
                            .push(vec![Axis::Named(current_ident.clone())]),
                    }
                }
            }
        }

        Ok(())
    }

    fn check_axis_name(name: &str) -> (bool, Option<&str>) {
        if name.starts_with("_") || name.ends_with("_") {
            return (
                false,
                Some("axis name should not start or end with underscore"),
            );
        }
        return (true, None);
    }

    fn new(expression: &str) -> Result<Self, EinopsError> {
        let mut expression = expression.to_string();

        let mut parsed_expression = Self::default();

        let mut current_ident: Option<String> = None;
        let mut bracket_group: Option<Vec<Axis>> = None;

        if expression.contains(".") {
            if !expression.contains("...") {
                return Err(EinopsError::Parse(
                    "expression may contain dots only inside ellipsis (...)".to_string(),
                ));
            }
            let count = expression
                .matches("...")
                .collect::<Vec<&str>>()
                .iter()
                .count();
            if count != 1 {
                return Err(EinopsError::Parse(
                    "expression may contain dots only inside (...): only one ellipsis for tensor"
                        .to_string(),
                ));
            }
            expression = expression.replace("...", &ELLIPSIS);
            parsed_expression.has_ellipsis = true;
        }

        for char in expression.chars() {
            match char {
                '(' => {
                    parsed_expression.add_axis_name(&current_ident, &mut bracket_group)?;
                    current_ident = None;
                    match bracket_group {
                        Some(_) => return Err(EinopsError::Parse(
                            "axis composition is one-level (brackets inside brackets not allowed)"
                                .to_string(),
                        )),
                        None => bracket_group = Some(vec![]),
                    }
                }
                ')' => {
                    parsed_expression.add_axis_name(&current_ident, &mut bracket_group)?;
                    current_ident = None;
                    match bracket_group.take() {
                        Some(value) => parsed_expression.composition.push(value),
                        None => {
                            return Err(EinopsError::Parse(
                                "brackets are not balanced".to_string(),
                            ))
                        }
                    }
                }
                ' ' => {
                    parsed_expression.add_axis_name(&current_ident, &mut bracket_group)?;
                    current_ident = None;
                }
                '_' | '…' => match current_ident.as_mut() {
                    Some(value) => value.push(char),
                    None => current_ident = Some(char.to_string()),
                },
                _ if char.is_alphanumeric() => match current_ident.as_mut() {
                    Some(value) => value.push(char),
                    None => current_ident = Some(char.to_string()),
                },
                _ => return Err(EinopsError::Parse(format!("unknown character '{}'", char))),
            }
        }

        if bracket_group.is_some() {
            return Err(EinopsError::Parse(format!(
                "imbalanced parentheses in expression: {}",
                expression
            )));
        }
        parsed_expression.add_axis_name(&current_ident, &mut bracket_group)?;

        Ok(parsed_expression)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let tests = vec![
            // #1
            (
                ParsedExpression::new("a1 b1  c1   d1").unwrap(),
                ParsedExpression {
                    has_ellipsis: false,
                    has_ellipsis_parenthesized: false,
                    has_non_unitary_anonymous_axes: false,
                    identifiers: ["a1", "b1", "c1", "d1"]
                        .iter()
                        .cloned()
                        .map(String::from)
                        .collect(),
                    composition: vec![
                        vec![Axis::Named("a1".to_string())],
                        vec![Axis::Named("b1".to_string())],
                        vec![Axis::Named("c1".to_string())],
                        vec![Axis::Named("d1".to_string())],
                    ],
                },
            ),
            // #2
            (
                ParsedExpression::new("() () () ()").unwrap(),
                ParsedExpression {
                    has_ellipsis: false,
                    has_ellipsis_parenthesized: false,
                    has_non_unitary_anonymous_axes: false,
                    identifiers: HashSet::new(),
                    composition: vec![vec![], vec![], vec![], vec![]],
                },
            ),
            // #3
            (
                ParsedExpression::new("1 1 1 ()").unwrap(),
                ParsedExpression {
                    has_ellipsis: false,
                    has_ellipsis_parenthesized: false,
                    has_non_unitary_anonymous_axes: false,
                    identifiers: HashSet::new(),
                    composition: vec![vec![], vec![], vec![], vec![]],
                },
            ),
            // #4
            (
                ParsedExpression::new("5 (3 4)").unwrap(),
                ParsedExpression {
                    has_ellipsis: false,
                    has_ellipsis_parenthesized: false,
                    has_non_unitary_anonymous_axes: true,
                    identifiers: ["3", "4", "5"].iter().cloned().map(String::from).collect(),
                    composition: vec![
                        vec![Axis::Anonymous(5)],
                        vec![Axis::Anonymous(3), Axis::Anonymous(4)],
                    ],
                },
            ),
            // #5
            (
                ParsedExpression::new("5 1 (1 4) 1").unwrap(),
                ParsedExpression {
                    has_ellipsis: false,
                    has_ellipsis_parenthesized: false,
                    has_non_unitary_anonymous_axes: true,
                    identifiers: ["4", "5"].iter().cloned().map(String::from).collect(),
                    composition: vec![
                        vec![Axis::Anonymous(5)],
                        vec![],
                        vec![Axis::Anonymous(4)],
                        vec![],
                    ],
                },
            ),
            // #6
            (
                ParsedExpression::new("name1 ... a1 12 (name2 14)").unwrap(),
                ParsedExpression {
                    has_ellipsis: true,
                    has_ellipsis_parenthesized: false,
                    has_non_unitary_anonymous_axes: true,
                    identifiers: ["name1", ELLIPSIS, "a1", "name2", "12", "14"]
                        .iter()
                        .cloned()
                        .map(String::from)
                        .collect(),
                    composition: vec![
                        vec![Axis::Named("name1".to_string())],
                        vec![Axis::Named(ELLIPSIS.to_string())],
                        vec![Axis::Named("a1".to_string())],
                        vec![Axis::Anonymous(12)],
                        vec![Axis::Named("name2".to_string()), Axis::Anonymous(14)],
                    ],
                },
            ),
            // #7
            (
                ParsedExpression::new("(name1 ... a1 12) name2 14").unwrap(),
                ParsedExpression {
                    has_ellipsis: true,
                    has_ellipsis_parenthesized: true,
                    has_non_unitary_anonymous_axes: true,
                    identifiers: ["name1", ELLIPSIS, "a1", "name2", "12", "14"]
                        .iter()
                        .cloned()
                        .map(String::from)
                        .collect(),
                    composition: vec![
                        vec![
                            Axis::Named("name1".to_string()),
                            Axis::Named(ELLIPSIS.to_string()),
                            Axis::Named("a1".to_string()),
                            Axis::Anonymous(12),
                        ],
                        vec![Axis::Named("name2".to_string())],
                        vec![Axis::Anonymous(14)],
                    ],
                },
            ),
            // #8
            (
                ParsedExpression::new("(name1 ... a1 12 12) name2 14").unwrap(),
                ParsedExpression {
                    has_ellipsis: true,
                    has_ellipsis_parenthesized: true,
                    has_non_unitary_anonymous_axes: true,
                    identifiers: ["name1", ELLIPSIS, "a1", "name2", "12", "14"]
                        .iter()
                        .cloned()
                        .map(String::from)
                        .collect(),
                    composition: vec![
                        vec![
                            Axis::Named("name1".to_string()),
                            Axis::Named(ELLIPSIS.to_string()),
                            Axis::Named("a1".to_string()),
                            Axis::Anonymous(12),
                            Axis::Anonymous(12),
                        ],
                        vec![Axis::Named("name2".to_string())],
                        vec![Axis::Anonymous(14)],
                    ],
                },
            ),
        ];

        for (output, expected) in tests {
            assert_eq!(output, expected);
        }
    }
}
