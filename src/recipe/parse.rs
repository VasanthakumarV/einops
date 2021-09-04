use std::collections::HashSet;
use std::str::FromStr;

use crate::error::EinopsError;

pub const ELLIPSIS: &str = "…";

#[derive(Debug, Default, PartialEq)]
pub struct Axis {
    pub name: String,
    pub size: Option<usize>,
    pub pos: usize,
}

#[derive(Default, Debug, PartialEq)]
pub struct ParsedExpression {
    pub has_ellipsis: bool,
    pub has_ellipsis_parenthesized: bool,
    pub has_non_unitary_anonymous_axes: bool,
    pub identifiers_named: HashSet<String>,
    pub composition: Vec<Vec<Axis>>,
}

impl ParsedExpression {
    pub fn new(expression: &str) -> Result<Self, EinopsError> {
        let mut expression = expression.to_string();

        let mut parsed_expression = Self::default();

        // A string to tract the characters of the current identifier
        let mut current_ident: Option<String> = None;

        // A vector to store identifiers that are present inside parenthesis
        let mut bracket_group: Option<Vec<Axis>> = None;

        if expression.contains('.') {
            if !expression.contains("...") {
                return Err(EinopsError::Parse(
                    "expression may contain dots only inside ellipsis (...)".to_string(),
                ));
            }

            let count = expression.matches("...").count();
            if count != 1 {
                return Err(EinopsError::Parse(
                    "expression may contain dots only inside (...): only one ellipsis for tensor"
                        .to_string(),
                ));
            }

            expression = expression.replace("...", ELLIPSIS);

            parsed_expression.has_ellipsis = true;
        }

        for char in expression.chars() {
            match char {
                '(' => {
                    // Add currently tracked identifier to `composition` and reinitialize the
                    // `current_ident` variable
                    parsed_expression.add_axis_name(&current_ident, &mut bracket_group)?;
                    current_ident = None;

                    match bracket_group {
                        // Nested parenthesis is not supported
                        Some(_) => return Err(EinopsError::Parse(
                            "axis composition is one-level (brackets inside brackets not allowed)"
                                .to_string(),
                        )),
                        // Initialize vector to store identifiers inside parenthesis
                        None => bracket_group = Some(vec![]),
                    }
                }
                ')' => {
                    // Add `current_ident` to the bracket_group
                    parsed_expression.add_axis_name(&current_ident, &mut bracket_group)?;
                    current_ident = None;

                    // Push the contents of `bracket_group` to `composition and revert
                    // it back to `None`
                    match bracket_group.take() {
                        Some(value) => parsed_expression.composition.push(value),
                        None => {
                            return Err(EinopsError::Parse("brackets are not balanced".to_string()))
                        }
                    }
                }
                // A space marks the end of the name of a identifier
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

        // `bracket_group` should be `None`, once we exhaust all the characters
        // of the expression
        if bracket_group.is_some() {
            return Err(EinopsError::Parse(format!(
                "imbalanced parentheses in expression: {}",
                expression
            )));
        }

        // We flush the content of `current_ident` to composition
        parsed_expression.add_axis_name(&current_ident, &mut bracket_group)?;

        Ok(parsed_expression)
    }

    fn add_axis_name(
        &mut self,
        current_ident: &Option<String>,
        bracket_group: &mut Option<Vec<Axis>>,
    ) -> Result<(), EinopsError> {
        let current_ident = match current_ident.as_ref() {
            Some(value) => {
                // We raise an error, if the name of the identifier is a duplicate
                if self.identifiers_named.contains(value.as_str()) {
                    return Err(EinopsError::Parse(
                        "indexing expression contains duplicate dimension".to_string(),
                    ));
                }
                value
            }
            // We return fast, if empty
            None => return Ok(()),
        };

        if current_ident == ELLIPSIS {
            self.identifiers_named.insert(ELLIPSIS.to_string());
            match bracket_group.as_mut() {
                Some(value) => {
                    value.push(Axis {
                        name: ELLIPSIS.to_string(),
                        size: None,
                        ..Default::default()
                    });
                    self.has_ellipsis_parenthesized = true;
                }
                None => {
                    self.composition.push(vec![Axis {
                        name: ELLIPSIS.to_string(),
                        size: None,
                        ..Default::default()
                    }]);
                    self.has_ellipsis_parenthesized = false;
                }
            }
        } else {
            // We try to parse the string as an integer
            let size = usize::from_str(current_ident);

            match size {
                Ok(1) => {
                    if bracket_group.is_none() {
                        self.composition.push(vec![]);
                    }

                    return Ok(());
                }
                Ok(size) => {
                    self.has_non_unitary_anonymous_axes = true;

                    match bracket_group.as_mut() {
                        Some(value) => value.push(Axis {
                            name: size.to_string(),
                            size: Some(size),
                            ..Default::default()
                        }),
                        None => self.composition.push(vec![Axis {
                            name: size.to_string(),
                            size: Some(size),
                            ..Default::default()
                        }]),
                    }
                }
                _ => {
                    let (is_axis_name, reason) = ParsedExpression::check_axis_name(current_ident);
                    if !is_axis_name {
                        // `unwrap` is safe, because it will always have a value
                        // if the axis name is invalid
                        return Err(EinopsError::Parse(format!(
                            "invalid axis identifier: {}",
                            reason.unwrap()
                        )));
                    }

                    self.identifiers_named.insert(current_ident.clone());

                    match bracket_group.as_mut() {
                        Some(value) => value.push(Axis {
                            name: current_ident.clone(),
                            size: None,
                            ..Default::default()
                        }),
                        None => self.composition.push(vec![Axis {
                            name: current_ident.clone(),
                            size: None,
                            ..Default::default()
                        }]),
                    }
                }
            }
        }

        Ok(())
    }

    fn check_axis_name(name: &str) -> (bool, Option<&str>) {
        if name.starts_with('_') || name.ends_with('_') {
            return (
                false,
                Some("axis name should not start or end with underscore"),
            );
        }

        (true, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_expressions() {
        let tests = vec![
            ParsedExpression::new("... a b c d ..."),
            ParsedExpression::new("... a b c (d ...)"),
            ParsedExpression::new("(... a) b c (d ...)"),
            ParsedExpression::new("(a)) b c (d ...)"),
            ParsedExpression::new("(a b c (d ...)"),
            ParsedExpression::new("(a) (()) b c (d ...)"),
            ParsedExpression::new("(a) ((b c) (d ...)"),
        ];

        for output in tests {
            assert!(output.is_err());
        }
    }

    #[test]
    fn parse_expressions() {
        let tests = vec![
            // #1
            (
                ParsedExpression::new("a1 b1  c1   d1").unwrap(),
                ParsedExpression {
                    has_ellipsis: false,
                    has_ellipsis_parenthesized: false,
                    has_non_unitary_anonymous_axes: false,
                    identifiers_named: ["a1", "b1", "c1", "d1"]
                        .iter()
                        .cloned()
                        .map(String::from)
                        .collect(),
                    composition: vec![
                        vec![Axis {
                            name: "a1".to_string(),
                            size: None,
                            pos: 0,
                        }],
                        vec![Axis {
                            name: "b1".to_string(),
                            size: None,
                            pos: 0,
                        }],
                        vec![Axis {
                            name: "c1".to_string(),
                            size: None,
                            pos: 0,
                        }],
                        vec![Axis {
                            name: "d1".to_string(),
                            size: None,
                            pos: 0,
                        }],
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
                    identifiers_named: HashSet::new(),
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
                    identifiers_named: HashSet::new(),
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
                    identifiers_named: HashSet::new(),
                    composition: vec![
                        vec![Axis {
                            name: 5.to_string(),
                            size: Some(5),
                            pos: 0,
                        }],
                        vec![
                            Axis {
                                name: 3.to_string(),
                                size: Some(3),
                                pos: 0,
                            },
                            Axis {
                                name: 4.to_string(),
                                size: Some(4),
                                pos: 0,
                            },
                        ],
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
                    identifiers_named: HashSet::new(),
                    composition: vec![
                        vec![Axis {
                            name: 5.to_string(),
                            size: Some(5),
                            pos: 0,
                        }],
                        vec![],
                        vec![Axis {
                            name: 4.to_string(),
                            size: Some(4),
                            pos: 0,
                        }],
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
                    identifiers_named: ["name1", ELLIPSIS, "a1", "name2"]
                        .iter()
                        .cloned()
                        .map(String::from)
                        .collect(),
                    composition: vec![
                        vec![Axis {
                            name: "name1".to_string(),
                            size: None,
                            pos: 0,
                        }],
                        vec![Axis {
                            name: ELLIPSIS.to_string(),
                            size: None,
                            pos: 0,
                        }],
                        vec![Axis {
                            name: "a1".to_string(),
                            size: None,
                            pos: 0,
                        }],
                        vec![Axis {
                            name: 12.to_string(),
                            size: Some(12),
                            pos: 0,
                        }],
                        vec![
                            Axis {
                                name: "name2".to_string(),
                                size: None,
                                pos: 0,
                            },
                            Axis {
                                name: 14.to_string(),
                                size: Some(14),
                                pos: 0,
                            },
                        ],
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
                    identifiers_named: ["name1", ELLIPSIS, "a1", "name2"]
                        .iter()
                        .cloned()
                        .map(String::from)
                        .collect(),
                    composition: vec![
                        vec![
                            Axis {
                                name: "name1".to_string(),
                                size: None,
                                pos: 0,
                            },
                            Axis {
                                name: ELLIPSIS.to_string(),
                                size: None,
                                pos: 0,
                            },
                            Axis {
                                name: "a1".to_string(),
                                size: None,
                                pos: 0,
                            },
                            Axis {
                                name: 12.to_string(),
                                size: Some(12),
                                pos: 0,
                            },
                        ],
                        vec![Axis {
                            name: "name2".to_string(),
                            size: None,
                            pos: 0,
                        }],
                        vec![Axis {
                            name: 14.to_string(),
                            size: Some(14),
                            pos: 0,
                        }],
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
                    identifiers_named: ["name1", ELLIPSIS, "a1", "name2"]
                        .iter()
                        .cloned()
                        .map(String::from)
                        .collect(),
                    composition: vec![
                        vec![
                            Axis {
                                name: "name1".to_string(),
                                size: None,
                                pos: 0,
                            },
                            Axis {
                                name: ELLIPSIS.to_string(),
                                size: None,
                                pos: 0,
                            },
                            Axis {
                                name: "a1".to_string(),
                                size: None,
                                pos: 0,
                            },
                            Axis {
                                name: 12.to_string(),
                                size: Some(12),
                                pos: 0,
                            },
                            Axis {
                                name: 12.to_string(),
                                size: Some(12),
                                pos: 0,
                            },
                        ],
                        vec![Axis {
                            name: "name2".to_string(),
                            size: None,
                            pos: 0,
                        }],
                        vec![Axis {
                            name: 14.to_string(),
                            size: Some(14),
                            pos: 0,
                        }],
                    ],
                },
            ),
        ];

        for (output, expected) in tests {
            assert_eq!(output, expected);
        }
    }
}
