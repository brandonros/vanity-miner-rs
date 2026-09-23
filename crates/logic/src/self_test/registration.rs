// Each check owns its documentation and no-inline boundary.
// A same-named module holds metadata alongside the function (separate namespaces).
macro_rules! register_self_test {
    ($(#[doc = $description:literal])+
        fn $name:ident() -> u32 $body:block
    ) => {
        $(#[doc = $description])+
        #[inline(never)]
        pub fn $name() -> u32 $body

        pub mod $name {
            pub const LABEL: &str = concat!($($description, "\n"),+).trim_ascii();
        }
    };
}

// The central list supplies only paths, mode ownership, and ordering.
macro_rules! define_self_tests {
    ($($mode:ident ($feature:literal, $file:literal) {
        $($($check:ident)::+;)*
    })*) => {
        $(#[cfg(feature = $feature)] #[path = $file] pub mod $mode;)*

        paste::paste! {
            /// Internal buffer positions. Names, not discriminants, are the public identity.
            #[repr(usize)]
            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            pub enum Slot { $($([<$mode:camel $($check:camel)*>],)*)* }

            impl Slot {
                pub const fn index(self) -> usize { self as usize }
            }

            pub const SELF_TEST_NUM_CHECKS: usize = metadata::CASES.len();

            pub mod metadata {
                use super::Slot;
                #[derive(Clone, Copy, Debug)]
                pub struct Case {
                    pub slot: usize,
                    pub name: &'static str,
                    /// Description for enabled checks; the name for disabled checks.
                    pub label: &'static str,
                    pub kernel: &'static str,
                    pub enabled: bool,
                }
                pub const CASES: &[Case] = &[$($(Case {
                    slot: Slot::[<$mode:camel $($check:camel)*>].index(),
                    name: concat!(stringify!($mode), ".", define_self_tests!(@name $($check)::+)),
                    label: {
                        #[cfg(feature = $feature)]
                        { super::$mode::$($check)::+::LABEL }
                        #[cfg(not(feature = $feature))]
                        { concat!(stringify!($mode), ".", define_self_tests!(@name $($check)::+)) }
                    },
                    kernel: concat!("kernel_self_test_", stringify!($mode)),
                    enabled: cfg!(feature = $feature),
                },)*)*];

                pub fn find(name: &str) -> Option<&'static Case> {
                    CASES.iter().find(|case| case.name == name)
                }
            }

            // Check disabled modes too. Different files can declare the same
            // function name, but a mode's CLI selectors must remain unique.
            const _: () = {
                let mut i = 0;
                while i < metadata::CASES.len() {
                    let left = metadata::CASES[i].name.as_bytes();
                    let mut j = i + 1;
                    while j < metadata::CASES.len() {
                        let right = metadata::CASES[j].name.as_bytes();
                        if left.len() == right.len() {
                            let mut k = 0;
                            while k < left.len() && left[k] == right[k] {
                                k += 1;
                            }
                            assert!(k != left.len(), "duplicate self-test name");
                        }
                        j += 1;
                    }
                    i += 1;
                }
            };

            pub mod runners {
                $(#[cfg(feature = $feature)]
                pub mod $mode {
                    use super::super::{Slot, $mode as checks};
                    pub fn run(results: &mut [u32]) {
                        $(results[Slot::[<$mode:camel $($check:camel)*>].index()] = checks::$($check)::+();)*
                    }
                    pub fn run_selected(results: &mut [u32], selected: &[Slot]) {
                        $(if selected.contains(&Slot::[<$mode:camel $($check:camel)*>]) {
                            results[Slot::[<$mode:camel $($check:camel)*>].index()] = checks::$($check)::+();
                        })*
                    }
                })*
            }

            /// All enabled checks, in registry order.
            pub fn run_self_test(results: &mut [u32]) {
                $(#[cfg(feature = $feature)] runners::$mode::run(results);)*
            }

            /// Execute only named selections on CPU. GPU kernels still run their owning group.
            pub fn run_selected_self_tests(results: &mut [u32], selected: &[Slot]) {
                $(#[cfg(feature = $feature)] runners::$mode::run_selected(results, selected);)*
            }

            impl Slot {
                pub fn from_name(name: &str) -> Option<Self> {
                    match name {
                        $($(concat!(stringify!($mode), ".", define_self_tests!(@name $($check)::+)) => Some(Self::[<$mode:camel $($check:camel)*>]),)*)*
                        _ => None,
                    }
                }
            }
        }
    };
    (@name $head:ident :: $($rest:ident)::+) => {
        define_self_tests!(@name $($rest)::+)
    };
    (@name $name:ident) => { stringify!($name) };
}

#[cfg(test)]
mod tests {
    use crate::self_test::{SELF_TEST_NUM_CHECKS, Slot, metadata};

    #[test]
    fn registered_check_keeps_its_label() {
        // Local functions cannot be reached via super, so keep the generated
        // function and its metadata together in a module, as in real checks.
        mod probes {
            register_self_test! {
                /// Check with a label.
                fn labeled() -> u32 {
                    1
                }
            }
        }

        assert_eq!(probes::labeled::LABEL, "Check with a label.");
        assert_eq!(probes::labeled(), 1);
    }

    #[test]
    fn registry_names_and_indices_are_unique_and_dense() {
        assert_eq!(metadata::CASES.len(), SELF_TEST_NUM_CHECKS);
        for (index, case) in metadata::CASES.iter().enumerate() {
            assert_eq!(case.slot, index);
            assert!(!case.label.is_empty());
            let id = Slot::from_name(case.name).expect("registered name");
            assert_eq!(id.index(), index);
            assert_eq!(metadata::find(case.name).unwrap().slot, index);
            assert!(
                metadata::CASES[..index]
                    .iter()
                    .all(|other| other.name != case.name)
            );
            let mode = case.name.split('.').next().unwrap();
            assert_eq!(case.kernel.strip_prefix("kernel_self_test_"), Some(mode));
        }
        assert!(Slot::from_name("185").is_none());
        assert!(metadata::find("unknown.check").is_none());
    }

    #[cfg(feature = "self_test_p256_public_key")]
    #[test]
    fn selected_cpu_execution_leaves_other_checks_untouched() {
        let selected = Slot::from_name("p256_public_key.order_minus_one").unwrap();
        let mut results = [0xa5a5a5a5; SELF_TEST_NUM_CHECKS];
        crate::self_test::run_selected_self_tests(&mut results, &[selected]);
        for (index, value) in results.into_iter().enumerate() {
            assert_eq!(
                value,
                if index == selected.index() {
                    1
                } else {
                    0xa5a5a5a5
                }
            );
        }
    }
}
