// Each check owns its documentation, no-inline boundary, and device skip policy.
// A same-named module holds metadata alongside the function (separate namespaces).
macro_rules! register_self_test {
    ($(#[doc = $description:literal])+ $(#[gpu_skip = $skip:literal])?
        fn $name:ident() -> u32 $body:block
    ) => {
        $(#[doc = $description])+
        #[inline(never)]
        pub fn $name() -> u32 $body

        pub mod $name {
            pub const LABEL: &str = concat!($($description, "\n"),+).trim_ascii();
            pub const GPU_SKIP: Option<&str> = register_self_test!(@skip $($skip)?);

            #[inline(always)]
            pub fn run_device() -> u32 {
                register_self_test!(@device $name $(, $skip)?)
            }
        }
    };
    (@skip) => { None };
    (@skip $reason:literal) => { Some($reason) };
    (@device $name:ident) => { super::$name() };
    // No reference to the skipped function is emitted into its device runner.
    (@device $name:ident, $reason:literal) => { 2 };
}

// A single declaration supplies both grouped and per-check self-test layouts.
macro_rules! self_test_kernel {
    ($module:ident, $entry:ident, $export:expr, |$selector:ident, $results:ident| $body:block) => {
        llvm_metal_kernel::kernel! {
            pub mod $module;
            entry $export;
            /// # Safety
            /// One invocation, a valid selector and disjoint initialized registry storage.
            #[cfg(feature = "self_test_metal_entries")]
            pub unsafe extern "C" fn $entry(
                $selector: Read Fixed u32,
                $results: ReadWrite Fixed [u32; crate::self_test::SELF_TEST_NUM_CHECKS],
            ) {
                let $results = $results.cast::<u32>();
                $body
            } dispatch Single;
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
                    pub metal_entry: &'static str,
                    pub enabled: bool,
                    /// Device skip policy, available when the check is enabled.
                    pub gpu_skip: Option<&'static str>,
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
                    metal_entry: concat!("kernel_self_test_", stringify!($mode), "_", define_self_tests!(@name $($check)::+)),
                    enabled: cfg!(feature = $feature),
                    gpu_skip: {
                        #[cfg(feature = $feature)]
                        { super::$mode::$($check)::+::GPU_SKIP }
                        #[cfg(not(feature = $feature))]
                        { None }
                    },
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
                    self_test_kernel!(group_abi, [<kernel_self_test_ $mode>], concat!("kernel_self_test_", stringify!($mode)), |selector, results| {
                        let slot = unsafe { selector.read() };
                        let results = unsafe { core::slice::from_raw_parts_mut(results, crate::self_test::SELF_TEST_NUM_CHECKS) };
                        run_slot(results, slot as usize);
                    });
                    $(self_test_kernel!([<abi $(_ $check)*>], [<kernel_self_test_ $mode $(_ $check)*>], concat!("kernel_self_test_", stringify!($mode), "_", define_self_tests!(@name $($check)::+)), |selector, results| {
                        let slot = Slot::[<$mode:camel $($check:camel)*>].index();
                        let selected = unsafe { selector.read() };
                        if selected == slot as u32 || selected == u32::MAX {
                            unsafe { results.add(slot).write(checks::$($check)::+()); }
                        }
                    });)*
                    pub fn descriptor(entry: &str) -> Option<&'static [u8]> {
                        if entry == group_abi::ENTRY { return Some(&group_abi::DESCRIPTOR); }
                        $(if entry == [<abi $(_ $check)*>]::ENTRY { return Some(&[<abi $(_ $check)*>]::DESCRIPTOR); })*
                        None
                    }
                    #[cfg(feature = "self_test_metal_entries")]
                    pub fn metal_entry(slot: usize) -> Option<unsafe extern "C" fn(*const u32, *mut [u32; crate::self_test::SELF_TEST_NUM_CHECKS])> {
                        $(if slot == Slot::[<$mode:camel $($check:camel)*>].index() {
                            return Some([<kernel_self_test_ $mode $(_ $check)*>]);
                        })*
                        None
                    }
                    pub fn run(results: &mut [u32]) {
                        $(results[Slot::[<$mode:camel $($check:camel)*>].index()] = checks::$($check)::+();)*
                    }
                    /// Execute one original check, or the full group for u32::MAX.
                    /// Unlike run_device, this does not apply backend-specific skips.
                    pub fn run_slot(results: &mut [u32], slot: usize) {
                        $(if slot == u32::MAX as usize || slot == Slot::[<$mode:camel $($check:camel)*>].index() {
                            results[Slot::[<$mode:camel $($check:camel)*>].index()] = checks::$($check)::+();
                        })*
                    }
                    pub fn run_device(results: &mut [u32]) {
                        $(results[Slot::[<$mode:camel $($check:camel)*>].index()] =
                            checks::$($check)::+::run_device();)*
                    }
                    pub fn run_selected(results: &mut [u32], selected: &[Slot]) {
                        $(if selected.contains(&Slot::[<$mode:camel $($check:camel)*>]) {
                            results[Slot::[<$mode:camel $($check:camel)*>].index()] = checks::$($check)::+();
                        })*
                    }
                })*
            }

            /// Native expected descriptors from the same declarations as device entries.
            pub fn descriptor(entry: &str) -> Option<&'static [u8]> {
                $(#[cfg(feature = $feature)]
                if let Some(d) = runners::$mode::descriptor(entry) { return Some(d); })*
                None
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
    fn device_skip_does_not_execute_the_check_body() {
        use core::sync::atomic::Ordering;

        // Local functions cannot be reached via super, so keep the generated
        // function and its metadata together in a module, as in real checks.
        mod probes {
            use core::sync::atomic::{AtomicU32, Ordering};

            pub(super) static CALLS: AtomicU32 = AtomicU32::new(0);

            register_self_test! {
                /// CPU-only check with an observable side effect.
                #[gpu_skip = "test skip"]
                fn cpu_only() -> u32 {
                    CALLS.fetch_add(1, Ordering::Relaxed);
                    0
                }
            }
        }

        assert_eq!(
            probes::cpu_only::LABEL,
            "CPU-only check with an observable side effect."
        );
        assert_eq!(probes::cpu_only::GPU_SKIP, Some("test skip"));
        assert_eq!(probes::cpu_only::run_device(), 2);
        assert_eq!(probes::CALLS.load(Ordering::Relaxed), 0);
        assert_eq!(probes::cpu_only(), 0);
        assert_eq!(probes::CALLS.load(Ordering::Relaxed), 1);
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
