//! A JSON writer for what crosses to the page: compact, keys in the order written.
//!
//! ```
//! use switch_core::json;
//! let text = json::object(|o| {
//!     o.field("name", "Taisei").field("frames", 3u64);
//!     o.array("sizes", |a| {
//!         a.push(1u32).push(2u32);
//!     });
//! });
//! assert_eq!(text, r#"{"name":"Taisei","frames":3,"sizes":[1,2]}"#);
//! ```

use std::fmt::Write;

/// Something that writes itself as one JSON value.
pub trait Value {
    fn write_json(&self, out: &mut String);
}

/// Build a JSON object.
pub fn object(build: impl FnOnce(&mut Object)) -> String {
    let mut out = String::new();
    Object::write(&mut out, build);
    out
}

/// Build a JSON array.
pub fn array(build: impl FnOnce(&mut Array)) -> String {
    let mut out = String::new();
    Array::write(&mut out, build);
    out
}

/// An object being written; see [`object`].
pub struct Object<'a> {
    out: &'a mut String,
    empty: bool,
}

impl Object<'_> {
    fn write(out: &mut String, build: impl FnOnce(&mut Object)) {
        out.push('{');
        build(&mut Object { out, empty: true });
        out.push('}');
    }

    fn key(&mut self, key: &str) {
        if !self.empty {
            self.out.push(',');
        }
        self.empty = false;
        key.write_json(self.out);
        self.out.push(':');
    }

    pub fn field(&mut self, key: &str, value: impl Value) -> &mut Self {
        self.key(key);
        value.write_json(self.out);
        self
    }

    pub fn object(&mut self, key: &str, build: impl FnOnce(&mut Object)) -> &mut Self {
        self.key(key);
        Object::write(self.out, build);
        self
    }

    pub fn array(&mut self, key: &str, build: impl FnOnce(&mut Array)) -> &mut Self {
        self.key(key);
        Array::write(self.out, build);
        self
    }
}

/// An array being written; see [`array`].
pub struct Array<'a> {
    out: &'a mut String,
    empty: bool,
}

impl Array<'_> {
    fn write(out: &mut String, build: impl FnOnce(&mut Array)) {
        out.push('[');
        build(&mut Array { out, empty: true });
        out.push(']');
    }

    fn next(&mut self) {
        if !self.empty {
            self.out.push(',');
        }
        self.empty = false;
    }

    pub fn push(&mut self, value: impl Value) -> &mut Self {
        self.next();
        value.write_json(self.out);
        self
    }

    pub fn object(&mut self, build: impl FnOnce(&mut Object)) -> &mut Self {
        self.next();
        Object::write(self.out, build);
        self
    }

    pub fn array(&mut self, build: impl FnOnce(&mut Array)) -> &mut Self {
        self.next();
        Array::write(self.out, build);
        self
    }

    /// Push every item of `values`.
    pub fn extend<V: Value>(&mut self, values: impl IntoIterator<Item = V>) -> &mut Self {
        for value in values {
            self.push(value);
        }
        self
    }
}

/// A number written with `places` decimal places, to keep reports short.
#[derive(Debug, Clone, Copy)]
pub struct Decimals(pub f64, pub usize);

impl Value for Decimals {
    fn write_json(&self, out: &mut String) {
        let Decimals(value, places) = *self;
        if value.is_finite() {
            let _ = write!(out, "{value:.places$}");
        } else {
            out.push_str("null");
        }
    }
}

impl Value for str {
    fn write_json(&self, out: &mut String) {
        out.push('"');
        for c in self.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    let _ = write!(out, "\\u{:04x}", c as u32);
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
}

impl Value for String {
    fn write_json(&self, out: &mut String) {
        self.as_str().write_json(out);
    }
}

impl Value for bool {
    fn write_json(&self, out: &mut String) {
        out.push_str(if *self { "true" } else { "false" });
    }
}

macro_rules! integers {
    ($($ty:ty),*) => {$(
        impl Value for $ty {
            fn write_json(&self, out: &mut String) {
                let _ = write!(out, "{self}");
            }
        }
    )*};
}
integers!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

macro_rules! floats {
    ($($ty:ty),*) => {$(
        /// NaN and the infinities have no JSON form, so they are `null`.
        impl Value for $ty {
            fn write_json(&self, out: &mut String) {
                if self.is_finite() {
                    let _ = write!(out, "{self}");
                } else {
                    out.push_str("null");
                }
            }
        }
    )*};
}
floats!(f32, f64);

impl<T: Value> Value for Option<T> {
    fn write_json(&self, out: &mut String) {
        match self {
            Some(value) => value.write_json(out),
            None => out.push_str("null"),
        }
    }
}

impl<T: Value + ?Sized> Value for &T {
    fn write_json(&self, out: &mut String) {
        (**self).write_json(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_values_are_written_compactly_in_order() {
        let text = object(|o| {
            o.field("on", true)
                .field("count", 42u64)
                .field("delta", -3i32)
                .field("missing", None::<u32>);
            o.object("read", |r| {
                r.field("textures", 1u64);
            });
            o.array("pairs", |a| {
                a.object(|p| {
                    p.field("a", "x");
                })
                .array(|inner| {
                    inner.extend([1u8, 2]);
                });
            });
            o.object("empty", |_| {});
        });
        assert_eq!(
            text,
            r#"{"on":true,"count":42,"delta":-3,"missing":null,"read":{"textures":1},"pairs":[{"a":"x"},[1,2]],"empty":{}}"#
        );
        assert_eq!(array(|_| {}), "[]");
    }

    #[test]
    fn strings_escape_what_json_requires_and_keep_the_rest() {
        let text = array(|a| {
            a.push("quote \" backslash \\ line\nreturn\rtab\t bell\u{7} JUST DANCE® 2017");
        });
        assert_eq!(
            text,
            r#"["quote \" backslash \\ line\nreturn\rtab\t bell\u0007 JUST DANCE® 2017"]"#
        );
    }

    #[test]
    fn numbers_without_a_json_form_are_null() {
        let text = array(|a| {
            a.push(0.5f64)
                .push(f64::NAN)
                .push(f32::INFINITY)
                .push(Decimals(13000.66, 1))
                .push(Decimals(f64::NAN, 1));
        });
        assert_eq!(text, "[0.5,null,null,13000.7,null]");
    }
}
