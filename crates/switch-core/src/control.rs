//! A title's control data: the name, publisher and icon a console's home menu
//! shows for it, and the rest of what its NACP declares.
//!
//! Every application ships a Control NCA alongside its Program NCA. Its RomFS
//! holds `control.nacp` — a fixed-layout 0x4000-byte metadata blob — and one
//! `icon_<language>.dat` JPEG per language the title was localized for.
//!
//! The NACP begins with 16 title entries, one per language slot (see
//! [`LANGUAGES`]), each 0x300 bytes: a 0x200-byte name followed by a
//! 0x100-byte publisher, both NUL-padded UTF-8. A title is not localized into
//! every slot, so the entries for languages it doesn't support are blank —
//! [`Nacp::preferred`] picks the first slot that isn't.
//!
//! Fixed fields follow at 0x3000. The ones read here:
//!
//! ```text
//! 0x3000  ISBN (0x25 bytes)
//! 0x3025  startup user account (u8)
//! 0x3028  attribute flags (u32)
//! 0x3034  screenshot (u8)
//! 0x3035  video capture (u8)
//! 0x3040  age rating per rating organisation (32 x i8, -1 = unrated)
//! 0x3060  display version (0x10 bytes)
//! 0x3070  add-on content base id (u64)
//! 0x3078  save data owner id (u64)
//! 0x3080  user account save data size / journal size (2 x i64)
//! 0x3090  device save data size / journal size (2 x i64)
//! 0x30A0  BCAT delivery cache storage size (i64)
//! 0x30A8  application error code category (8 bytes)
//! ```

use crate::keys::KeySet;
use crate::nca::{ContentType, Nca};
use crate::nsp::Pfs0File;
use crate::romfs::RomFs;
use crate::source::{ByteSource, SliceSource, Window};
use crate::Error;

/// The NACP language slots, in the order their title entries appear.
pub const LANGUAGES: [&str; 16] = [
    "AmericanEnglish",
    "BritishEnglish",
    "Japanese",
    "French",
    "German",
    "LatinAmericanSpanish",
    "Spanish",
    "Italian",
    "Dutch",
    "CanadianFrench",
    "Portuguese",
    "Russian",
    "Korean",
    "TraditionalChinese",
    "SimplifiedChinese",
    "BrazilianPortuguese",
];

/// Names the SDK's own tooling used for the last two slots before they were
/// renamed, and which repack tools still emit for the icon files. Indexed
/// alongside [`LANGUAGES`], empty where the name never differed.
const LEGACY_LANGUAGE_NAMES: [&str; 16] = [
    "", "", "", "", "", "", "", "", "", "", "", "", "", "Taiwanese", "Chinese", "",
];

/// The rating boards the age-rating array has a slot for, in slot order.
pub const RATING_ORGANISATIONS: [&str; 13] = [
    "CERO",
    "GRACGCRB",
    "GSRMR",
    "ESRB",
    "ClassInd",
    "USK",
    "PEGI",
    "PEGI Portugal",
    "PEGI BBFC",
    "Russian",
    "ACB",
    "OFLC",
    "IARC",
];

/// Size of one NACP title entry: name then publisher.
const TITLE_ENTRY_SIZE: usize = 0x300;
const TITLE_NAME_SIZE: usize = 0x200;
const TITLE_PUBLISHER_SIZE: usize = 0x100;

const ISBN_OFFSET: usize = 0x3000;
const ISBN_SIZE: usize = 0x25;
const STARTUP_USER_ACCOUNT_OFFSET: usize = 0x3025;
const ATTRIBUTE_FLAG_OFFSET: usize = 0x3028;
const SCREENSHOT_OFFSET: usize = 0x3034;
const VIDEO_CAPTURE_OFFSET: usize = 0x3035;
const RATING_AGE_OFFSET: usize = 0x3040;
const RATING_AGE_SLOTS: usize = 32;
const DISPLAY_VERSION_OFFSET: usize = 0x3060;
const DISPLAY_VERSION_SIZE: usize = 0x10;
const ADD_ON_CONTENT_BASE_ID_OFFSET: usize = 0x3070;
const SAVE_DATA_OWNER_ID_OFFSET: usize = 0x3078;
const USER_ACCOUNT_SAVE_DATA_SIZE_OFFSET: usize = 0x3080;
const USER_ACCOUNT_SAVE_DATA_JOURNAL_SIZE_OFFSET: usize = 0x3088;
const DEVICE_SAVE_DATA_SIZE_OFFSET: usize = 0x3090;
const DEVICE_SAVE_DATA_JOURNAL_SIZE_OFFSET: usize = 0x3098;
const BCAT_STORAGE_SIZE_OFFSET: usize = 0x30A0;
const ERROR_CODE_CATEGORY_OFFSET: usize = 0x30A8;
const ERROR_CODE_CATEGORY_SIZE: usize = 8;
/// A real `control.nacp` is 0x4000 bytes, but nothing past the last field
/// read here is needed — so that, not the full size, is what's required.
const NACP_MIN_SIZE: usize = ERROR_CODE_CATEGORY_OFFSET + ERROR_CODE_CATEGORY_SIZE;

/// `AttributeFlag` bit 0: the title is a demo build.
const ATTRIBUTE_DEMO: u32 = 1 << 0;
/// An age-rating slot the title wasn't submitted to.
const RATING_UNRATED: i8 = -1;

/// One language's name and publisher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Title {
    /// The slot's name from [`LANGUAGES`].
    pub language: &'static str,
    /// The title as the home menu shows it.
    pub name: String,
    /// The publisher line under it.
    pub publisher: String,
}

/// One rating board's verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rating {
    /// The board's name from [`RATING_ORGANISATIONS`].
    pub organisation: &'static str,
    /// Minimum age the board passed the title at.
    pub age: u8,
}

/// Whether a user profile has to be chosen before the title starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupUserAccount {
    None,
    Required,
    RequiredWithNetworkServiceAccountAvailable,
    Unknown(u8),
}

impl StartupUserAccount {
    pub fn from_u8(v: u8) -> StartupUserAccount {
        match v {
            0 => StartupUserAccount::None,
            1 => StartupUserAccount::Required,
            2 => StartupUserAccount::RequiredWithNetworkServiceAccountAvailable,
            other => StartupUserAccount::Unknown(other),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            StartupUserAccount::None => "not required",
            StartupUserAccount::Required => "required",
            StartupUserAccount::RequiredWithNetworkServiceAccountAvailable => {
                "required (with network service account)"
            }
            StartupUserAccount::Unknown(_) => "unknown",
        }
    }
}

/// Whether the console's capture button works in this title.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSupport {
    /// Screenshots: 0 allows the capture button, 1 blocks it. Video: 0 is no
    /// recording at all, 1 is the long-press recording every title gets, 2 is
    /// recording the title itself can start.
    Screenshot(u8),
    Video(u8),
}

impl CaptureSupport {
    pub fn name(&self) -> &'static str {
        match self {
            CaptureSupport::Screenshot(0) => "allowed",
            CaptureSupport::Screenshot(1) => "blocked",
            CaptureSupport::Video(0) => "disabled",
            CaptureSupport::Video(1) => "manual",
            CaptureSupport::Video(2) => "enabled",
            _ => "unknown",
        }
    }
}

/// A parsed `control.nacp`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nacp {
    /// Only the slots the title is actually localized into.
    pub titles: Vec<Title>,
    /// The version string the title displays (`1.0.2`), not its NCA version.
    pub display_version: String,
    /// Set by a handful of titles with a physical release.
    pub isbn: String,
    /// Whether this is a demo build, from the attribute flags.
    pub is_demo: bool,
    pub startup_user_account: StartupUserAccount,
    pub screenshot: CaptureSupport,
    pub video_capture: CaptureSupport,
    /// Only the boards that actually rated the title.
    pub ratings: Vec<Rating>,
    /// Base id of the title's DLC, 0 when it has none.
    pub add_on_content_base_id: u64,
    /// Which title's save data this one shares, 0 when it owns its own.
    pub save_data_owner_id: u64,
    /// Save data the title reserves, in bytes. The journal is the write-ahead
    /// area on top of it; either can be 0 for a title that saves nothing.
    pub user_account_save_data_size: i64,
    pub user_account_save_data_journal_size: i64,
    /// Console-wide (not per-profile) save data, same shape.
    pub device_save_data_size: i64,
    pub device_save_data_journal_size: i64,
    /// Space reserved for BCAT, the background data-delivery cache.
    pub bcat_delivery_cache_storage_size: i64,
    /// The prefix of the error codes the title reports (`2181` in
    /// `2181-0002`), empty when it uses the system's.
    pub application_error_code_category: String,
}

impl Nacp {
    /// The file's name in a Control NCA's RomFS root.
    pub const PATH: &'static str = "/control.nacp";

    /// Parse a `control.nacp`.
    pub fn parse(data: &[u8]) -> Result<Nacp, Error> {
        if data.len() < NACP_MIN_SIZE {
            return Err(Error::Truncated {
                what: "control.nacp".into(),
                expected: NACP_MIN_SIZE,
                got: data.len(),
            });
        }
        let mut titles = Vec::new();
        for (index, language) in LANGUAGES.iter().enumerate() {
            let entry = index * TITLE_ENTRY_SIZE;
            let name = nul_terminated(&data[entry..entry + TITLE_NAME_SIZE]);
            let publisher = nul_terminated(
                &data[entry + TITLE_NAME_SIZE..entry + TITLE_NAME_SIZE + TITLE_PUBLISHER_SIZE],
            );
            if name.is_empty() && publisher.is_empty() {
                continue;
            }
            titles.push(Title { language, name, publisher });
        }

        let mut ratings = Vec::new();
        for (slot, organisation) in RATING_ORGANISATIONS.iter().enumerate() {
            // The array has 32 slots for 13 named boards; the rest are
            // reserved and always unrated.
            debug_assert!(slot < RATING_AGE_SLOTS);
            let age = data[RATING_AGE_OFFSET + slot] as i8;
            if age != RATING_UNRATED {
                ratings.push(Rating { organisation, age: age as u8 });
            }
        }

        Ok(Nacp {
            titles,
            display_version: nul_terminated(
                &data[DISPLAY_VERSION_OFFSET..DISPLAY_VERSION_OFFSET + DISPLAY_VERSION_SIZE],
            ),
            isbn: nul_terminated(&data[ISBN_OFFSET..ISBN_OFFSET + ISBN_SIZE]),
            is_demo: crate::nsp::read_u32(data, ATTRIBUTE_FLAG_OFFSET) & ATTRIBUTE_DEMO != 0,
            startup_user_account: StartupUserAccount::from_u8(data[STARTUP_USER_ACCOUNT_OFFSET]),
            screenshot: CaptureSupport::Screenshot(data[SCREENSHOT_OFFSET]),
            video_capture: CaptureSupport::Video(data[VIDEO_CAPTURE_OFFSET]),
            ratings,
            add_on_content_base_id: crate::nsp::read_u64(data, ADD_ON_CONTENT_BASE_ID_OFFSET),
            save_data_owner_id: crate::nsp::read_u64(data, SAVE_DATA_OWNER_ID_OFFSET),
            user_account_save_data_size: read_i64(data, USER_ACCOUNT_SAVE_DATA_SIZE_OFFSET),
            user_account_save_data_journal_size: read_i64(
                data,
                USER_ACCOUNT_SAVE_DATA_JOURNAL_SIZE_OFFSET,
            ),
            device_save_data_size: read_i64(data, DEVICE_SAVE_DATA_SIZE_OFFSET),
            device_save_data_journal_size: read_i64(data, DEVICE_SAVE_DATA_JOURNAL_SIZE_OFFSET),
            bcat_delivery_cache_storage_size: read_i64(data, BCAT_STORAGE_SIZE_OFFSET),
            application_error_code_category: nul_terminated(
                &data[ERROR_CODE_CATEGORY_OFFSET..ERROR_CODE_CATEGORY_OFFSET + ERROR_CODE_CATEGORY_SIZE],
            ),
        })
    }

    /// The entry to show: American English when the title has it, otherwise
    /// whichever language slot comes first.
    pub fn preferred(&self) -> Option<&Title> {
        self.titles
            .iter()
            .find(|t| t.language == LANGUAGES[0])
            .or_else(|| self.titles.first())
    }
}

/// What a Control NCA says about its title.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Control {
    /// The Control NCA's own title id.
    pub title_id: u64,
    /// Which [`LANGUAGES`] slot [`Control::name`] and [`Control::publisher`]
    /// came from.
    pub language: &'static str,
    pub name: String,
    pub publisher: String,
    /// The icon as stored — a JPEG on every retail title seen so far, but
    /// served with a sniffed type rather than an assumed one. Empty when the
    /// Control NCA carries no icon at all.
    pub icon: Vec<u8>,
    /// Everything else the title's NACP declares.
    pub nacp: Nacp,
}

impl Control {
    /// Read the control data out of a Control NCA.
    ///
    /// `nca` is a source over the whole NCA; `keys` needs the header key and
    /// whatever unlocks the section (a key area key, or the title key for a
    /// title-key crypto NCA — resolve that from the container's ticket
    /// first).
    pub fn from_source<S: ByteSource>(nca: S, keys: &KeySet) -> Result<Control, Error> {
        let header = Nca::parse_source(&nca, Some(keys))?;
        if header.content_type != ContentType::Control {
            return Err(Error::Nca(format!(
                "not a Control NCA (content type is {})",
                header.content_type.name()
            )));
        }
        let index = header
            .romfs_section_index()
            .ok_or_else(|| Error::Nca("Control NCA has no RomFS section".into()))?;
        let romfs_source = header.romfs_source(nca, keys, index)?;

        // Unlike a Program NCA's RomFS — the game's data, and the reason
        // `romfs_source` streams at all — a Control NCA's is a handful of
        // icons and a 0x4000-byte NACP, so it is read whole and walked in
        // memory. The bound is what keeps that true: an NCA claiming a
        // game-sized RomFS here is an error, not an allocation.
        const MAX_CONTROL_ROMFS: u64 = 64 * 1024 * 1024;
        if romfs_source.len() > MAX_CONTROL_ROMFS {
            return Err(Error::Nca(format!(
                "Control NCA RomFS is {} bytes, far past anything an icon and a NACP need",
                romfs_source.len()
            )));
        }
        let image = romfs_source.read_vec(0, romfs_source.len())?;
        let romfs = RomFs::parse(&image)?;

        let nacp = Nacp::parse(
            romfs
                .read_path(Nacp::PATH)
                .ok_or_else(|| Error::RomFs(format!("no {} in the Control NCA", Nacp::PATH)))?,
        )?;
        let title = nacp
            .preferred()
            .ok_or_else(|| Error::RomFs("control.nacp has no title in any language".into()))?;

        // Prefer the icon for the language the name came from, so the two
        // agree; a title localized into a language it has no icon for falls
        // back to whichever icon the image does carry.
        let icon = icon_for(&romfs, title.language)
            .or_else(|| any_icon(&romfs))
            .unwrap_or_default()
            .to_vec();
        let (language, name, publisher) =
            (title.language, title.name.clone(), title.publisher.clone());

        Ok(Control {
            title_id: header.title_id,
            language,
            name,
            publisher,
            icon,
            nacp,
        })
    }

    /// Read the control data out of a Control NCA already in memory.
    pub fn from_nca(raw: &[u8], keys: &KeySet) -> Result<Control, Error> {
        Control::from_source(SliceSource(raw), keys)
    }

    /// The icon's media type, sniffed from its magic. Empty when there is no
    /// icon.
    pub fn icon_mime(&self) -> &'static str {
        match self.icon.get(..4) {
            Some([0xFF, 0xD8, 0xFF, _]) => "image/jpeg",
            Some([0x89, b'P', b'N', b'G']) => "image/png",
            _ if self.icon.is_empty() => "",
            _ => "application/octet-stream",
        }
    }
}

/// Find the Control NCA in a PFS0 container's file table, returning its index
/// and parsed header.
///
/// Every `.nca` in the container has to be opened to find it: the file names
/// are content hashes, so which one is the Control NCA is only visible in the
/// (possibly encrypted) header. `keys` therefore needs the header key, or
/// nothing here is readable at all.
pub fn find_control_nca<S: ByteSource>(
    files: &[Pfs0File],
    src: &S,
    keys: &KeySet,
) -> Option<(usize, Nca)> {
    files.iter().enumerate().find_map(|(index, f)| {
        if !f.name.to_ascii_lowercase().ends_with(".nca") {
            return None;
        }
        let window = Window::new(src, f.offset, f.size, &f.name).ok()?;
        let nca = Nca::parse_source(&window, Some(keys)).ok()?;
        (nca.content_type == ContentType::Control).then_some((index, nca))
    })
}

/// Read the control data of the title in a PFS0 container.
pub fn from_pfs0<S: ByteSource>(
    files: &[Pfs0File],
    src: &S,
    keys: &KeySet,
) -> Result<Control, Error> {
    let (index, _) = find_control_nca(files, src, keys)
        .ok_or_else(|| Error::Nca("no Control NCA in this container".into()))?;
    let f = &files[index];
    Control::from_source(Window::new(src, f.offset, f.size, &f.name)?, keys)
}

/// The icon file for one language, under either the current or the legacy
/// spelling of its name.
fn icon_for<'a>(romfs: &RomFs<'a>, language: &str) -> Option<&'a [u8]> {
    let slot = LANGUAGES.iter().position(|l| *l == language)?;
    let legacy = LEGACY_LANGUAGE_NAMES[slot];
    romfs
        .read_path(&format!("/icon_{}.dat", language))
        .or_else(|| {
            if legacy.is_empty() {
                None
            } else {
                romfs.read_path(&format!("/icon_{}.dat", legacy))
            }
        })
}

/// Any icon in the image, for a title whose localized icon is missing.
fn any_icon<'a>(romfs: &RomFs<'a>) -> Option<&'a [u8]> {
    let file = romfs.files().iter().find(|f| {
        let lower = f.path.to_ascii_lowercase();
        lower.starts_with("/icon_") && lower.ends_with(".dat")
    })?;
    romfs.read(file)
}

/// A fixed-width NACP string: UTF-8 up to the first NUL, with anything
/// undecodable dropped rather than failing the whole read.
fn nul_terminated(field: &[u8]) -> String {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    String::from_utf8_lossy(&field[..end]).trim().to_owned()
}

/// The NACP's sizes are signed: an unset one is 0, and a few titles ship a
/// negative value that would read as an implausible size unsigned.
fn read_i64(data: &[u8], at: usize) -> i64 {
    crate::nsp::read_u64(data, at) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NacpBuilder {
        data: Vec<u8>,
    }

    impl NacpBuilder {
        fn new() -> NacpBuilder {
            NacpBuilder { data: vec![0u8; NACP_MIN_SIZE] }
        }

        fn title(mut self, slot: usize, name: &str, publisher: &str) -> NacpBuilder {
            let at = slot * TITLE_ENTRY_SIZE;
            self.data[at..at + name.len()].copy_from_slice(name.as_bytes());
            let at = at + TITLE_NAME_SIZE;
            self.data[at..at + publisher.len()].copy_from_slice(publisher.as_bytes());
            self
        }

        fn text(mut self, at: usize, value: &str) -> NacpBuilder {
            self.data[at..at + value.len()].copy_from_slice(value.as_bytes());
            self
        }

        fn byte(mut self, at: usize, value: u8) -> NacpBuilder {
            self.data[at] = value;
            self
        }

        fn u32(mut self, at: usize, value: u32) -> NacpBuilder {
            self.data[at..at + 4].copy_from_slice(&value.to_le_bytes());
            self
        }

        fn i64(mut self, at: usize, value: i64) -> NacpBuilder {
            self.data[at..at + 8].copy_from_slice(&value.to_le_bytes());
            self
        }

        fn unrated(mut self) -> NacpBuilder {
            self.data[RATING_AGE_OFFSET..RATING_AGE_OFFSET + RATING_AGE_SLOTS]
                .fill(RATING_UNRATED as u8);
            self
        }

        fn build(self) -> Vec<u8> {
            self.data
        }
    }

    #[test]
    fn reads_only_the_localized_slots() {
        let nacp = Nacp::parse(
            &NacpBuilder::new()
                .title(0, "A Game", "A Studio")
                .title(2, "ア ゲーム", "ア スタジオ")
                .text(DISPLAY_VERSION_OFFSET, "1.0.2")
                .unrated()
                .build(),
        )
        .unwrap();
        assert_eq!(nacp.titles.len(), 2);
        assert_eq!(nacp.titles[0].language, "AmericanEnglish");
        assert_eq!(nacp.titles[0].publisher, "A Studio");
        assert_eq!(nacp.titles[1].language, "Japanese");
        assert_eq!(nacp.titles[1].name, "ア ゲーム");
        assert_eq!(nacp.display_version, "1.0.2");
    }

    #[test]
    fn prefers_american_english() {
        let nacp = Nacp::parse(
            &NacpBuilder::new()
                .title(2, "ゲーム", "スタジオ")
                .title(0, "Game", "Studio")
                .unrated()
                .build(),
        )
        .unwrap();
        assert_eq!(nacp.preferred().unwrap().name, "Game");
    }

    #[test]
    fn falls_back_to_the_first_localized_slot() {
        let nacp = Nacp::parse(
            &NacpBuilder::new().title(4, "Ein Spiel", "Ein Studio").unrated().build(),
        )
        .unwrap();
        let title = nacp.preferred().unwrap();
        assert_eq!(title.language, "German");
        assert_eq!(title.name, "Ein Spiel");
        assert_eq!(nacp.display_version, "");
    }

    #[test]
    fn reads_the_ratings_boards_that_rated_it() {
        let nacp = Nacp::parse(
            &NacpBuilder::new()
                .title(0, "Game", "Studio")
                .unrated()
                .byte(RATING_AGE_OFFSET + 3, 10) // ESRB
                .byte(RATING_AGE_OFFSET + 6, 7) // PEGI
                .build(),
        )
        .unwrap();
        assert_eq!(nacp.ratings.len(), 2);
        assert_eq!(nacp.ratings[0].organisation, "ESRB");
        assert_eq!(nacp.ratings[0].age, 10);
        assert_eq!(nacp.ratings[1].organisation, "PEGI");
        assert_eq!(nacp.ratings[1].age, 7);
    }

    #[test]
    fn reads_the_flags_and_sizes() {
        let nacp = Nacp::parse(
            &NacpBuilder::new()
                .title(0, "Game", "Studio")
                .unrated()
                .text(ISBN_OFFSET, "9781234567897")
                .byte(STARTUP_USER_ACCOUNT_OFFSET, 1)
                .u32(ATTRIBUTE_FLAG_OFFSET, ATTRIBUTE_DEMO)
                .byte(SCREENSHOT_OFFSET, 1)
                .byte(VIDEO_CAPTURE_OFFSET, 2)
                .i64(USER_ACCOUNT_SAVE_DATA_SIZE_OFFSET, 0x100_0000)
                .i64(USER_ACCOUNT_SAVE_DATA_JOURNAL_SIZE_OFFSET, 0x20_0000)
                .i64(BCAT_STORAGE_SIZE_OFFSET, 0x40_0000)
                .text(ERROR_CODE_CATEGORY_OFFSET, "2181")
                .build(),
        )
        .unwrap();
        assert_eq!(nacp.isbn, "9781234567897");
        assert!(nacp.is_demo);
        assert_eq!(nacp.startup_user_account, StartupUserAccount::Required);
        assert_eq!(nacp.screenshot.name(), "blocked");
        assert_eq!(nacp.video_capture.name(), "enabled");
        assert_eq!(nacp.user_account_save_data_size, 0x100_0000);
        assert_eq!(nacp.user_account_save_data_journal_size, 0x20_0000);
        assert_eq!(nacp.bcat_delivery_cache_storage_size, 0x40_0000);
        assert_eq!(nacp.application_error_code_category, "2181");
    }

    #[test]
    fn a_title_with_no_ratings_reports_none() {
        let nacp =
            Nacp::parse(&NacpBuilder::new().title(0, "Game", "Studio").unrated().build()).unwrap();
        assert!(nacp.ratings.is_empty());
        assert!(!nacp.is_demo);
        assert_eq!(nacp.startup_user_account, StartupUserAccount::None);
        assert_eq!(nacp.screenshot.name(), "allowed");
    }

    #[test]
    fn rejects_a_truncated_nacp() {
        assert!(matches!(Nacp::parse(&[0u8; 0x100]), Err(Error::Truncated { .. })));
    }

    #[test]
    fn icon_mime_is_sniffed_not_assumed() {
        let nacp = Nacp::parse(&NacpBuilder::new().title(0, "Game", "Studio").unrated().build())
            .unwrap();
        let mut control = Control {
            title_id: 0,
            language: LANGUAGES[0],
            name: "Game".into(),
            publisher: "Studio".into(),
            icon: vec![0xFF, 0xD8, 0xFF, 0xE0],
            nacp,
        };
        assert_eq!(control.icon_mime(), "image/jpeg");
        control.icon = vec![0x89, b'P', b'N', b'G'];
        assert_eq!(control.icon_mime(), "image/png");
        control.icon.clear();
        assert_eq!(control.icon_mime(), "");
    }
}
