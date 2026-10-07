//! A title's control data: the NACP (name, publisher, metadata) and icon, read
//! from a Control NCA or a homebrew NRO's asset section.

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

/// Legacy names still used for the last two slots' icon files, indexed with [`LANGUAGES`].
const LEGACY_LANGUAGE_NAMES: [&str; 16] = [
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "Taiwanese",
    "Chinese",
    "",
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
/// Last field required; fields past this are optional (a real NACP is 0x4000 bytes).
const NACP_MIN_SIZE: usize = ERROR_CODE_CATEGORY_OFFSET + ERROR_CODE_CATEGORY_SIZE;
/// Ceilings each save may be extended to.
const USER_ACCOUNT_SAVE_DATA_SIZE_MAX_OFFSET: usize = 0x3148;
const USER_ACCOUNT_SAVE_DATA_JOURNAL_SIZE_MAX_OFFSET: usize = 0x3150;
const DEVICE_SAVE_DATA_SIZE_MAX_OFFSET: usize = 0x3158;
const DEVICE_SAVE_DATA_JOURNAL_SIZE_MAX_OFFSET: usize = 0x3160;
/// Cache storage: deletable scratch space, addressed by index.
const CACHE_STORAGE_SIZE_OFFSET: usize = 0x3170;
const CACHE_STORAGE_JOURNAL_SIZE_OFFSET: usize = 0x3178;
const CACHE_STORAGE_DATA_AND_JOURNAL_SIZE_MAX_OFFSET: usize = 0x3180;
const CACHE_STORAGE_INDEX_MAX_OFFSET: usize = 0x3188;

/// `AttributeFlag` bit 0: the title is a demo build.
const ATTRIBUTE_DEMO: u32 = 1 << 0;
/// An age-rating slot the title wasn't submitted to.
const RATING_UNRATED: i8 = -1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Title {
    pub language: &'static str,
    pub name: String,
    pub publisher: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rating {
    pub organisation: &'static str,
    pub age: u8,
}

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
    /// Screenshot: 0 allowed, 1 blocked. Video: 0 disabled, 1 manual, 2 enabled.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nacp {
    /// Only the slots the title is localized into.
    pub titles: Vec<Title>,
    /// Display version string (`1.0.2`), not the NCA version.
    pub display_version: String,
    pub isbn: String,
    pub is_demo: bool,
    pub startup_user_account: StartupUserAccount,
    pub screenshot: CaptureSupport,
    pub video_capture: CaptureSupport,
    /// Only the boards that rated the title.
    pub ratings: Vec<Rating>,
    /// 0 when the title has no DLC.
    pub add_on_content_base_id: u64,
    /// Title whose save data this one shares, 0 when it owns its own.
    pub save_data_owner_id: u64,
    pub user_account_save_data_size: i64,
    pub user_account_save_data_journal_size: i64,
    pub device_save_data_size: i64,
    pub device_save_data_journal_size: i64,
    pub bcat_delivery_cache_storage_size: i64,
    /// 0 when unset or past the end of a short NACP.
    pub user_account_save_data_size_max: i64,
    pub user_account_save_data_journal_size_max: i64,
    pub device_save_data_size_max: i64,
    pub device_save_data_journal_size_max: i64,
    pub cache_storage_size: i64,
    pub cache_storage_journal_size: i64,
    pub cache_storage_data_and_journal_size_max: i64,
    pub cache_storage_index_max: u16,
    /// Error code prefix (`2181` in `2181-0002`), empty for the system's.
    pub application_error_code_category: String,
}

impl Nacp {
    pub const PATH: &'static str = "/control.nacp";

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
            titles.push(Title {
                language,
                name,
                publisher,
            });
        }

        let mut ratings = Vec::new();
        for (slot, organisation) in RATING_ORGANISATIONS.iter().enumerate() {
            // 32 slots for 13 named boards; the rest are always unrated.
            debug_assert!(slot < RATING_AGE_SLOTS);
            let age = data[RATING_AGE_OFFSET + slot] as i8;
            if age != RATING_UNRATED {
                ratings.push(Rating {
                    organisation,
                    age: age as u8,
                });
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
            user_account_save_data_size_max: read_i64_if_present(
                data,
                USER_ACCOUNT_SAVE_DATA_SIZE_MAX_OFFSET,
            ),
            user_account_save_data_journal_size_max: read_i64_if_present(
                data,
                USER_ACCOUNT_SAVE_DATA_JOURNAL_SIZE_MAX_OFFSET,
            ),
            device_save_data_size_max: read_i64_if_present(data, DEVICE_SAVE_DATA_SIZE_MAX_OFFSET),
            device_save_data_journal_size_max: read_i64_if_present(
                data,
                DEVICE_SAVE_DATA_JOURNAL_SIZE_MAX_OFFSET,
            ),
            cache_storage_size: read_i64_if_present(data, CACHE_STORAGE_SIZE_OFFSET),
            cache_storage_journal_size: read_i64_if_present(
                data,
                CACHE_STORAGE_JOURNAL_SIZE_OFFSET,
            ),
            cache_storage_data_and_journal_size_max: read_i64_if_present(
                data,
                CACHE_STORAGE_DATA_AND_JOURNAL_SIZE_MAX_OFFSET,
            ),
            cache_storage_index_max: match data.len() >= CACHE_STORAGE_INDEX_MAX_OFFSET + 2 {
                true => u16::from_le_bytes([
                    data[CACHE_STORAGE_INDEX_MAX_OFFSET],
                    data[CACHE_STORAGE_INDEX_MAX_OFFSET + 1],
                ]),
                false => 0,
            },
            application_error_code_category: nul_terminated(
                &data[ERROR_CODE_CATEGORY_OFFSET
                    ..ERROR_CODE_CATEGORY_OFFSET + ERROR_CODE_CATEGORY_SIZE],
            ),
        })
    }

    /// What a program with no NACP declares: every figure 0.
    pub fn empty() -> Nacp {
        Nacp {
            titles: Vec::new(),
            display_version: String::new(),
            isbn: String::new(),
            is_demo: false,
            startup_user_account: StartupUserAccount::None,
            screenshot: CaptureSupport::Screenshot(0),
            video_capture: CaptureSupport::Video(0),
            ratings: Vec::new(),
            add_on_content_base_id: 0,
            save_data_owner_id: 0,
            user_account_save_data_size: 0,
            user_account_save_data_journal_size: 0,
            device_save_data_size: 0,
            device_save_data_journal_size: 0,
            bcat_delivery_cache_storage_size: 0,
            user_account_save_data_size_max: 0,
            user_account_save_data_journal_size_max: 0,
            device_save_data_size_max: 0,
            device_save_data_journal_size_max: 0,
            cache_storage_size: 0,
            cache_storage_journal_size: 0,
            cache_storage_data_and_journal_size_max: 0,
            cache_storage_index_max: 0,
            application_error_code_category: String::new(),
        }
    }

    /// American English if present, else the first localized slot.
    pub fn preferred(&self) -> Option<&Title> {
        self.titles
            .iter()
            .find(|t| t.language == LANGUAGES[0])
            .or_else(|| self.titles.first())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Control {
    pub title_id: u64,
    pub language: &'static str,
    pub name: String,
    pub publisher: String,
    /// The icon as stored (JPEG on retail), empty when absent.
    pub icon: Vec<u8>,
    pub nacp: Nacp,
}

impl Control {
    /// Read the control data out of a Control NCA. `keys` must resolve the
    /// section key (the title key for title-key crypto, from the ticket).
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

        // A Control RomFS is small and read whole; reject one claiming game size.
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

    pub fn from_nca(raw: &[u8], keys: &KeySet) -> Result<Control, Error> {
        Control::from_source(SliceSource(raw), keys)
    }

    /// Read the `control.nacp` and icon from a homebrew NRO's asset section.
    pub fn from_nro(data: &[u8]) -> Option<Control> {
        let assets = crate::nro::assets(data)?;
        if assets.icon.is_empty() && assets.nacp.is_empty() {
            return None;
        }
        // A NACP that fails to parse still leaves a usable icon.
        let nacp = Nacp::parse(assets.nacp).unwrap_or_else(|_| Nacp::empty());
        let title = nacp.preferred();
        Some(Control {
            title_id: 0,
            language: title.map_or("", |t| t.language),
            name: title.map_or(String::new(), |t| t.name.clone()),
            publisher: title.map_or(String::new(), |t| t.publisher.clone()),
            icon: assets.icon.to_vec(),
            nacp,
        })
    }

    pub fn icon_mime(&self) -> &'static str {
        match self.icon.get(..4) {
            Some([0xFF, 0xD8, 0xFF, _]) => "image/jpeg",
            Some([0x89, b'P', b'N', b'G']) => "image/png",
            _ if self.icon.is_empty() => "",
            _ => "application/octet-stream",
        }
    }
}

pub fn find_control_nca<S: ByteSource>(
    files: &[Pfs0File],
    src: &S,
    keys: &KeySet,
) -> Option<(usize, Nca)> {
    crate::nca::find_nca_by_type(files, src, keys, ContentType::Control)
}

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

/// UTF-8 up to the first NUL, undecodable bytes replaced.
fn nul_terminated(field: &[u8]) -> String {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    String::from_utf8_lossy(&field[..end]).trim().to_owned()
}

/// NACP sizes are signed; some titles ship negative values.
fn read_i64(data: &[u8], at: usize) -> i64 {
    crate::nsp::read_u64(data, at) as i64
}

/// 0 when the NACP is too short to hold the field.
fn read_i64_if_present(data: &[u8], at: usize) -> i64 {
    match data.len() >= at + 8 {
        true => read_i64(data, at),
        false => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NacpBuilder {
        data: Vec<u8>,
    }

    impl NacpBuilder {
        fn new() -> NacpBuilder {
            NacpBuilder {
                data: vec![0u8; NACP_MIN_SIZE],
            }
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

        fn u16(mut self, at: usize, value: u16) -> NacpBuilder {
            self.data[at..at + 2].copy_from_slice(&value.to_le_bytes());
            self
        }

        /// A full 0x4000-byte NACP, unlike [`NacpBuilder::new`].
        fn full() -> NacpBuilder {
            NacpBuilder {
                data: vec![0u8; 0x4000],
            }
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
            &NacpBuilder::new()
                .title(4, "Ein Spiel", "Ein Studio")
                .unrated()
                .build(),
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
        let nacp = Nacp::parse(
            &NacpBuilder::new()
                .title(0, "Game", "Studio")
                .unrated()
                .build(),
        )
        .unwrap();
        assert!(nacp.ratings.is_empty());
        assert!(!nacp.is_demo);
        assert_eq!(nacp.startup_user_account, StartupUserAccount::None);
        assert_eq!(nacp.screenshot.name(), "allowed");
    }

    #[test]
    fn rejects_a_truncated_nacp() {
        assert!(matches!(
            Nacp::parse(&[0u8; 0x100]),
            Err(Error::Truncated { .. })
        ));
    }

    #[test]
    fn icon_mime_is_sniffed_not_assumed() {
        let nacp = Nacp::parse(
            &NacpBuilder::new()
                .title(0, "Game", "Studio")
                .unrated()
                .build(),
        )
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

    /// Distinct values so a field read from the wrong offset fails.
    fn nacp_with_save_data() -> Vec<u8> {
        NacpBuilder::full()
            .title(0, "Game", "Studio")
            .unrated()
            .i64(USER_ACCOUNT_SAVE_DATA_SIZE_OFFSET, 0x100_0000)
            .i64(USER_ACCOUNT_SAVE_DATA_JOURNAL_SIZE_OFFSET, 0x20_0000)
            .i64(USER_ACCOUNT_SAVE_DATA_SIZE_MAX_OFFSET, 0x200_0000)
            .i64(USER_ACCOUNT_SAVE_DATA_JOURNAL_SIZE_MAX_OFFSET, 0x40_0000)
            .i64(DEVICE_SAVE_DATA_SIZE_MAX_OFFSET, 0x300_0000)
            .i64(DEVICE_SAVE_DATA_JOURNAL_SIZE_MAX_OFFSET, 0x50_0000)
            .i64(CACHE_STORAGE_SIZE_OFFSET, 0x10_0000)
            .i64(CACHE_STORAGE_JOURNAL_SIZE_OFFSET, 0x8_0000)
            .i64(CACHE_STORAGE_DATA_AND_JOURNAL_SIZE_MAX_OFFSET, 0x400_0000)
            .u16(CACHE_STORAGE_INDEX_MAX_OFFSET, 3)
            .build()
    }

    #[test]
    fn every_save_data_figure_a_nacp_declares_reaches_the_cpu() {
        let nacp = Nacp::parse(&nacp_with_save_data()).unwrap();
        let quota = crate::cpu::SaveDataQuota::from(&nacp);
        assert_eq!(quota.size, 0x100_0000);
        assert_eq!(quota.journal_size, 0x20_0000);
        assert_eq!(quota.size_max, 0x200_0000);
        assert_eq!(quota.journal_size_max, 0x40_0000);
        assert_eq!(quota.device_size_max, 0x300_0000);
        assert_eq!(quota.device_journal_size_max, 0x50_0000);
        assert_eq!(quota.cache_storage_size_max, 0x400_0000);
        assert_eq!(quota.cache_storage_index_max, 3);
    }

    /// A minimal NRO with an asset section appended.
    fn nro_with_assets(icon: &[u8], nacp: &[u8]) -> Vec<u8> {
        const HEADER_SIZE: u32 = 0x50;
        let mut out = vec![0u8; HEADER_SIZE as usize];
        out[0..4].copy_from_slice(&crate::nro::NRO0_MAGIC.to_le_bytes());
        out[4..8].copy_from_slice(&1u32.to_le_bytes());
        out[8..12].copy_from_slice(&HEADER_SIZE.to_le_bytes());
        out.extend_from_slice(&crate::nro::ASET_MAGIC.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        let mut offset = 0x38u64;
        for part in [icon, nacp, &[][..]] {
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&(part.len() as u64).to_le_bytes());
            offset += part.len() as u64;
        }
        out.extend_from_slice(icon);
        out.extend_from_slice(nacp);
        out
    }

    #[test]
    fn reads_a_homebrew_nros_own_name_and_icon() {
        let nacp = NacpBuilder::new()
            .title(0, "NX-Shell", "DefenderOfHyrule")
            .text(DISPLAY_VERSION_OFFSET, "4.0.2")
            .unrated()
            .build();
        let control = Control::from_nro(&nro_with_assets(b"\xFF\xD8\xFF\xE0jpeg", &nacp)).unwrap();
        assert_eq!(control.name, "NX-Shell");
        assert_eq!(control.publisher, "DefenderOfHyrule");
        assert_eq!(control.language, "AmericanEnglish");
        assert_eq!(control.nacp.display_version, "4.0.2");
        assert_eq!(control.icon, b"\xFF\xD8\xFF\xE0jpeg");
        assert_eq!(control.icon_mime(), "image/jpeg");
        assert_eq!(control.title_id, 0);
    }

    #[test]
    fn an_nro_with_an_icon_and_no_nacp_keeps_the_icon() {
        let control = Control::from_nro(&nro_with_assets(b"\x89PNGicon", b"")).unwrap();
        assert_eq!(control.icon_mime(), "image/png");
        assert!(control.name.is_empty());
        assert!(control.nacp.titles.is_empty());
    }

    #[test]
    fn an_nro_with_nothing_appended_has_no_control() {
        assert_eq!(Control::from_nro(&nro_with_assets(b"", b"")), None);
        assert_eq!(Control::from_nro(&[0u8; 0x50]), None);
    }

    #[test]
    fn a_nacp_that_stops_before_the_ceilings_still_parses() {
        // A short NACP reports 0 for the optional fields.
        let short = NacpBuilder::new()
            .title(0, "Game", "Studio")
            .unrated()
            .i64(USER_ACCOUNT_SAVE_DATA_SIZE_OFFSET, 0x100_0000)
            .build();
        assert!(short.len() < USER_ACCOUNT_SAVE_DATA_SIZE_MAX_OFFSET);
        let nacp = Nacp::parse(&short).unwrap();
        assert_eq!(nacp.user_account_save_data_size, 0x100_0000);
        assert_eq!(nacp.user_account_save_data_size_max, 0);
        assert_eq!(nacp.cache_storage_index_max, 0);
    }
}
