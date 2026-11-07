//! `ssl`: contexts and options are real objects; connections never connect.

use crate::cpu::Cpu;
use crate::Result;

#[derive(Debug)]
pub(crate) struct SslCertificate {
    /// `CaCertificateId`; -1 is `All`.
    id: i32,
    /// `TrustedCertStatus`, passed through as stored.
    status: u32,
    der: Vec<u8>,
}

/// The firmware system data archive holding the certificate store, and the file in it.
const CERT_STORE_DATA_ID: u64 = 0x0100_0000_0000_0800;

const CERT_STORE_PATH: &str = "/ssl_TrustedCerts.bdf";

/// `CaCertificateId_All`.
const CERT_ID_ALL: i32 = -1;

/// `BuiltInCertificateInfo`, one per certificate plus a terminator, ahead of the DER bytes.
const CERT_INFO_SIZE: u32 = 0x18;

/// Parse `ssl_TrustedCerts.bdf`: an `sslT` header with a count, 0x10-byte entries,
/// then DER bodies. Offsets are from the end of the header.
fn parse_cert_store(data: &[u8]) -> Vec<SslCertificate> {
    const MAGIC: u32 = u32::from_le_bytes(*b"sslT");
    const HEADER_SIZE: usize = 8;
    const ENTRY_SIZE: usize = 0x10;
    let read_u32 = |at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?))
    };
    if read_u32(0) != Some(MAGIC) {
        return Vec::new();
    }
    let Some(count) = read_u32(4) else {
        return Vec::new();
    };
    let mut certs = Vec::new();
    for index in 0..count as usize {
        let at = HEADER_SIZE + index * ENTRY_SIZE;
        let (Some(id), Some(status), Some(size), Some(offset)) = (
            read_u32(at),
            read_u32(at + 4),
            read_u32(at + 8),
            read_u32(at + 12),
        ) else {
            break;
        };
        let from = HEADER_SIZE + offset as usize;
        let Some(der) = data.get(from..from.saturating_add(size as usize)) else {
            continue;
        };
        certs.push(SslCertificate {
            id: id as i32,
            status,
            der: der.to_vec(),
        });
    }
    certs
}

impl Cpu {
    /// `ssl`: contexts and options are real objects; connections never connect.
    pub(crate) fn ssl_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "ssl:service");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "ssl:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("ssl:service")
                .to_string()
        } else {
            match self.service_name(handle) {
                Some("ssl") | None => "ssl:service".to_string(),
                Some(name) => name.to_string(),
            }
        };
        let data = self.ipc_request_data(tls);
        match iface.as_str() {
            "ssl:service" => match cmd_id {
                // CreateContext(SslVersion, pid placeholder) -> ISslContext.
                Some(0) => {
                    self.process_mut().ssl_contexts += 1;
                    self.reply_with_interface(tls, handle, "ssl:context")?;
                    Ok(())
                }
                // GetContextCount.
                Some(1) => {
                    let count = self.process().ssl_contexts;
                    self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
                }
                // SetInterfaceVersion(u32).
                Some(5) => {
                    self.process_mut().ssl_interface_version = self.mem.read_u32(data)?;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetCertificateBufSize(ids in a buffer) -> u32 size.
                Some(3) => {
                    let ids = self.ssl_requested_ids(tls);
                    let (size, _) = self.ssl_certificate_extent(&ids);
                    self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
                }
                // GetCertificates(ids) -> u32 count, with records, a terminator and the DER
                // bodies in an out buffer.
                Some(2) => {
                    let ids = self.ssl_requested_ids(tls);
                    let count = self.ssl_write_certificates(tls, &ids)?;
                    self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
                }
                // FlushSessionCache.
                Some(6) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "ssl:context" => match cmd_id {
                // Set/GetOption(SslContextOption, s32), stored per context.
                Some(0) => {
                    let option = self.mem.read_u32(data)?;
                    let value = self.mem.read_u32(data.wrapping_add(4))?;
                    let key = Self::object_key(handle, object_id);
                    self.process_mut().ssl_options.insert((key, option), value);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(1) => {
                    let option = self.mem.read_u32(data)?;
                    let key = Self::object_key(handle, object_id);
                    let value = self
                        .process()
                        .ssl_options
                        .get(&(key, option))
                        .copied()
                        .unwrap_or(0);
                    self.write_ipc_response(tls, 0, &[], &value.to_le_bytes(), &[])
                }
                // GetConnectionCount: none.
                Some(3) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // ImportServerPki / ImportClientPki(format, certificates) -> u64 id. Accepted, not kept.
                Some(4) | Some(5) => {
                    let id = self.ssl_next_pki_id;
                    self.ssl_next_pki_id = id.wrapping_add(1);
                    self.write_ipc_response(tls, 0, &[], &id.to_le_bytes(), &[])
                }
                // RemoveServerPki / RemoveClientPki(id).
                Some(6) | Some(7) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// The `CaCertificateId`s a request named. A single `All`, or no buffer, means every root.
    fn ssl_requested_ids(&mut self, tls: u32) -> Vec<i32> {
        let Some((addr, len)) = self.ipc_input_buffer(tls, 0) else {
            return vec![CERT_ID_ALL];
        };
        let mut ids = Vec::new();
        for index in 0..len / 4 {
            match self.mem.read_u32(addr.wrapping_add(index * 4)) {
                Ok(id) => ids.push(id as i32),
                Err(_) => break,
            }
        }
        if ids.is_empty() {
            ids.push(CERT_ID_ALL);
        }
        ids
    }

    fn ssl_certificate_store(&mut self) -> &[SslCertificate] {
        if self.ssl_certificates.is_none() {
            let mut certs = Vec::new();
            if let Some(src) = self.data_archives.get(&CERT_STORE_DATA_ID) {
                let mut image = vec![0u8; src.len() as usize];
                if src.read_at(0, &mut image).is_ok() {
                    if let Some(file) = crate::romfs::RomFs::parse(&image)
                        .ok()
                        .and_then(|romfs| romfs.read_path(CERT_STORE_PATH))
                    {
                        certs = parse_cert_store(file);
                    }
                }
            }
            if certs.is_empty() {
                self.log(
                    crate::trace::Level::Warn,
                    "[ssl] no certificate store: a browser aborts without one; register a \
                     firmware directory",
                );
            }
            self.ssl_certificates = Some(certs);
        }
        self.ssl_certificates.as_deref().unwrap_or(&[])
    }

    fn ssl_wanted(ids: &[i32], cert: &SslCertificate) -> bool {
        ids == [CERT_ID_ALL] || ids.contains(&cert.id)
    }

    /// The size `GetCertificates` needs for `ids`, and how many it writes. DER bodies are
    /// padded to 4 bytes; the terminator is always counted.
    fn ssl_certificate_extent(&mut self, ids: &[i32]) -> (u32, u32) {
        let mut size = CERT_INFO_SIZE;
        let mut count = 0u32;
        let store = self.ssl_certificate_store();
        for cert in store {
            if !Self::ssl_wanted(ids, cert) {
                continue;
            }
            size += CERT_INFO_SIZE + (cert.der.len() as u32).next_multiple_of(4);
            count += 1;
        }
        (size, count)
    }

    /// Write the records and DER bodies, returning the count. A buffer too small for
    /// [`Cpu::ssl_certificate_extent`] is left alone.
    fn ssl_write_certificates(&mut self, tls: u32, ids: &[i32]) -> Result<u32> {
        let (size, count) = self.ssl_certificate_extent(ids);
        let Some((addr, len)) = self.ipc_output_buffer(tls, 0) else {
            return Ok(0);
        };
        if len < size {
            return Ok(0);
        }
        let mut der_at = (count + 1) * CERT_INFO_SIZE;
        let mut info_at = 0u32;
        let store = std::mem::take(&mut self.ssl_certificates).unwrap_or_default();
        for cert in &store {
            if !Self::ssl_wanted(ids, cert) {
                continue;
            }
            let mut info = [0u8; CERT_INFO_SIZE as usize];
            info[0..4].copy_from_slice(&cert.id.to_le_bytes());
            info[4..8].copy_from_slice(&cert.status.to_le_bytes());
            info[8..16].copy_from_slice(&(cert.der.len() as u64).to_le_bytes());
            info[16..24].copy_from_slice(&u64::from(der_at).to_le_bytes());
            for (index, &byte) in info.iter().enumerate() {
                self.mem
                    .write_u8(addr.wrapping_add(info_at + index as u32), byte)?;
            }
            for (index, &byte) in cert.der.iter().enumerate() {
                self.mem
                    .write_u8(addr.wrapping_add(der_at + index as u32), byte)?;
            }
            info_at += CERT_INFO_SIZE;
            der_at += (cert.der.len() as u32).next_multiple_of(4);
        }
        self.ssl_certificates = Some(store);
        // The terminator: `CaCertificateId_All` with an empty body.
        let mut end = [0u8; CERT_INFO_SIZE as usize];
        end[0..4].copy_from_slice(&CERT_ID_ALL.to_le_bytes());
        for (index, &byte) in end.iter().enumerate() {
            self.mem
                .write_u8(addr.wrapping_add(info_at + index as u32), byte)?;
        }
        Ok(count)
    }
}
