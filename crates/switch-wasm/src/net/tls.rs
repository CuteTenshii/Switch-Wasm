//! TLS the page runs for the guest: one `rustls` client connection per stream the guest
//! started TLS on. The core sees plaintext only; the relay sees ciphertext only.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::sync::Arc;

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::client::WebPkiServerVerifier;
use rustls::crypto::{ring, verify_tls12_signature, verify_tls13_signature, CryptoProvider};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{
    AlertDescription, CertificateError, ClientConfig, ClientConnection, DigitallySignedStruct,
    Error, InvalidMessage, PeerIncompatible, ProtocolVersion, RootCertStore, SignatureScheme,
};
use switch_core::services::netbridge::TlsFailure;

/// `VerifyOption` bits.
const VERIFY_PEER_CA: u32 = 1 << 0;
const VERIFY_HOST_NAME: u32 = 1 << 1;
const VERIFY_DATE_CHECK: u32 = 1 << 2;

/// What the guest asked to have checked about the server's certificate.
#[derive(Debug)]
struct Policy {
    /// `None` when there are no roots, so nothing can be verified against them.
    chain: Option<Arc<WebPkiServerVerifier>>,
    provider: Arc<CryptoProvider>,
    peer_ca: bool,
    host_name: bool,
    date_check: bool,
}

impl ServerCertVerifier for Policy {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        if !self.peer_ca {
            return Ok(ServerCertVerified::assertion());
        }
        let Some(chain) = &self.chain else {
            return Err(Error::InvalidCertificate(CertificateError::UnknownIssuer));
        };
        match chain.verify_server_cert(end_entity, intermediates, server_name, ocsp, now) {
            // The guest chose not to check the name, or the dates: that part is let through.
            Err(Error::InvalidCertificate(
                CertificateError::NotValidForName | CertificateError::NotValidForNameContext { .. },
            )) if !self.host_name => Ok(ServerCertVerified::assertion()),
            Err(Error::InvalidCertificate(
                CertificateError::Expired
                | CertificateError::ExpiredContext { .. }
                | CertificateError::NotValidYet
                | CertificateError::NotValidYetContext { .. },
            )) if !self.date_check => Ok(ServerCertVerified::assertion()),
            other => other,
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// The trusted roots from DER certificates; ones that do not parse are skipped.
pub(crate) fn roots(der: &[Vec<u8>]) -> Arc<RootCertStore> {
    let mut store = RootCertStore::empty();
    for certificate in der {
        let _ = store.add(CertificateDer::from(certificate.clone()));
    }
    Arc::new(store)
}

/// How the handshake went wrong, as the guest tells failures apart.
fn classify(error: &Error) -> TlsFailure {
    match error {
        Error::InvalidCertificate(
            CertificateError::UnknownIssuer | CertificateError::UnhandledCriticalExtension,
        ) => TlsFailure::UnknownCa,
        Error::InvalidCertificate(
            CertificateError::Expired
            | CertificateError::ExpiredContext { .. }
            | CertificateError::NotValidYet
            | CertificateError::NotValidYetContext { .. },
        ) => TlsFailure::Expired,
        Error::InvalidCertificate(
            CertificateError::NotValidForName | CertificateError::NotValidForNameContext { .. },
        ) => TlsFailure::NotValidForName,
        Error::InvalidCertificate(_) | Error::NoCertificatesPresented => TlsFailure::BadCertificate,
        Error::PeerIncompatible(
            PeerIncompatible::Tls12NotOffered | PeerIncompatible::Tls13RequiredForQuic,
        )
        | Error::AlertReceived(AlertDescription::ProtocolVersion) => TlsFailure::ProtocolVersion,
        Error::InvalidMessage(InvalidMessage::MessageTooLarge) => TlsFailure::Handshake,
        _ => TlsFailure::Handshake,
    }
}

/// What the handshake negotiated.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Established {
    pub protocol: String,
    pub cipher: String,
    pub alpn: Option<Vec<u8>>,
    pub chain: Vec<Vec<u8>>,
}

/// What a batch of ciphertext from the relay came to.
#[derive(Debug, Default)]
pub(crate) struct Fed {
    /// Decrypted bytes for the guest.
    pub plaintext: Vec<u8>,
    /// Ciphertext for the relay: handshake messages, alerts.
    pub outgoing: Vec<u8>,
    /// Ciphertext bytes the relay can be credited for now.
    pub credit: u32,
    pub established: Option<Established>,
    /// The handshake failed, or the connection broke afterwards (`established` was earlier).
    pub failure: Option<TlsFailure>,
    /// The server sent close_notify.
    pub closed: bool,
}

pub(crate) struct TlsStream {
    connection: ClientConnection,
    established: bool,
    failed: bool,
    /// Plaintext the guest has not read and the ciphertext it came from, oldest first.
    owed: VecDeque<(u32, u32)>,
}

impl TlsStream {
    /// A connection to `server_name` and the ClientHello to send. `verify` is the guest's
    /// `VerifyOption`, `alpn` the protocols to offer.
    pub fn new(
        server_name: &str,
        verify: u32,
        alpn: Vec<Vec<u8>>,
        roots: &Arc<RootCertStore>,
    ) -> Result<(TlsStream, Vec<u8>), TlsFailure> {
        let provider = Arc::new(ring::default_provider());
        let chain = if roots.is_empty() {
            None
        } else {
            WebPkiServerVerifier::builder_with_provider(roots.clone(), provider.clone())
                .build()
                .ok()
        };
        let policy = Policy {
            chain,
            provider: provider.clone(),
            peer_ca: verify & VERIFY_PEER_CA != 0,
            host_name: verify & VERIFY_HOST_NAME != 0,
            date_check: verify & VERIFY_DATE_CHECK != 0,
        };
        let mut config = ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|_| TlsFailure::ProtocolVersion)?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(policy))
            .with_no_client_auth();
        config.alpn_protocols = alpn;
        let name =
            ServerName::try_from(server_name.to_owned()).map_err(|_| TlsFailure::Handshake)?;
        let connection =
            ClientConnection::new(Arc::new(config), name).map_err(|_| TlsFailure::Handshake)?;
        let mut stream = TlsStream {
            connection,
            established: false,
            failed: false,
            owed: VecDeque::new(),
        };
        let hello = stream.take_outgoing();
        Ok((stream, hello))
    }

    pub fn is_established(&self) -> bool {
        self.established
    }

    /// Everything the connection has to send.
    fn take_outgoing(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        while self.connection.wants_write() {
            if self.connection.write_tls(&mut out).is_err() {
                break;
            }
        }
        out
    }

    /// Ciphertext from the relay.
    pub fn feed(&mut self, ciphertext: &[u8]) -> Fed {
        let mut fed = Fed::default();
        if self.failed {
            return fed;
        }
        let mut remaining = ciphertext;
        while !remaining.is_empty() {
            match self.connection.read_tls(&mut remaining) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            if let Err(error) = self.connection.process_new_packets() {
                self.failed = true;
                fed.failure = Some(classify(&error));
                break;
            }
            self.read_plaintext(&mut fed);
        }
        fed.outgoing = self.take_outgoing();
        if !self.established && !self.failed && !self.connection.is_handshaking() {
            self.established = true;
            fed.established = Some(self.describe());
        }
        let cost = ciphertext.len() as u32;
        if fed.plaintext.is_empty() {
            fed.credit = cost;
        } else {
            self.owed.push_back((fed.plaintext.len() as u32, cost));
        }
        fed
    }

    fn read_plaintext(&mut self, fed: &mut Fed) {
        let mut buffer = [0u8; 16 * 1024];
        loop {
            match self.connection.reader().read(&mut buffer) {
                Ok(0) => {
                    fed.closed = true;
                    break;
                }
                Ok(read) => fed.plaintext.extend_from_slice(&buffer[..read]),
                Err(_) => break,
            }
        }
    }

    fn describe(&self) -> Established {
        let connection = &self.connection;
        Established {
            protocol: match connection.protocol_version() {
                Some(ProtocolVersion::TLSv1_3) => "TLSv1.3".to_owned(),
                Some(ProtocolVersion::TLSv1_2) => "TLSv1.2".to_owned(),
                Some(other) => format!("{other:?}"),
                None => String::new(),
            },
            cipher: connection
                .negotiated_cipher_suite()
                .map(|suite| format!("{:?}", suite.suite()))
                .unwrap_or_default(),
            alpn: connection.alpn_protocol().map(<[u8]>::to_vec),
            chain: connection
                .peer_certificates()
                .map(|chain| chain.iter().map(|der| der.as_ref().to_vec()).collect())
                .unwrap_or_default(),
        }
    }

    /// Plaintext from the guest, as ciphertext for the relay.
    pub fn write(&mut self, plaintext: &[u8]) -> Vec<u8> {
        if !self.failed && self.connection.writer().write_all(plaintext).is_err() {
            self.failed = true;
        }
        self.take_outgoing()
    }

    /// A close_notify for the relay.
    pub fn close_notify(&mut self) -> Vec<u8> {
        self.connection.send_close_notify();
        self.take_outgoing()
    }

    /// The guest read `plaintext` bytes: the ciphertext they cost, now the relay's to send more of.
    pub fn consumed(&mut self, mut plaintext: u32) -> u32 {
        let mut credit = 0;
        while plaintext > 0 {
            let Some((left, cost)) = self.owed.front_mut() else {
                break;
            };
            let take = plaintext.min(*left);
            let give = if take == *left {
                *cost
            } else {
                (u64::from(*cost) * u64::from(take) / u64::from(*left)) as u32
            };
            *left -= take;
            *cost -= give;
            credit += give;
            plaintext -= take;
            if *left == 0 {
                self.owed.pop_front();
            }
        }
        credit
    }
}

/// A `rustls` server to run the client against, with the test certificates in `testdata/`
/// (a CA, a `localhost` certificate it signed, and the same one long expired).
#[cfg(test)]
pub(super) mod harness {
    use super::*;
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::PrivateKeyDer;
    use rustls::{ServerConfig, ServerConnection};

    const CA: &[u8] = include_bytes!("testdata/ca.pem");
    pub const SERVER: &[u8] = include_bytes!("testdata/server.pem");
    pub const EXPIRED: &[u8] = include_bytes!("testdata/expired.pem");
    const KEY: &[u8] = include_bytes!("testdata/server.key.pem");

    pub fn ca_der() -> Vec<u8> {
        CertificateDer::from_pem_slice(CA).unwrap().to_vec()
    }

    pub fn server(certificate: &[u8], alpn: &[&[u8]]) -> ServerConnection {
        let provider = Arc::new(ring::default_provider());
        let mut config = ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(
                vec![CertificateDer::from_pem_slice(certificate).unwrap()],
                PrivateKeyDer::from_pem_slice(KEY).unwrap(),
            )
            .unwrap();
        config.alpn_protocols = alpn.iter().map(|p| p.to_vec()).collect();
        ServerConnection::new(Arc::new(config)).unwrap()
    }

    /// Give the server ciphertext and take what it answers with.
    pub fn serve(server: &mut ServerConnection, mut ciphertext: &[u8]) -> Vec<u8> {
        while !ciphertext.is_empty() {
            if server.read_tls(&mut ciphertext).unwrap() == 0 {
                break;
            }
            let _ = server.process_new_packets();
        }
        let mut out = Vec::new();
        while server.wants_write() {
            server.write_tls(&mut out).unwrap();
        }
        out
    }

    /// Run a handshake to its end between a client and a server, whichever way it goes.
    pub fn handshake(
        client: &mut TlsStream,
        server: &mut ServerConnection,
        hello: Vec<u8>,
    ) -> Result<Established, TlsFailure> {
        let mut to_server = hello;
        let mut established = None;
        for _ in 0..8 {
            let to_client = serve(server, &to_server);
            if to_client.is_empty() {
                break;
            }
            let fed = client.feed(&to_client);
            if let Some(reason) = fed.failure {
                return Err(reason);
            }
            established = established.or(fed.established);
            to_server = fed.outgoing;
            if established.is_some() && to_server.is_empty() {
                break;
            }
        }
        established.ok_or(TlsFailure::Handshake)
    }
}

#[cfg(test)]
mod tests {
    use super::harness::*;
    use super::*;

    const PEER_CA: u32 = VERIFY_PEER_CA;
    const BOTH: u32 = VERIFY_PEER_CA | VERIFY_HOST_NAME;
    const ALL: u32 = BOTH | VERIFY_DATE_CHECK;

    fn run(
        name: &str,
        verify: u32,
        trusted: &[Vec<u8>],
        certificate: &[u8],
    ) -> Result<Established, TlsFailure> {
        let (mut client, hello) = TlsStream::new(name, verify, Vec::new(), &roots(trusted))?;
        handshake(&mut client, &mut server(certificate, &[]), hello)
    }

    #[test]
    fn a_server_the_guest_trusts_completes_and_says_what_it_negotiated() {
        let found = run("localhost", ALL, &[ca_der()], SERVER).unwrap();
        assert_eq!(found.protocol, "TLSv1.3");
        assert!(found.cipher.starts_with("TLS13_"), "{}", found.cipher);
        assert_eq!(found.chain.len(), 1);
        assert_eq!(found.alpn, None);
    }

    #[test]
    fn an_unknown_ca_fails_unless_the_guest_checks_nothing() {
        assert_eq!(
            run("localhost", BOTH, &[], SERVER),
            Err(TlsFailure::UnknownCa),
            "no roots at all"
        );
        assert_eq!(
            run("localhost", PEER_CA, &[vec![1, 2, 3]], SERVER),
            Err(TlsFailure::UnknownCa)
        );
        assert!(
            run("localhost", 0, &[], SERVER).is_ok(),
            "no verification asked for"
        );
    }

    #[test]
    fn the_name_is_checked_only_when_the_guest_asks() {
        assert_eq!(
            run("other.example", BOTH, &[ca_der()], SERVER),
            Err(TlsFailure::NotValidForName)
        );
        assert!(run("other.example", PEER_CA, &[ca_der()], SERVER).is_ok());
    }

    #[test]
    fn the_dates_are_checked_only_when_the_guest_asks() {
        assert_eq!(
            run("localhost", ALL, &[ca_der()], EXPIRED),
            Err(TlsFailure::Expired)
        );
        assert!(run("localhost", BOTH, &[ca_der()], EXPIRED).is_ok());
    }

    #[test]
    fn an_offered_alpn_protocol_is_the_one_negotiated() {
        let (mut client, hello) = TlsStream::new(
            "localhost",
            ALL,
            vec![b"h2".to_vec(), b"http/1.1".to_vec()],
            &roots(&[ca_der()]),
        )
        .unwrap();
        let found = handshake(&mut client, &mut server(SERVER, &[b"http/1.1"]), hello).unwrap();
        assert_eq!(found.alpn, Some(b"http/1.1".to_vec()));
    }

    /// A connection that has finished its handshake.
    fn open() -> (TlsStream, rustls::ServerConnection) {
        let (mut client, hello) =
            TlsStream::new("localhost", ALL, Vec::new(), &roots(&[ca_der()])).unwrap();
        let mut server = server(SERVER, &[]);
        handshake(&mut client, &mut server, hello).unwrap();
        (client, server)
    }

    fn read_all(server: &mut rustls::ServerConnection) -> Vec<u8> {
        let mut out = Vec::new();
        let mut buffer = [0u8; 4096];
        while let Ok(read) = std::io::Read::read(&mut server.reader(), &mut buffer) {
            if read == 0 {
                break;
            }
            out.extend_from_slice(&buffer[..read]);
        }
        out
    }

    #[test]
    fn plaintext_goes_out_encrypted_and_comes_back_decrypted() {
        let (mut client, mut server) = open();
        let wire = client.write(b"ping");
        assert!(
            !wire.windows(4).any(|w| w == b"ping"),
            "the relay never sees it"
        );
        serve(&mut server, &wire);
        assert_eq!(read_all(&mut server), b"ping");

        std::io::Write::write_all(&mut server.writer(), b"pong").unwrap();
        let mut reply = Vec::new();
        while server.wants_write() {
            server.write_tls(&mut reply).unwrap();
        }
        let fed = client.feed(&reply);
        assert_eq!(fed.plaintext, b"pong");
        assert!(fed.failure.is_none() && !fed.closed);
    }

    #[test]
    fn the_servers_close_notify_ends_the_stream_and_ours_is_sent_on_request() {
        let (mut client, mut server) = open();
        server.send_close_notify();
        let mut reply = Vec::new();
        while server.wants_write() {
            server.write_tls(&mut reply).unwrap();
        }
        assert!(client.feed(&reply).closed);

        let (mut client, mut server) = open();
        serve(&mut server, &client.close_notify());
        assert!(read_all(&mut server).is_empty());
    }

    #[test]
    fn a_server_that_speaks_nonsense_fails_the_handshake() {
        let (mut client, _) =
            TlsStream::new("localhost", ALL, Vec::new(), &roots(&[ca_der()])).unwrap();
        let fed = client.feed(&[0x16, 0x03, 0x03, 0x00, 0x04, 1, 2, 3, 4]);
        assert!(fed.failure.is_some());
        assert!(
            client.feed(b"more").failure.is_none(),
            "a failed connection reads no more"
        );
    }

    #[test]
    fn credit_is_the_ciphertext_the_plaintext_read_came_from() {
        let (mut client, mut server) = open();
        std::io::Write::write_all(&mut server.writer(), &[7u8; 1000]).unwrap();
        let mut wire = Vec::new();
        while server.wants_write() {
            server.write_tls(&mut wire).unwrap();
        }
        let fed = client.feed(&wire);
        assert_eq!(fed.plaintext.len(), 1000);
        assert_eq!(fed.credit, 0, "held until the guest reads it");

        let first = client.consumed(400);
        let second = client.consumed(600);
        assert_eq!(
            first + second,
            wire.len() as u32,
            "every byte of ciphertext comes back"
        );
        assert!(first > 0 && first < wire.len() as u32);
        assert_eq!(client.consumed(5), 0, "nothing more is owed");
    }
}
