//! Programs as the loader sees them: an ExeFS's modules and `main.npdm`, booted into
//! the running process or into a new one.

use std::rc::Rc;

use crate::cpu::*;
use crate::nca::Nca;
use crate::npdm::Npdm;
use crate::nsp::Pfs0;
use crate::source::ByteSource;
use crate::trace::Level;
use crate::{Error, Result};

/// ExeFS modules in load order.
const MODULE_ORDER: &[&str] = &[
    "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5", "subsdk6",
    "subsdk7", "subsdk8", "subsdk9", "sdk",
];

/// A program's ExeFS, parsed.
pub struct Program {
    pub program_id: u64,
    exefs: Vec<u8>,
    pfs0: Pfs0,
}

impl Program {
    pub fn parse(program_id: u64, exefs: Vec<u8>) -> Result<Program> {
        let pfs0 = Pfs0::parse(&exefs)?;
        Ok(Program {
            program_id,
            exefs,
            pfs0,
        })
    }

    /// The modules present, in load order.
    pub fn modules(&self) -> Vec<(&'static str, &[u8])> {
        MODULE_ORDER
            .iter()
            .filter_map(|&name| {
                let file = self.pfs0.find(name)?;
                let start = file.offset as usize;
                Some((name, &self.exefs[start..start + file.size as usize]))
            })
            .collect()
    }

    pub fn is_64_bit(&self) -> bool {
        Npdm::is_64_bit_of(&self.pfs0, &self.exefs)
    }

    pub fn system_resource_size(&self) -> u32 {
        Npdm::system_resource_size_of(&self.pfs0, &self.exefs)
    }
}

/// A program on the NAND the kernel can start on its own, with sources that already decrypt.
#[derive(Debug, Clone)]
pub struct Installed {
    pub nca: Nca,
    /// The ExeFS section, decrypted.
    pub exefs: Rc<dyn ByteSource>,
    pub exefs_index: usize,
    pub romfs: Option<Rc<dyn ByteSource>>,
}

impl Cpu {
    pub fn install_program(&mut self, installed: Installed) {
        self.installed.insert(installed.nca.program_id, installed);
    }

    pub(crate) fn is_installed(&self, program_id: u64) -> bool {
        self.installed.contains_key(&program_id)
    }

    /// Take `program`'s identity and `main.npdm` settings for the running process.
    fn adopt_program(&mut self, program: &Program) -> Result<()> {
        let modules = program.modules();
        self.log(
            Level::Debug,
            &format!(
                "[exefs] entries: {}, loading: {}",
                program
                    .pfs0
                    .files
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                modules
                    .iter()
                    .map(|(name, _)| *name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
        if !modules.iter().any(|(name, _)| *name == "main") {
            return Err(Error::Cpu("no 'main' executable in this ExeFS".into()));
        }
        self.set_program_id(program.program_id);
        let system_resource = program.system_resource_size();
        self.log(
            Level::Debug,
            &format!(
                "[npdm] system resource {system_resource:#x}: {}",
                if system_resource == 0 {
                    "plain heap"
                } else {
                    "virtual address memory"
                }
            ),
        );
        self.set_system_resource_size(system_resource);
        if let Some(priority) = Npdm::main_thread_priority_of(&program.pfs0, &program.exefs) {
            self.set_main_thread_priority(priority);
        }
        if let Some(core) = Npdm::main_thread_core_of(&program.pfs0, &program.exefs) {
            self.set_main_thread_core(core);
        }
        if let Some(mask) = Npdm::core_mask_of(&program.pfs0, &program.exefs) {
            self.set_process_core_mask(mask);
        }
        Ok(())
    }

    /// Boot `program` in the running process. Returns where each module landed, entry first.
    pub fn boot_program(&mut self, program: &Program) -> Result<Vec<crate::nso::LoadedNso>> {
        self.adopt_program(program)?;
        if !program.is_64_bit() {
            self.log(
                Level::Info,
                "[npdm] AArch32 title: running the A32 interpreter",
            );
            self.set_mode(ExecMode::A32);
        }
        self.boot_retail_program(&program.modules())
    }

    /// Start installed program `program_id` as a new process beside the running one.
    pub(crate) fn spawn_installed(&mut self, program_id: u64) -> Result<u64> {
        let installed = self
            .installed
            .get(&program_id)
            .cloned()
            .ok_or_else(|| Error::Cpu(format!("{program_id:016x} is not installed")))?;
        let exefs = installed
            .nca
            .pfs0_from_section(&*installed.exefs, installed.exefs_index)?;
        let program = Program::parse(program_id, exefs)?;
        if !program.is_64_bit() {
            return Err(Error::Cpu(format!(
                "{program_id:016x} is a 32-bit program, which only runs as the booted title"
            )));
        }
        let pid = self.create_process();
        let entry = self.in_process(pid, |cpu| -> Result<u32> {
            cpu.adopt_program(&program)?;
            if let Some(romfs) = installed.romfs {
                cpu.set_romfs_source(Box::new(romfs));
            }
            let loaded = cpu.load_modules(&program.modules())?;
            Ok(loaded[0].entry)
        });
        match entry {
            Ok(entry) => {
                self.start_main_thread(pid, entry, 0);
                Ok(pid)
            }
            Err(e) => {
                self.machine.processes.remove(&pid);
                Err(e)
            }
        }
    }
}
