//! Each process's handle table, and the objects that go when no process holds them.
//!
//! Values come from one console-wide counter, so every map keyed by a handle stays valid.

use std::collections::BTreeSet;

use crate::cpu::*;

impl Cpu {
    /// Record that the running process received `handle`.
    pub(crate) fn grant_handle(&mut self, handle: u64) {
        *self.process_mut().handles.entry(handle).or_default() += 1;
    }

    /// `svcCloseHandle`: drop one of the running process's copies of `handle`.
    pub(crate) fn close_handle(&mut self, handle: u64) {
        let handles = &mut self.process_mut().handles;
        let Some(copies) = handles.get_mut(&handle) else {
            return;
        };
        *copies -= 1;
        if *copies > 0 {
            return;
        }
        handles.remove(&handle);
        if self.transfer_memories.contains_key(&handle) && !self.held_handles().contains(&handle) {
            self.transfer_memories.remove(&handle);
        }
        if self.service_handles.contains_key(&handle) && !self.held_handles().contains(&handle) {
            self.close_session(handle);
        }
    }

    /// Every handle some live process holds.
    fn held_handles(&self) -> BTreeSet<u64> {
        self.processes
            .values()
            .flat_map(|process| process.handles.keys().copied())
            .collect()
    }

    /// After a process is gone: close the sessions only it held, and drop the events
    /// whose owner is gone and that no live process holds.
    pub(crate) fn release_handles(&mut self, ended: impl IntoIterator<Item = u64>) {
        let held = self.held_handles();
        for handle in ended {
            if self.service_handles.contains_key(&handle) && !held.contains(&handle) {
                self.close_session(handle);
            }
        }
        self.machine
            .transfer_memories
            .retain(|handle, _| held.contains(handle));
        let processes = &self.machine.processes;
        self.machine.events.retain(|handle, event| {
            event
                .owner
                .is_none_or(|owner| processes.contains_key(&owner))
                || held.contains(handle)
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::*;

    #[test]
    fn an_ended_process_takes_its_sessions_and_events_but_not_the_consoles() {
        let mut cpu = Cpu::new();
        cpu.bootstrap();
        let other = cpu.create_process();
        let (session, own, kept) = cpu.in_process(other, |cpu| {
            let session = cpu.alloc_handle();
            cpu.record_handle(session, "lm");
            cpu.grant_handle(session);
            let own = cpu.alloc_event("test:own", false);
            cpu.grant_handle(own);
            let kept = cpu.alloc_event("test:kept", false);
            (session, own, kept)
        });
        let console = cpu.alloc_console_event("test:console", false);
        // The booted process holds a handle to one of the other's events.
        cpu.grant_handle(kept);

        cpu.end_process(other, Ending::Exited);

        assert_eq!(
            cpu.service_name(session),
            None,
            "the session outlived its process"
        );
        assert_eq!(
            cpu.event_signaled(own),
            None,
            "the event outlived its process"
        );
        assert!(
            cpu.event_signaled(kept).is_some(),
            "a held event went with its owner"
        );
        assert!(
            cpu.event_signaled(console).is_some(),
            "a console event went with a process"
        );
    }

    #[test]
    fn closing_the_last_handle_to_a_session_closes_it() {
        let mut cpu = Cpu::new();
        cpu.bootstrap();
        let session = cpu.alloc_handle();
        cpu.record_handle(session, "lm");
        cpu.grant_handle(session);
        cpu.grant_handle(session);

        cpu.close_handle(session);
        assert_eq!(
            cpu.service_name(session),
            Some("lm"),
            "closed with a copy still held"
        );
        cpu.close_handle(session);
        assert_eq!(cpu.service_name(session), None);
    }
}
