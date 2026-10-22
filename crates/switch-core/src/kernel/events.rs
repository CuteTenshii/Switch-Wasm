//! Kernel events, applet messages and service handle bookkeeping.

use crate::cpu::*;

/// A kernel event a service handed the guest a handle to.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Event {
    /// Diagnostics only.
    pub(crate) name: &'static str,
    /// Events start unsignalled.
    pub(crate) signaled: bool,
    /// Whether a successful wait consumes the signal (auto-clear).
    pub(crate) auto_clear: bool,
    /// The process it was made for; `None` for the console's own.
    pub(crate) owner: Option<u64>,
}

/// The AM messages queued for the running applet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppletMessage {
    /// Asks a library applet to exit.
    ExitRequest = 4,
    /// Queued once at startup, then only on a real change.
    FocusStateChanged = 15,
    /// For an applet that called `SetHandlesRequestToDisplay`.
    RequestToDisplay = 41,
    /// The same transition told to an applet rather than an application.
    ChangeIntoForeground = 1,
    /// Docked or undocked; titles re-read `GetOperationMode` on this.
    OperationModeChanged = 30,
    /// Sent alongside `OperationModeChanged`.
    PerformanceModeChanged = 31,
}

impl Cpu {
    /// Allocate an event for the running process; it must reach the guest as a copy handle.
    pub(crate) fn alloc_event(&mut self, name: &'static str, auto_clear: bool) -> u64 {
        self.insert_event(name, auto_clear, Some(self.pid))
    }

    /// An event the console keeps for every process, which outlives any of them.
    pub(crate) fn alloc_console_event(&mut self, name: &'static str, auto_clear: bool) -> u64 {
        self.insert_event(name, auto_clear, None)
    }

    fn insert_event(&mut self, name: &'static str, auto_clear: bool, owner: Option<u64>) -> u64 {
        let handle = self.alloc_handle();
        self.events.insert(
            handle,
            Event {
                name,
                signaled: false,
                auto_clear,
                owner,
            },
        );
        if crate::trace::enabled(crate::trace::Trace::Wait) {
            crate::traceln!("[event] {name} = {handle:#x} auto_clear={auto_clear}");
        }
        handle
    }

    pub(crate) fn event_name(&self, handle: u64) -> Option<&'static str> {
        self.events.get(&handle).map(|event| event.name)
    }

    /// Queue an applet message and wake whatever polls for one.
    pub(crate) fn queue_applet_message(&mut self, message: AppletMessage) {
        self.process_mut().applet_messages.push_back(message as u32);
        if let Some(handle) = self.process().applet_event {
            self.signal_event(handle);
        }
    }

    pub fn operation_mode(&self) -> OperationMode {
        self.operation_mode
    }

    /// Dock or undock while running: queues `OperationModeChanged` and
    /// `PerformanceModeChanged` on a real change.
    pub fn set_operation_mode(&mut self, mode: OperationMode) {
        if self.operation_mode == mode {
            return;
        }
        self.operation_mode = mode;
        // Undocking lifts any touch; republish the sample either way.
        self.set_touch_state(&[]);
        // Default buffer queue geometry; sizes a guest already dequeued are kept.
        let (width, height) = mode.display_size();
        self.display.set_default_size(width, height);
        for pid in self.pids() {
            self.in_process(pid, |cpu| {
                cpu.queue_applet_message(AppletMessage::OperationModeChanged);
                cpu.queue_applet_message(AppletMessage::PerformanceModeChanged);
                if let Some(event) = cpu.process().display_resolution_event {
                    cpu.signal_event(event);
                }
            });
        }
    }

    /// Tell the running application it lost or regained focus to a library applet. An
    /// applet is not told.
    pub(crate) fn set_application_focus(&mut self, in_focus: bool) {
        let process = self.process_mut();
        if !process.applet_is_application || process.in_focus == in_focus {
            return;
        }
        process.in_focus = in_focus;
        self.queue_applet_message(AppletMessage::FocusStateChanged);
    }

    /// Record the proxy kind, which selects the focus message.
    pub(crate) fn set_applet_is_application(&mut self, is_application: bool) {
        self.process_mut().applet_is_application = is_application;
    }

    /// Next AM message; the startup focus transition comes first, once.
    pub(crate) fn next_applet_message(&mut self) -> Option<u32> {
        if !self.process().applet_focus_announced {
            self.process_mut().applet_focus_announced = true;
            return Some(if self.process().applet_is_application {
                AppletMessage::FocusStateChanged as u32
            } else {
                AppletMessage::ChangeIntoForeground as u32
            });
        }
        self.process_mut().applet_messages.pop_front()
    }

    pub(crate) fn has_applet_message(&self) -> bool {
        !self.process().applet_focus_announced || !self.process().applet_messages.is_empty()
    }

    /// Fire an event and wake every parked waiter; each rechecks its own handles.
    pub fn signal_event(&mut self, handle: u64) {
        let Some(event) = self.events.get_mut(&handle) else {
            return;
        };
        // Only a transition wakes waiters.
        if event.signaled {
            return;
        }
        event.signaled = true;
        self.wake_event_waiters();
    }

    /// Wake every thread parked in `svcWaitSynchronization`; each reissues its wait.
    pub(crate) fn wake_event_waiters(&mut self) {
        let threads = self
            .machine
            .processes
            .values_mut()
            .flat_map(|p| &mut p.threads);
        for thread in threads {
            if matches!(thread.state, ThreadState::WaitEvent { .. }) {
                thread.state = ThreadState::Runnable;
            }
        }
    }

    /// Whether `handle` is a fired event or an exited thread; `None` if neither.
    pub(crate) fn waitable_signaled(&self, handle: u64) -> Option<bool> {
        if let Some(signaled) = self.event_signaled(handle) {
            return Some(signaled);
        }
        self.process()
            .threads
            .iter()
            .find(|thread| thread.handle == handle)
            .map(|thread| thread.state == ThreadState::Finished)
    }

    /// Whether `handle` names a fired event; `None` if it is not an event.
    pub fn event_signaled(&self, handle: u64) -> Option<bool> {
        self.events.get(&handle).map(|event| event.signaled)
    }

    /// Consume an auto-clear event's signal after a wait reported it.
    pub(crate) fn consume_event(&mut self, handle: u64) {
        if let Some(event) = self.events.get_mut(&handle) {
            if event.auto_clear {
                event.signaled = false;
            }
        }
    }

    pub(crate) fn clear_event(&mut self, handle: u64) {
        if let Some(event) = self.events.get_mut(&handle) {
            event.signaled = false;
        }
    }

    /// `svcResetSignal`: returns whether the event was signalled; unmodelled handles count as signalled.
    pub(crate) fn reset_signal(&mut self, handle: u64) -> bool {
        match self.events.get_mut(&handle) {
            Some(event) => std::mem::replace(&mut event.signaled, false),
            None => true,
        }
    }

    /// Bind a handle to a service name directly, for tests.
    pub fn register_service_handle(&mut self, handle: u64, name: &str) {
        self.record_handle(handle, name);
    }

    /// The interface a domain object id on `handle` names, or `None` once closed.
    pub fn domain_interface_name(&self, handle: u64, object_id: u32) -> Option<String> {
        self.domain_interface(handle, object_id)
            .map(|s| s.to_owned())
    }

    /// Debug: dump the fake-handle to service-name map.
    pub fn service_handles_snapshot(&self) -> Vec<(u64, String)> {
        let mut v: Vec<(u64, String)> = self
            .service_handles
            .iter()
            .map(|(&h, s)| (h, s.clone()))
            .collect();
        v.sort();
        v
    }
}
