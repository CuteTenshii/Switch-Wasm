//! Kernel events, applet messages and service handle bookkeeping.

use super::*;

/// The AM messages queued for the running applet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppletMessage {
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
    /// Allocate an event handle; it must reach the guest as a copy handle.
    pub(crate) fn alloc_event(&mut self, name: &'static str, auto_clear: bool) -> u64 {
        let handle = self.alloc_handle();
        self.events.insert(
            handle,
            Event {
                name,
                signaled: false,
                auto_clear,
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
    pub(super) fn queue_applet_message(&mut self, message: AppletMessage) {
        self.applet_messages.push_back(message as u32);
        if let Some(handle) = self.applet_event {
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
        self.queue_applet_message(AppletMessage::OperationModeChanged);
        self.queue_applet_message(AppletMessage::PerformanceModeChanged);
        if let Some(event) = self.display_resolution_event {
            self.signal_event(event);
        }
    }

    /// Record the proxy kind, which selects the focus message.
    pub(super) fn set_applet_is_application(&mut self, is_application: bool) {
        self.applet_is_application = is_application;
    }

    /// Next AM message; the startup focus transition comes first, once.
    pub(super) fn next_applet_message(&mut self) -> Option<u32> {
        if !self.applet_focus_announced {
            self.applet_focus_announced = true;
            return Some(if self.applet_is_application {
                AppletMessage::FocusStateChanged as u32
            } else {
                AppletMessage::ChangeIntoForeground as u32
            });
        }
        self.applet_messages.pop_front()
    }

    pub(super) fn has_applet_message(&self) -> bool {
        !self.applet_focus_announced || !self.applet_messages.is_empty()
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
    pub(super) fn wake_event_waiters(&mut self) {
        for thread in &mut self.threads {
            if matches!(thread.state, ThreadState::WaitEvent { .. }) {
                thread.state = ThreadState::Runnable;
            }
        }
    }

    /// Whether `handle` is a fired event or an exited thread; `None` if neither.
    pub(super) fn waitable_signaled(&self, handle: u64) -> Option<bool> {
        if let Some(signaled) = self.event_signaled(handle) {
            return Some(signaled);
        }
        self.threads
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
