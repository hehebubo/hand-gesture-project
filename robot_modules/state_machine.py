from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import carb

from joint_def import JointMotionController

if TYPE_CHECKING:
    from robot_modules.ros_interface import HandTargetIKBridge


class Mode(enum.Enum):
    LOOP = "A"
    PAUSE = "B"
    RETURNING = "C_return"
    HOME = "C_hold"
    ROS_CONTROL = "D_ros"


class LoopRoutine:
    def __init__(self, controller: JointMotionController, keyframes):
        self._controller = controller
        self._targets = [controller.clamp(k.pose) for k in keyframes]
        self._durations = [max(1e-3, k.duration) for k in keyframes]
        self._segment = 0
        self._elapsed = 0.0
        self._segment_start = controller.current_pose()
        self._active = False

    def reset_cycle(self):
        self._segment = 0
        self._elapsed = 0.0
        self._segment_start = self._controller.current_pose()

    def resume(self):
        self._active = True

    def pause(self):
        self._active = False

    def update(self, dt: float):
        if not self._active or not self._targets:
            return

        target = self._targets[self._segment]
        duration = self._durations[self._segment]
        alpha = min(1.0, self._elapsed / duration)
        pose = [
            self._segment_start[i] + (target[i] - self._segment_start[i]) * alpha
            for i in range(self._controller.dof_count)
        ]
        self._controller.apply_pose(pose)
        self._elapsed += dt

        if self._elapsed >= duration:
            self._segment = (self._segment + 1) % len(self._targets)
            self._elapsed = 0.0
            self._segment_start = self._controller.current_pose()


class ReturnHomeAction:
    def __init__(self, controller: JointMotionController, home_pose, duration: float = 2.0):
        self._controller = controller
        self._home = controller.clamp(home_pose)
        self._duration = max(1e-3, duration)
        self._active = False
        self._elapsed = 0.0
        self._start = self._home

    def start(self):
        self._active = True
        self._elapsed = 0.0
        self._start = self._controller.current_pose()

    def stop(self):
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def update(self, dt: float) -> bool:
        if not self._active:
            return True

        alpha = min(1.0, self._elapsed / self._duration)
        pose = [
            self._start[i] + (self._home[i] - self._start[i]) * alpha
            for i in range(self._controller.dof_count)
        ]
        self._controller.apply_pose(pose)
        self._elapsed += dt

        if self._elapsed >= self._duration:
            self._controller.apply_pose(self._home)
            self._active = False
            return True
        return False


class ModeManager:
    def __init__(
        self,
        controller,
        home_pose,
        routine: LoopRoutine,
        ik_bridge: "HandTargetIKBridge | None" = None,
    ):
        self._controller = controller
        self._home_pose = controller.clamp(home_pose)
        self._routine = routine
        self._return_home = ReturnHomeAction(controller, self._home_pose, duration=2.5)
        self._mode = Mode.HOME
        self._hold_pose = self._home_pose[:]

        # ROS 2 can be handled externally; this manager only switches modes.
        self._ros_enabled = True
        self._ros_sub = None
        self._ik_bridge = ik_bridge

    def handle_mode_request(self, requested: Mode):
        if requested == Mode.LOOP:
            self._enter_loop()
        elif requested == Mode.PAUSE:
            self._enter_pause()
        elif requested == Mode.RETURNING:
            self._enter_return()
        elif requested == Mode.ROS_CONTROL:
            self._enter_ros_control()

    def update(self, dt: float):
        if self._mode == Mode.LOOP:
            self._routine.update(dt)
        elif self._mode == Mode.PAUSE:
            self._controller.apply_pose(self._hold_pose)
        elif self._mode == Mode.RETURNING:
            finished = self._return_home.update(dt)
            if finished:
                self._mode = Mode.HOME
                self._hold_pose = self._home_pose[:]

        elif self._mode == Mode.ROS_CONTROL:
            if self._ik_bridge:
                self._ik_bridge.update()

        if self._mode == Mode.HOME:
            self._controller.apply_pose(self._home_pose)

    @property
    def current_mode(self) -> Mode:
        return self._mode

    def _enter_loop(self):
        self._routine.reset_cycle()
        self._return_home.stop()
        self._mode = Mode.LOOP
        self._routine.resume()
        print("[mode] A: Loop")

    def _enter_pause(self):
        self._return_home.stop()
        self._routine.pause()
        self._hold_pose = self._controller.current_pose()
        self._mode = Mode.PAUSE
        print("[mode] B: Pause")

    def _enter_return(self):
        self._routine.pause()
        self._return_home.start()
        self._mode = Mode.RETURNING
        print("[mode] C: Home")

    def _enter_ros_control(self):
        self._routine.pause()
        self._return_home.stop()
        self._mode = Mode.ROS_CONTROL
        print("[mode] D: ROS Control")


class KeyboardModeSwitcher:
    def __init__(self, mode_manager: ModeManager):
        self._mode_manager = mode_manager
        self._input = carb.input.acquire_input_interface()
        self._subscription = self._input.subscribe_to_input_events(
            self._on_input_event, order=-100
        )
        self._debug_left = 8

    def shutdown(self):
        if self._subscription is not None:
            self._input.unsubscribe_to_input_events(self._subscription)
            self._subscription = None

    def _on_input_event(self, event, *_args, **_kwargs) -> bool:
        """
        Compatible with Isaac Sim 5.0 keyboard event layouts.
        """
        print("[raw-event]", event)

        def _extract_key(evt):
            kb = getattr(evt, "keyboard", None)
            if kb is not None:
                return getattr(kb, "input", None) or getattr(kb, "key", None)
            evtev = getattr(evt, "event", None)
            if evtev is not None:
                return (
                    getattr(evtev, "input", None)
                    or getattr(evtev, "key", None)
                    or getattr(evtev, "character", None)
                )
            return (
                getattr(evt, "input", None)
                or getattr(evt, "key", None)
                or getattr(evt, "character", None)
            )

        def _extract_type(evt):
            kb = getattr(evt, "keyboard", None)
            if kb is not None:
                return getattr(kb, "type", None)
            evtev = getattr(evt, "event", None)
            if evtev is not None:
                return getattr(evtev, "type", None)
            return getattr(evt, "type", None)

        def _extract_value(evt):
            kb = getattr(evt, "keyboard", None)
            if kb is not None:
                return getattr(kb, "value", None)
            evtev = getattr(evt, "event", None)
            if evtev is not None:
                return getattr(evtev, "value", None)
            return getattr(evt, "value", None)

        key = _extract_key(event)
        event_type = _extract_type(event)
        key_value = _extract_value(event)

        try:
            kb = getattr(event, "keyboard", None)
            evtev = getattr(event, "event", None)
            if self._debug_left > 0:
                attrs = {}
                for name in ("device", "device_id", "device_type", "flags", "timestamp"):
                    if hasattr(event, name):
                        attrs[name] = getattr(event, name)
                attrs_list = [a for a in dir(event) if not a.startswith("_")]
                ev_dir = [a for a in dir(evtev) if not a.startswith("_")] if evtev else None
                print(
                    "[event-debug]",
                    "key=", key,
                    "type=", event_type,
                    "value=", key_value,
                    "kbd.input=", getattr(kb, "input", None) if kb else None,
                    "kbd.key=", getattr(kb, "key", None) if kb else None,
                    "kbd.type=", getattr(kb, "type", None) if kb else None,
                    "kbd.value=", getattr(kb, "value", None) if kb else None,
                    "ev.event=", evtev,
                    "ev.key=", getattr(evtev, "key", None) if evtev else None,
                    "ev.input=", getattr(evtev, "input", None) if evtev else None,
                    "ev.type=", getattr(evtev, "type", None) if evtev else None,
                    "ev.value=", getattr(evtev, "value", None) if evtev else None,
                    "ev.character=", getattr(evtev, "character", None) if evtev else None,
                    "evt.input=", getattr(event, "input", None),
                    "evt.key=", getattr(event, "key", None),
                    "evt.character=", getattr(event, "character", None),
                    "attrs=", attrs,
                    "dir=", attrs_list,
                    "ev.dir=", ev_dir,
                )
                self._debug_left -= 1
        except Exception as exc:
            print("[event-debug] print failed:", exc)

        def _match(k, target_enum):
            if k is None:
                return False
            if k == target_enum:
                return True
            if isinstance(k, int):
                return (
                    k == int(target_enum)
                    or k == ord(str(int(int(target_enum) - int(carb.input.KeyboardInput.KEY_0))))
                )
            if isinstance(k, str):
                key_str = k.lower()
                target_str = str(target_enum).lower()
                if key_str == target_str or target_str in key_str:
                    return True
                if len(k) == 1 and k.isdigit():
                    return int(k) == int(target_enum) - int(carb.input.KeyboardInput.KEY_0)
                return False
            return False

        press_types = {
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        }
        if event_type is not None and event_type not in press_types:
            return True
        if key_value is not None and key_value <= 0:
            return True
        if key is None:
            return True

        if _match(key, carb.input.KeyboardInput.KEY_1):
            self._mode_manager.handle_mode_request(Mode.LOOP)
            print("[key] 1 -> Loop")
            return False
        if _match(key, carb.input.KeyboardInput.KEY_2):
            self._mode_manager.handle_mode_request(Mode.PAUSE)
            print("[key] 2 -> Pause")
            return False
        if _match(key, carb.input.KeyboardInput.KEY_3):
            self._mode_manager.handle_mode_request(Mode.RETURNING)
            print("[key] 3 -> Home")
            return False
        if _match(key, carb.input.KeyboardInput.KEY_4):
            self._mode_manager.handle_mode_request(Mode.ROS_CONTROL)
            print("[key] 4 -> ROS Control")
            return False
        return True
