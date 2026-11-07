// H3 Reference Director - Speed Mode: fill in the step count of the selected mode.
//
// The node works without this file: steps = 0 means "use the mode's own count" (20 / 8 / 4).
// This script only makes the field SHOW that number, and refills it when the mode is changed.
// A number typed by hand is kept until the mode is changed again.
import { app } from "/scripts/app.js";

const NODE_TYPE = "H3RDSpeedMode";

function modeSteps(mode) {
    const m = String(mode ?? "");
    if (m.startsWith("OFF")) return 20;
    if (m.includes("8-step")) return 8;
    if (m.includes("4-step")) return 4;
    return 0;
}

function widgetsOf(node) {
    const list = node?.widgets ?? [];
    return {
        mode: list.find((w) => w.name === "speed_mode"),
        steps: list.find((w) => w.name === "steps"),
    };
}

function setSteps(node, steps, value) {
    if (!steps || !(value > 0)) return;
    steps.value = value;
    node?.setDirtyCanvas?.(true, true);
}

// 0 means "automatic": show the real number instead.
function fillIfAutomatic(node) {
    const { mode, steps } = widgetsOf(node);
    if (!mode || !steps) return;
    if (Number(steps.value) === 0) setSteps(node, steps, modeSteps(mode.value));
}

function hook(node) {
    if (node?.__h3rdSpeedHooked) return;
    const { mode, steps } = widgetsOf(node);
    if (!mode || !steps) return;
    node.__h3rdSpeedHooked = true;
    const original = mode.callback;
    mode.callback = function (value, ...rest) {
        const result = original ? original.apply(this, [value, ...rest]) : undefined;
        setSteps(node, steps, modeSteps(value));   // a new mode always starts at its own count
        return result;
    };
}

app.registerExtension({
    name: "H3RD.SpeedMode.AutoSteps",
    nodeCreated(node) {
        if (node?.comfyClass !== NODE_TYPE && node?.type !== NODE_TYPE) return;
        try {
            hook(node);
            setTimeout(() => { try { fillIfAutomatic(node); } catch (e) { console.warn("[H3RD] speed mode fill", e); } }, 0);
        } catch (e) { console.warn("[H3RD] speed mode hook", e); }
    },
    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_TYPE && node?.type !== NODE_TYPE) return;
        try { hook(node); fillIfAutomatic(node); } catch (e) { console.warn("[H3RD] speed mode load", e); }
    },
});
