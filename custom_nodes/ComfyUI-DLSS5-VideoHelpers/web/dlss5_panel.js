// DLSS5 Control Panel: applies the panel's toggles to the rest of the graph.
//  - Batch folder on  -> the single-video pair (Load Video + Render Video) is muted, the batch pair runs.
//    Batch folder off -> the batch pair (Batch Folder + Render Folder) is muted.
//  - Each feature toggle drives the hidden switch widget on its settings node and greys out
//    that node's settings while the feature is off, so it is obvious what is in use.
import { app } from "../../scripts/app.js";

const ALWAYS = 0;
const NEVER = 2; // muted: skipped when the prompt is queued
const PREVIEW_SECONDS = 5;

// Widgets the panel drives (plus one rarely needed safety switch): never shown on the nodes.
const HIDDEN_WIDGETS = {
    DLSS5PreprocessSettings: ["clean_enabled", "interpolate_fps", "target_fps"],
    DLSS5HDRSettings: ["enabled"],
    DLSS5RenderVideo: ["preview_seconds", "verify_neural_rendering"],
    DLSS5BatchEnhanceFolder: ["preview_seconds", "verify_neural_rendering"],
};
// Settings nodes are sized to their content; render nodes keep their room for the preview.
const FIT_TO_CONTENT = ["DLSS5ControlPanel", "DLSS5PreprocessSettings", "DLSS5HDRSettings", "DLSS5BatchFolderInput"];

function tidyNode(node) {
    const type = node.comfyClass ?? node.type;
    const hidden = HIDDEN_WIDGETS[type];
    if (hidden) {
        for (const widget of node.widgets ?? []) {
            if (hidden.includes(widget.name)) widget.hidden = true;
        }
    }
    if (FIT_TO_CONTENT.includes(type) && typeof node.computeSize === "function") {
        node.setSize([node.size[0], node.computeSize()[1]]);
    }
}

const graphNodes = (graph) => graph?._nodes ?? graph?.nodes ?? [];
const widgetOf = (node, name) => node.widgets?.find((w) => w.name === name);
const valueOf = (node, name) => !!widgetOf(node, name)?.value;

function setValue(node, name, value) {
    const widget = widgetOf(node, name);
    if (widget && widget.value !== value) {
        widget.value = value;
        widget.callback?.(value);
    }
}

function setDisabled(node, disabled, except = []) {
    for (const widget of node.widgets ?? []) {
        if (!except.includes(widget.name)) widget.disabled = disabled;
    }
}

function setDisabledNames(node, names, disabled) {
    for (const widget of node.widgets ?? []) {
        if (names.includes(widget.name)) widget.disabled = disabled;
    }
}

function setMode(node, mode) {
    if (!node || node.mode === mode) return;
    if (typeof node.changeMode === "function") node.changeMode(mode);
    else node.mode = mode;
}

function upstream(graph, node, inputName) {
    const input = node.inputs?.find((i) => i.name === inputName);
    if (input?.link == null) return null;
    const link = graph.links?.get?.(input.link) ?? graph.links?.[input.link];
    return link ? graph.getNodeById(link.origin_id) : null;
}

function applyPanel(panel) {
    const graph = panel.graph ?? app.graph;
    const batch = valueOf(panel, "batch_folder");
    const deblur = valueOf(panel, "rtx_deblur");
    // 60 FPS and 30 FPS are exclusive (the toggle callbacks keep only one on); 60 wins if both load on.
    const targetFps = valueOf(panel, "fps_60") ? 60 : valueOf(panel, "fps_30") ? 30 : 0;
    const hdr = valueOf(panel, "hdr");
    const preview = valueOf(panel, "quick_preview") ? PREVIEW_SECONDS : 0;

    for (const node of graphNodes(graph)) {
        switch (node.comfyClass ?? node.type) {
            case "DLSS5RenderVideo":
                setMode(node, batch ? NEVER : ALWAYS);
                setMode(upstream(graph, node, "video"), batch ? NEVER : ALWAYS);
                setValue(node, "preview_seconds", preview);
                break;
            case "DLSS5BatchEnhanceFolder":
                setMode(node, batch ? ALWAYS : NEVER);
                setMode(upstream(graph, node, "folder"), batch ? ALWAYS : NEVER);
                setValue(node, "preview_seconds", preview);
                break;
            case "DLSS5PreprocessSettings":
                setValue(node, "clean_enabled", deblur);
                setValue(node, "interpolate_fps", targetFps > 0);
                if (targetFps) setValue(node, "target_fps", targetFps);
                setDisabledNames(node, ["rtx_clean", "rtx_clean_strength"], !deblur);
                setDisabledNames(node, ["scene_cut_threshold"], !targetFps);
                break;
            case "DLSS5HDRSettings":
                setValue(node, "enabled", hdr);
                setDisabled(node, !hdr, ["enabled"]);
                break;
        }
    }
    graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function applyAll() {
    for (const node of graphNodes(app.graph)) {
        if ((node.comfyClass ?? node.type) === "DLSS5ControlPanel") applyPanel(node);
    }
}

app.registerExtension({
    name: "DLSS5.ControlPanel",
    nodeCreated(node) {
        tidyNode(node);
        if ((node.comfyClass ?? node.type) !== "DLSS5ControlPanel") return;
        const exclusive = { fps_60: "fps_30", fps_30: "fps_60" };
        for (const widget of node.widgets ?? []) {
            const original = widget.callback;
            widget.callback = function (...args) {
                const result = original?.apply(this, args);
                const other = exclusive[widget.name] && widgetOf(node, exclusive[widget.name]);
                if (widget.value && other?.value) other.value = false;
                applyPanel(node);
                return result;
            };
        }
        // A panel dropped into an existing graph takes effect straight away.
        setTimeout(() => node.graph && applyPanel(node), 0);
    },
    afterConfigureGraph() {
        // Saved sizes are restored after nodeCreated, so tidy again once the graph is loaded.
        for (const node of graphNodes(app.graph)) tidyNode(node);
        applyAll();
    },
});
