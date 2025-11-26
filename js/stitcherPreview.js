import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "nv-inpaint.stitcherDebugInfo",

    nodeCreated(node) {
        if (node.comfyClass !== "NV_StitcherDebugInfo") {
            return;
        }

        // Create text widget for displaying preview data
        const textWidget = ComfyWidgets["STRING"](node, "preview_text", ["STRING", { multiline: true }], app).widget;
        textWidget.inputEl.readOnly = true;
        textWidget.inputEl.style.opacity = 0.8;
        textWidget.inputEl.style.fontFamily = "monospace";
        textWidget.inputEl.style.fontSize = "11px";
        textWidget.inputEl.style.minHeight = "200px";
        textWidget.value = "Run workflow to see stitcher debug info...";

        // Set initial size
        node.size = [350, 300];

        // Override onExecuted to capture the returned UI text
        const originalOnExecuted = node.onExecuted;
        node.onExecuted = function(message) {
            if (originalOnExecuted) {
                originalOnExecuted.apply(this, arguments);
            }

            if (message?.text) {
                const text = Array.isArray(message.text) ? message.text[0] : message.text;
                textWidget.value = text;

                // Resize node to fit content
                const lines = text.split('\n').length;
                const minHeight = Math.min(Math.max(lines * 14 + 80, 200), 500);
                if (node.size[1] < minHeight) {
                    node.setSize([Math.max(node.size[0], 350), minHeight]);
                }
            }
        };
    }
});
